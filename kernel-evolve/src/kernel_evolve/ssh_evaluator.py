"""SSH-based TPU kernel evaluation. Runs evaluate.py on a remote TPU-VM over SSH.

Drop-in alternative to KubeEvaluator for TPU-VM setups (e.g. a v6e-16 slice)
that have no GKE cluster. Instead of a K8s Job + ConfigMap + GCS bucket, this:

  1. writes the base64 payload to a remote temp file on every host (piped over ssh),
  2. co-launches `docker/evaluate.py` on all hosts with the v6e roofline env + IR
     dump flags (evaluate.py sets its own dump flags and preserves existing env),
  3. parses `EVAL_RESULT:` lines from the primary host's stdout (same wire format
     as KubeEvaluator),
  4. scp's artifacts (llo_final.txt / hlo_post_opt.txt / trace_events.json) back
     from ARTIFACTS_LOCAL_DIR on the primary host — no GCS involved.

Multi-host slices: a v6e-16 is 4 hosts sharing one ICI domain. libtpu does a
slice-wide init barrier, so evaluate.py MUST be co-launched on every host at
once — a single-host invocation hangs forever waiting for its peers. Set
`hosts` to all worker IPs; the first is the primary (its stdout + artifacts are
the result). The kernel template is single-device, so each host independently
runs the same kernel on its local chip and prints an identical EVAL_RESULT; we
just read the primary's. Single-host slices (v6e-1/-4/-8): leave `hosts` empty
and set `host`.

The remote VM must have the repo checked out at `remote_repo_dir` and the
kernel-evolve package installed into `python_bin`'s env
(`uv pip install -e kernel-evolve/` or equivalent), plus a working TPU runtime.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from kernel_evolve.evaluator import (
  BatchEvalRequest,
  BatchEvalResult,
  EvalRequest,
  EvalResult,
  Evaluator,
)


@dataclass
class SSHConfig:
  host: str = ""
  # All worker hosts of an ICI slice, co-launched together; the first is the
  # primary (its stdout + artifacts are the result). Empty → single-host [host].
  hosts: list[str] = field(default_factory=list)
  user: str = "cloud-user"
  ssh_key: str = ""
  # Where the repo is checked out on the VM (contains kernel-evolve/). Interpolated
  # unquoted into the remote command, so "~/" expands to the remote user's home.
  remote_repo_dir: str = "~/dracospark"
  # Python that has kernel-evolve + jax/libtpu installed.
  python_bin: str = "~/tpu-venv/bin/python"
  remote_tmp: str = "/tmp/kernel_eval"
  remote_artifacts_dir: str = "/tmp/kernel_eval_artifacts"
  # Where scp'd artifacts land locally (per variant under artifacts/<variant_id>/).
  local_artifacts_dir: str = "artifacts"
  timeout: int = 1800
  # v6e (Trillium) roofline — passed through as env to evaluate.py.
  peak_flops: float = 918e12
  peak_hbm_bw: float = 1759e9
  hbm_capacity_gb: float = 32.0
  vmem_capacity_mib: float = 128.0
  # Extra libtpu flags appended (evaluate.py already sets the dump flags).
  extra_libtpu_init_args: str = ""
  # Extra env for evaluate.py (e.g. TPU_PROCESS_BOUNDS single-host overrides).
  env: dict[str, str] = field(default_factory=dict)
  cleanup: bool = True
  ssh_opts: tuple[str, ...] = (
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "BatchMode=yes",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ConnectTimeout=15",
  )

  def all_hosts(self) -> list[str]:
    hosts = [h for h in self.hosts if h]
    if hosts:
      return hosts
    return [self.host] if self.host else []

  def primary(self) -> str:
    hosts = self.all_hosts()
    if not hosts:
      raise ValueError("SSHConfig has no host/hosts configured")
    return hosts[0]


# Top-level directories that never hold evaluator scratch space. /dev is listed, and
# /dev/shm is admitted explicitly, because a tmpfs there is the one exception.
_SYSTEM_TOP_LEVEL = frozenset(
  {
    "bin",
    "boot",
    "dev",
    "etc",
    "home",
    "lib",
    "lib32",
    "lib64",
    "libx32",
    "media",
    "mnt",
    "opt",
    "proc",
    "root",
    "run",
    "sbin",
    "srv",
    "sys",
    "usr",
    "var",
  }
)


def _is_scratch_path(path: PurePosixPath) -> bool:
  """Whether a configured directory may be treated as evaluator-owned scratch space."""
  if not path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
    return False
  if path == PurePosixPath("/dev/shm") or PurePosixPath("/dev/shm") in path.parents:
    return True
  return path.parts[1] not in _SYSTEM_TOP_LEVEL


class SSHEvaluator(Evaluator):
  """Evaluates kernel variants by running evaluate.py on a remote TPU-VM via SSH."""

  def __init__(self, config: SSHConfig):
    self._config = config

  # ── low-level ssh/scp plumbing ──────────────────────────────────────────

  def _key_args(self) -> list[str]:
    return ["-i", self._config.ssh_key] if self._config.ssh_key else []

  def _target(self, host: str) -> str:
    return f"{self._config.user}@{host}"

  async def _run(self, *args: str, stdin: str | None = None, timeout: int | None = None) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
      *args,
      stdin=asyncio.subprocess.PIPE if stdin is not None else None,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    try:
      stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=stdin.encode() if stdin is not None else None),
        timeout=timeout,
      )
    except asyncio.TimeoutError:
      proc.kill()
      return "", f"command timed out after {timeout}s", 124
    return stdout.decode(errors="replace"), stderr.decode(errors="replace"), proc.returncode

  async def _ssh(
    self, remote_cmd: str, host: str | None = None, stdin: str | None = None, timeout: int | None = None
  ) -> tuple[str, str, int]:
    host = host or self._config.primary()
    args = ["ssh", *self._key_args(), *self._config.ssh_opts, self._target(host), remote_cmd]
    return await self._run(*args, stdin=stdin, timeout=timeout)

  async def _ssh_all(
    self, remote_cmd: str, stdin: str | None = None, timeout: int | None = None
  ) -> dict[str, tuple[str, str, int]]:
    """Co-launch the same command on every host concurrently (barrier-safe)."""
    hosts = self._config.all_hosts()
    results = await asyncio.gather(*(self._ssh(remote_cmd, host=h, stdin=stdin, timeout=timeout) for h in hosts))
    return dict(zip(hosts, results))

  async def _scp_from(self, remote_path: str, local_path: str, host: str | None = None) -> tuple[str, str, int]:
    host = host or self._config.primary()
    args = ["scp", "-r", *self._key_args(), *self._config.ssh_opts, f"{self._target(host)}:{remote_path}", local_path]
    return await self._run(*args, timeout=300)

  # ── env / command construction ──────────────────────────────────────────

  def _env_prefix(self, artifacts_dir: str) -> str:
    c = self._config
    env = {
      "JAX_PLATFORMS": "tpu,cpu",
      "ENABLE_PJRT_COMPATIBILITY": "true",
      "PEAK_FLOPS": repr(c.peak_flops),
      "PEAK_HBM_BW": repr(c.peak_hbm_bw),
      "HBM_CAPACITY_GB": repr(c.hbm_capacity_gb),
      "VMEM_CAPACITY_MIB": repr(c.vmem_capacity_mib),
      "ARTIFACTS_LOCAL_DIR": artifacts_dir,
    }
    if c.extra_libtpu_init_args:
      env["LIBTPU_INIT_ARGS"] = c.extra_libtpu_init_args
    env.update(c.env)
    return " ".join(f"{k}={v}" for k, v in env.items())

  def _eval_cmd(self, payload_file: str, artifacts_dir: str) -> str:
    c = self._config
    return (
      f"cd {c.remote_repo_dir}/kernel-evolve && "
      f"{self._env_prefix(artifacts_dir)} "
      f"{c.python_bin} docker/evaluate.py --eval-payload-file {payload_file}"
    )

  @staticmethod
  def _parse_results(logs: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in logs.splitlines():
      if "EVAL_RESULT:" in line:
        raw = line.split("EVAL_RESULT:", 1)[1].strip()
        try:
          data = json.loads(raw)
        except json.JSONDecodeError:
          print(f"Malformed EVAL_RESULT: {raw[:200]}", file=sys.stderr)
          continue
        out[data.get("variant_id", "unknown")] = data
    return out

  async def _pull_artifacts(self, result: dict) -> None:
    remote_dir = (result.get("metadata") or {}).get("artifacts_local_dir")
    if not remote_dir:
      return
    vid = result.get("variant_id", "unknown")
    local = Path(self._config.local_artifacts_dir) / vid
    local.mkdir(parents=True, exist_ok=True)
    # copy the *contents* of the remote variant dir into local/<vid>/
    _, stderr, rc = await self._scp_from(f"{remote_dir}/.", str(local))
    if rc != 0:
      print(f"Artifact scp for {vid} failed: {stderr.strip()}", file=sys.stderr)

  async def _write_payload(self, payload: str, remote_file: str) -> int:
    """Stage the payload on every host (no shared FS across an ICI slice)."""
    results = await self._ssh_all(
      f"mkdir -p {self._config.remote_tmp} && cat > {remote_file}",
      stdin=payload,
      timeout=120,
    )
    rc = 0
    for host, (_, stderr, host_rc) in results.items():
      if host_rc != 0:
        print(f"Failed to write payload on {host}: {stderr.strip()}", file=sys.stderr)
        rc = host_rc
    return rc

  def _cleanup_roots(self) -> tuple[tuple[PurePosixPath, ...], tuple[PurePosixPath, ...]]:
    """Directories the evaluator owns, and the roots a target may live under.

    A configured directory is only treated as owned when it is plausible scratch space.
    Configuring ``remote_artifacts_dir`` as a home or system directory would otherwise
    make that directory removable in full.
    """
    owned = tuple(
      path
      for path in (PurePosixPath(p) for p in (self._config.remote_tmp, self._config.remote_artifacts_dir) if p)
      if _is_scratch_path(path)
    )
    shared = (PurePosixPath("/tmp"), PurePosixPath("/dev/shm"))
    return owned, owned + shared

  def _is_removable(self, raw: str) -> bool:
    """Whether ``rm -rf`` may be pointed at this remote path.

    An evaluator-owned directory may be removed whole; anything else must be a named
    descendant of one of those roots or of a scratch root. A shared root itself, a
    relative path, a ``..`` component, and a top-level directory are all refused.
    """
    path = PurePosixPath(raw)
    owned, roots = self._cleanup_roots()
    if not path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
      return False
    if path in owned:
      return True
    return path not in roots and any(root in path.parents for root in roots)

  def _guarded_rm_command(self, targets: list[str]) -> str:
    """Remote script that resolves every path before deleting anything.

    Lexical containment is not enough: ``/tmp/kernel_eval/link/child`` is inside the
    root by name while ``link`` can point anywhere. The worker therefore canonicalizes
    each target and each root with ``readlink -f`` and deletes only what resolves to an
    evaluator-owned directory or to a strict descendant of one of the roots. Lists travel
    newline-separated with globbing off, so a path containing spaces stays one word.
    """
    owned, roots = self._cleanup_roots()

    def block(values):
      return shlex.quote("\n".join(str(v) for v in values))

    # A configured root may itself be a symlink out of scratch space, so the resolved
    # root is checked against the same boundary the configured name had to pass.
    denied = "|".join(f"/{name}|/{name}/*" for name in sorted(_SYSTEM_TOP_LEVEL))

    return (
      f"KE_TARGETS={block(targets)}\n"
      f"KE_OWNED={block(owned)}\n"
      f"KE_ROOTS={block(roots)}\n"
      "set -f\n"
      "IFS='\n'\n"
      "ke_root_ok() {\n"
      '  case "$1" in\n'
      "    /dev/shm|/dev/shm/*) return 0 ;;\n"
      "    /) return 1 ;;\n"
      f"    {denied}) return 1 ;;\n"
      "  esac\n"
      "  return 0\n"
      "}\n"
      # A root that resolves onto shared scratch space, or onto a top-level directory,
      # may not be deleted in full: a configured path symlinked to /tmp would otherwise
      # authorize removing /tmp itself.
      "ke_whole_ok() {\n"
      '  case "$1" in\n'
      "    /tmp|/dev/shm) return 1 ;;\n"
      "  esac\n"
      '  case "$1" in\n'
      "    /*/*) return 0 ;;\n"
      "  esac\n"
      "  return 1\n"
      "}\n"
      "for target in $KE_TARGETS; do\n"
      '  resolved=$(readlink -f -- "$target" 2>/dev/null) || continue\n'
      '  [ -n "$resolved" ] || continue\n'
      "  allowed=0\n"
      "  for root in $KE_OWNED; do\n"
      '    rroot=$(readlink -f -- "$root" 2>/dev/null) || continue\n'
      '    ke_root_ok "$rroot" || continue\n'
      '    ke_whole_ok "$rroot" || continue\n'
      '    [ "$resolved" = "$rroot" ] && allowed=1\n'
      "  done\n"
      "  for root in $KE_ROOTS; do\n"
      '    rroot=$(readlink -f -- "$root" 2>/dev/null) || continue\n'
      '    ke_root_ok "$rroot" || continue\n'
      # ?* requires at least one character after the separator, so a root cannot
      # authorize itself through the descendant branch.
      '    case "$resolved" in\n'
      '      "$rroot"/?*) allowed=1 ;;\n'
      "    esac\n"
      "  done\n"
      '  if [ "$allowed" = 1 ]; then\n'
      '    rm -rf -- "$resolved"\n'
      "  else\n"
      '    echo "Refusing remote cleanup outside an evaluator root:'
      ' $target -> $resolved" >&2\n'
      "  fi\n"
      "done"
    )

  async def _cleanup(self, *remote_paths: str) -> None:
    if not self._config.cleanup or not remote_paths:
      return
    safe: list[str] = []
    for raw in remote_paths:
      candidate = raw.strip()
      if candidate and "\n" not in candidate and self._is_removable(candidate):
        safe.append(str(PurePosixPath(candidate)))
      else:
        print(f"Refusing unsafe remote cleanup target: {raw!r}", file=sys.stderr)
    if not safe:
      return
    await self._ssh_all(self._guarded_rm_command(safe), timeout=60)

  async def _cleanup_dump_dir(self) -> None:
    """Remove the explicit per-project IR dump after artifacts have been pulled.

    A profiled variant can leave roughly 15 GB on every worker. The configured value
    is validated by ``_cleanup`` like every other target, so a typo or an injected
    value cannot turn routine cleanup into ``rm -rf /tmp``, a home directory, or a
    second shell command.
    """
    raw = self._config.env.get("EVAL_DUMP_DIR", "").strip()
    if not raw:
      return
    await self._cleanup(raw)

  # ── public API (parallels KubeEvaluator) ────────────────────────────────

  async def evaluate(self, request: EvalRequest) -> EvalResult:
    job = re.sub(r"[^a-z0-9_.-]", "-", request.variant_id.lower()) or "variant"
    payload_file = f"{self._config.remote_tmp}/{job}.payload"
    artifacts_dir = f"{self._config.remote_artifacts_dir}/{job}"

    if await self._write_payload(request.encode_b64(), payload_file) != 0:
      return EvalResult.compile_error("Failed to stage payload on remote VM")

    # Co-launch on every host (ICI barrier); read the primary's stdout.
    all_out = await self._ssh_all(self._eval_cmd(payload_file, artifacts_dir), timeout=self._config.timeout)
    stdout, stderr, rc = all_out[self._config.primary()]
    results = self._parse_results(stdout)
    result = results.get(request.variant_id)

    if result is None:
      await self._cleanup(payload_file, artifacts_dir)
      tail = (stdout + stderr)[-500:]
      return EvalResult.compile_error(f"No EVAL_RESULT from primary host. Tail: {tail}")

    await self._pull_artifacts(result)
    await self._cleanup(payload_file, artifacts_dir)
    await self._cleanup_dump_dir()
    return EvalResult.from_dict(result)

  async def evaluate_batch(self, batch_request: BatchEvalRequest) -> BatchEvalResult:
    n = len(batch_request.variants)
    job = f"batch-{n}v"
    payload_file = f"{self._config.remote_tmp}/{job}.payload"
    artifacts_dir = self._config.remote_artifacts_dir

    if await self._write_payload(batch_request.encode_b64(), payload_file) != 0:
      err = EvalResult.compile_error("Failed to stage batch payload on remote VM")
      return BatchEvalResult(results={v["variant_id"]: err for v in batch_request.variants})

    timeout = 300 * n + 300
    all_out = await self._ssh_all(
      self._eval_cmd(payload_file, artifacts_dir), timeout=max(timeout, self._config.timeout)
    )
    stdout, stderr, rc = all_out[self._config.primary()]
    parsed = self._parse_results(stdout)

    results: dict[str, EvalResult] = {}
    for v in batch_request.variants:
      vid = v["variant_id"]
      data = parsed.get(vid)
      if data is None:
        results[vid] = EvalResult.compile_error("No EVAL_RESULT found in remote logs")
        continue
      await self._pull_artifacts(data)
      results[vid] = EvalResult.from_dict(data)

    await self._cleanup(payload_file, artifacts_dir)
    await self._cleanup_dump_dir()
    return BatchEvalResult(results=results)


def ssh_config_from_evaluator(evaluator: dict, roofline: dict | None = None) -> SSHConfig:
  """Build an SSHConfig from parsed config dicts (evaluator + optional roofline)."""
  r = roofline or {}
  return SSHConfig(
    host=evaluator.get("ssh_host", ""),
    hosts=[h for h in evaluator.get("ssh_hosts", []) if h],
    user=evaluator.get("ssh_user", "cloud-user"),
    ssh_key=evaluator.get("ssh_key", ""),
    remote_repo_dir=evaluator.get("ssh_remote_repo", "~/dracospark"),
    python_bin=evaluator.get("ssh_python", "~/tpu-venv/bin/python"),
    remote_tmp=evaluator.get("ssh_remote_tmp", "/tmp/kernel_eval"),
    remote_artifacts_dir=evaluator.get("ssh_artifacts_dir", "/tmp/kernel_eval_artifacts"),
    timeout=evaluator.get("timeout", 1800),
    peak_flops=float(r.get("peak_flops", 918e12)),
    peak_hbm_bw=float(r.get("peak_hbm_bw", 1759e9)),
    hbm_capacity_gb=float(r.get("hbm_capacity_gb", 32.0)),
    vmem_capacity_mib=float(r.get("vmem_capacity_mib", 128.0)),
    extra_libtpu_init_args=evaluator.get("ssh_extra_libtpu_init_args", ""),
    env=dict(evaluator.get("ssh_env", {}) or {}),
  )
