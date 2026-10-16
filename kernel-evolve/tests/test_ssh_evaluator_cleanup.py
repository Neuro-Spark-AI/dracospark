"""Scope tests for the remote cleanup paths of SSHEvaluator.

Cleanup runs ``rm -rf`` on every worker. Two layers are tested: the local filter that
refuses a target outright, and the remote script that resolves paths on the worker so an
intermediate symlink cannot carry the deletion outside an evaluator root. That script is
POSIX shell, so it is executed here against a sandbox directory.
"""

from __future__ import annotations

import asyncio
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile

import pytest

from kernel_evolve.ssh_evaluator import SSHConfig, SSHEvaluator

# The guarded cleanup script runs on the Linux TPU VM. macOS resolves /tmp and /var
# through /private, so the escape checks below only hold on Linux.
linux_only = pytest.mark.skipif(sys.platform != "linux", reason="remote cleanup script targets Linux paths")


def _evaluator(dump_dir: str | None, **config) -> tuple[SSHEvaluator, list[str]]:
  env = {} if dump_dir is None else {"EVAL_DUMP_DIR": dump_dir}
  ev = SSHEvaluator(SSHConfig(host="10.0.0.1", env=env, **config))
  sent: list[str] = []

  async def fake_ssh_all(remote_cmd, stdin=None, timeout=None):
    sent.append(remote_cmd)
    return {}

  ev._ssh_all = fake_ssh_all  # type: ignore[method-assign]
  return ev, sent


def _targets_block(command: str) -> str:
  """The KE_TARGETS assignment of the generated script, quoting included."""
  return command.split("KE_TARGETS=", 1)[1].split("\nKE_OWNED=", 1)[0]


# ── local filter ──


@pytest.mark.parametrize(
  "dump_dir",
  [
    "/tmp",
    "/dev/shm",
    "/",
    "/home/user",
    "/home/user/dracospark",
    "relative/path",
    "/tmp/../home/user",
    "",
    "   ",
  ],
)
def test_cleanup_refuses_paths_outside_a_scratch_child(dump_dir):
  ev, sent = _evaluator(dump_dir)
  asyncio.run(ev._cleanup_dump_dir())
  assert sent == []


@pytest.mark.parametrize(
  "dump_dir",
  ["/tmp/dracospark_ir_attn_out", "/tmp/ir_dumps/v1", "/dev/shm/dracospark_ir"],
)
def test_cleanup_accepts_a_named_scratch_child(dump_dir):
  ev, sent = _evaluator(dump_dir)
  asyncio.run(ev._cleanup_dump_dir())
  assert len(sent) == 1
  assert _targets_block(sent[0]) == shlex.quote(dump_dir)


def test_a_second_command_cannot_ride_along_in_the_configured_value():
  ev, sent = _evaluator("/tmp/dracospark_ir; rm -rf /home/user")
  asyncio.run(ev._cleanup_dump_dir())
  assert _targets_block(sent[0]) == "'/tmp/dracospark_ir; rm -rf /home/user'"


def test_generic_cleanup_removes_the_directories_the_evaluator_owns():
  """evaluate_batch passes the configured artifacts root itself; that must still work."""
  ev, sent = _evaluator(None)
  cfg = ev._config
  asyncio.run(ev._cleanup(f"{cfg.remote_tmp}/batch-4v.payload", cfg.remote_artifacts_dir))
  assert _targets_block(sent[0]) == shlex.quote("/tmp/kernel_eval/batch-4v.payload\n/tmp/kernel_eval_artifacts")


def test_generic_cleanup_drops_only_the_unsafe_target():
  ev, sent = _evaluator(None)
  asyncio.run(ev._cleanup("/home/user", "/tmp/kernel_eval/v1.payload"))
  assert _targets_block(sent[0]) == shlex.quote("/tmp/kernel_eval/v1.payload")


@pytest.mark.parametrize("configured", ["/home/user", "/etc/project", "/var/lib/dracospark", "/", "/usr"])
def test_a_configured_root_outside_scratch_space_is_not_removable(configured):
  """Misconfiguring the artifacts directory must not make that directory deletable."""
  ev, sent = _evaluator(None, remote_artifacts_dir=configured)
  asyncio.run(ev._cleanup(configured))
  asyncio.run(ev._cleanup(f"{configured.rstrip('/')}/job-1"))
  assert sent == []


@pytest.mark.parametrize("configured", ["/scratch/dracospark", "/dev/shm/dracospark", "/data/eval"])
def test_a_configured_scratch_root_is_removable_whole(configured):
  ev, sent = _evaluator(None, remote_artifacts_dir=configured)
  asyncio.run(ev._cleanup(configured))
  assert _targets_block(sent[0]) == shlex.quote(configured)


def test_cleanup_is_skipped_when_disabled():
  ev, sent = _evaluator("/tmp/x", cleanup=False)
  asyncio.run(ev._cleanup_dump_dir())
  assert sent == []


# ── the remote script, executed locally against a sandbox ──


@pytest.fixture
def sandbox(tmp_path):
  """Evaluator roots under tmp_path, and a protected directory outside every root.

  /tmp is an admitted scratch root, so a directory under pytest's tmp_path could not
  demonstrate an escape. The protected directory therefore lives under /var/tmp, whose
  top-level name is refused by the root rule.
  """
  scratch = tmp_path / "scratch"
  artifacts = tmp_path / "artifacts"
  (scratch / "job").mkdir(parents=True)
  (scratch / "job" / "file.txt").write_text("job artifact")
  artifacts.mkdir()
  (artifacts / "v1").mkdir()

  outside = pathlib.Path(tempfile.mkdtemp(prefix="ke-protected-", dir="/var/tmp"))
  try:
    (outside / "secret.txt").write_text("must survive")
    (scratch / "escape").symlink_to(outside)
    ev, _ = _evaluator(None, remote_tmp=str(scratch), remote_artifacts_dir=str(artifacts))
    yield ev, scratch, artifacts, outside
  finally:
    shutil.rmtree(outside, ignore_errors=True)


def _run(command: str) -> subprocess.CompletedProcess:
  return subprocess.run(["sh", "-c", command], capture_output=True, text=True, timeout=60)


def _run_dry(command: str) -> list[str]:
  """Run the authorization logic verbatim with the deletion replaced by a report.

  Cases that would delete a shared directory such as /tmp cannot be run for real, so
  the single rm is swapped for an echo and the authorized paths are read back.
  """
  removal = 'rm -rf -- "$resolved"'
  assert command.count(removal) == 1
  dry = command.replace(removal, 'echo "WOULD-REMOVE $resolved"')
  proc = subprocess.run(["sh", "-c", dry], capture_output=True, text=True, timeout=60)
  return [line.split(" ", 1)[1] for line in proc.stdout.splitlines() if line.startswith("WOULD-REMOVE ")]


def test_remote_script_deletes_a_real_descendant(sandbox):
  ev, scratch, _, _ = sandbox

  proc = _run(ev._guarded_rm_command([str(scratch / "job")]))

  assert proc.returncode == 0
  assert not (scratch / "job").exists()


def test_remote_script_refuses_a_target_behind_an_interior_symlink(sandbox):
  """scratch/escape/secret.txt is inside the root by name, outside it once resolved."""
  ev, scratch, _, outside = sandbox

  proc = _run(ev._guarded_rm_command([str(scratch / "escape" / "secret.txt")]))

  assert (outside / "secret.txt").read_text() == "must survive"
  assert "Refusing remote cleanup outside an evaluator root" in proc.stderr


def test_remote_script_refuses_a_symlinked_target(sandbox):
  ev, scratch, _, outside = sandbox

  proc = _run(ev._guarded_rm_command([str(scratch / "escape")]))

  assert outside.exists()
  assert (outside / "secret.txt").exists()
  assert "Refusing remote cleanup outside an evaluator root" in proc.stderr


def test_remote_script_removes_an_owned_root_whole(sandbox):
  ev, _, artifacts, _ = sandbox

  proc = _run(ev._guarded_rm_command([str(artifacts)]))

  assert proc.returncode == 0
  assert not artifacts.exists()


def test_remote_script_handles_a_path_with_spaces(sandbox):
  ev, scratch, _, _ = sandbox
  spaced = scratch / "a b"
  spaced.mkdir()

  proc = _run(ev._guarded_rm_command([str(spaced)]))

  assert proc.returncode == 0
  assert not spaced.exists()


@linux_only
def test_remote_script_refuses_a_configured_root_that_is_itself_a_symlink(tmp_path):
  """A root symlinked out of scratch space must not authorize deleting its target."""
  protected = pathlib.Path(tempfile.mkdtemp(prefix="ke-protected-", dir="/var/tmp"))
  try:
    (protected / "secret.txt").write_text("must survive")
    owned = tmp_path / "owned"
    owned.symlink_to(protected)
    ev, _ = _evaluator(None, remote_tmp=str(owned), remote_artifacts_dir=str(owned))

    proc = _run(ev._guarded_rm_command([str(owned), str(owned / "secret.txt")]))

    assert protected.exists()
    assert (protected / "secret.txt").read_text() == "must survive"
    assert "Refusing remote cleanup outside an evaluator root" in proc.stderr
  finally:
    shutil.rmtree(protected, ignore_errors=True)


def test_remote_script_still_uses_a_root_symlinked_within_scratch_space(tmp_path):
  """Resolving into another scratch directory stays authorized."""
  real = tmp_path / "real"
  (real / "job").mkdir(parents=True)
  link = tmp_path / "link"
  link.symlink_to(real)
  ev, _ = _evaluator(None, remote_tmp=str(link), remote_artifacts_dir=str(link))

  proc = _run(ev._guarded_rm_command([str(link / "job")]))

  assert proc.returncode == 0
  assert not (real / "job").exists()


@linux_only
@pytest.mark.parametrize("shared", ["/tmp", "/dev/shm"])
def test_a_root_resolving_onto_shared_scratch_cannot_delete_it(tmp_path, shared):
  """A configured root symlinked to /tmp must not authorize removing /tmp itself."""
  link = tmp_path / "owned"
  link.symlink_to(shared)
  ev, _ = _evaluator(None, remote_tmp=str(link), remote_artifacts_dir=str(link))

  removed = _run_dry(ev._guarded_rm_command([str(link), shared]))

  assert removed == []


def test_a_top_level_root_cannot_be_deleted_whole(tmp_path):
  link = tmp_path / "owned"
  link.symlink_to("/data")
  ev, _ = _evaluator(None, remote_tmp=str(link), remote_artifacts_dir=str(link))

  removed = _run_dry(ev._guarded_rm_command([str(link), "/data"]))

  assert removed == []


def test_named_scratch_targets_are_still_authorized(sandbox):
  ev, scratch, artifacts, _ = sandbox

  removed = _run_dry(ev._guarded_rm_command([str(scratch / "job"), str(artifacts)]))

  assert removed == [str(scratch / "job"), str(artifacts)]


def test_a_target_whose_parent_is_already_gone_is_skipped(sandbox):
  """readlink -f needs the parent to exist; nothing is left to remove in that case."""
  ev, scratch, _, _ = sandbox

  removed = _run_dry(ev._guarded_rm_command([str(scratch / "missing-dir" / "payload")]))

  assert removed == []


def test_remote_script_leaves_the_root_and_an_outside_target_alone(sandbox):
  ev, scratch, artifacts, outside = sandbox

  _run(ev._guarded_rm_command(["/", str(outside)]))

  assert outside.exists()
  assert (outside / "secret.txt").exists()
  assert artifacts.exists()
  assert scratch.exists()
