"""YAML config parsing and validation with Pydantic."""

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, StrictBool, model_validator


class EvolveMarkers(BaseModel):
  start: str = "# EVOLVE-BLOCK-START"
  end: str = "# EVOLVE-BLOCK-END"


class KernelConfig(BaseModel):
  name: str
  template: str
  reference: str
  evolve_markers: EvolveMarkers = Field(default_factory=EvolveMarkers)


class CorrectnessConfig(BaseModel):
  # "allclose": |candidate - reference| <= atol + rtol * |reference| (np.allclose bar).
  # "abs": |candidate - reference| <= atol, so a large reference magnitude does not
  # buy the candidate extra room. Anything else is rejected at config load instead
  # of silently falling back to allclose.
  method: Literal["allclose", "abs"] = "allclose"
  # A negative or NaN tolerance rejects everything; an infinite one accepts everything.
  rtol: float = Field(default=1e-2, ge=0.0, allow_inf_nan=False)
  atol: float = Field(default=1e-2, ge=0.0, allow_inf_nan=False)


class EvaluatorConfig(BaseModel):
  # "kube" (GKE K8s Job) or "ssh" (run evaluate.py on a TPU-VM over SSH).
  type: str = "kube"

  # --- kube transport ---
  namespace: str = "default"
  job_template: str = ".github/ci/kernel-eval-job.yaml"
  repo: str = ""
  branch: str = "main"
  poll_interval: int = 15
  timeout: int = 600

  # --- ssh transport (type: ssh) ---
  ssh_host: str = ""
  # For a multi-host ICI slice (e.g. v6e-16 = 4 hosts), list EVERY worker's IP.
  # evaluate.py must be co-launched on all of them at once: the hosts share one
  # ICI domain and libtpu blocks on a slice-wide init barrier, so a single-host
  # invocation hangs forever waiting for the peers. Results and artifacts are
  # collected from the first host in the list (the "primary"); the others run
  # the same program only to satisfy the barrier. When empty, falls back to the
  # single-host [ssh_host] (correct for single-host slices: v6e-1/-4/-8).
  ssh_hosts: list[str] = Field(default_factory=list)
  ssh_user: str = "cloud-user"
  ssh_key: str = ""
  # Remote paths are used unquoted in the remote shell command, so a leading "~/"
  # resolves against the SSH user's home directory on each worker.
  ssh_remote_repo: str = "~/dracospark"
  ssh_python: str = "~/tpu-venv/bin/python"
  ssh_remote_tmp: str = "/tmp/kernel_eval"
  ssh_artifacts_dir: str = "/tmp/kernel_eval_artifacts"
  ssh_extra_libtpu_init_args: str = ""
  # Extra environment for the remote evaluate.py, e.g. the libtpu single-host
  # overrides that let ONE host of a multi-host slice initialize its own 2x2
  # independently (required for multi-chip kernels evaluated on one host).
  ssh_env: dict[str, str] = Field(default_factory=dict)

  def resolved_ssh_hosts(self) -> list[str]:
    """All worker hosts to co-launch on; first is the primary. Falls back to [ssh_host]."""
    hosts = [h for h in self.ssh_hosts if h]
    if hosts:
      return hosts
    return [self.ssh_host] if self.ssh_host else []


class RooflineConfig(BaseModel):
  """Per-chip peak numbers for roofline metrics. Defaults are TPU v6e (Trillium)."""

  peak_flops: float = 918e12  # bf16 TFLOP/s
  peak_hbm_bw: float = 1759e9  # HBM GB/s
  hbm_capacity_gb: float = 32.0  # HBM per chip
  vmem_capacity_mib: float = 128.0  # physical VMEM per chip


class TPUConfig(BaseModel):
  cluster: str = ""
  zone: str = ""


class SessionConfig(BaseModel):
  max_iterations: int = 20
  output_dir: str = "runs/default"
  # Opt in to Issue creation, round comments, and Issue closure only.
  # A destination alone never enables GitHub writes or kernel publication.
  github_tracking: StrictBool = False
  tracking_repo: str = ""
  # Empty resolves to the actual timestamped run directory's AGENT.md.
  # An explicit path is local too; writing it never authorizes a commit/push.
  agent_md_path: str = ""

  @model_validator(mode="after")
  def validate_github_tracking(self) -> "SessionConfig":
    if self.github_tracking and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.tracking_repo):
      raise ValueError("github_tracking requires an explicit tracking_repo in owner/name form")
    return self

  def resolved_agent_md_path(self, run_dir: str | Path) -> Path:
    """Resolve new learnings against the actual run, including its timestamp."""
    return Path(self.agent_md_path) if self.agent_md_path else Path(run_dir) / "AGENT.md"


class BatchConfig(BaseModel):
  variants_per_round: int = Field(default=1, ge=1)
  top_k: int = Field(default=1, ge=1)
  max_active_lineages: int = Field(default=4, ge=1)


class EvolveConfig(BaseModel):
  model_config = {"extra": "forbid"}

  kernel: KernelConfig
  shapes: list[dict[str, Any]]
  correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
  evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
  roofline: RooflineConfig = Field(default_factory=RooflineConfig)
  tpu: TPUConfig = Field(default_factory=TPUConfig)
  session: SessionConfig = Field(default_factory=SessionConfig)
  batch: BatchConfig = Field(default_factory=BatchConfig)


def load_config(path: str | Path) -> EvolveConfig:
  """Load and validate an EvolveConfig from a YAML file."""
  with open(path) as f:
    data = yaml.safe_load(f)
  return EvolveConfig(**data)


def snapshot_run_inputs(config_path: str | Path, run_dir: str | Path) -> EvolveConfig:
  """Snapshot kernel inputs and a reloadable config without overwriting a run."""
  source = Path(config_path).resolve()
  cfg = load_config(source)
  # Read all inputs before creating outputs so missing sources leave no partial run.
  template = (source.parent / cfg.kernel.template).read_text()
  reference = (source.parent / cfg.kernel.reference).read_text()
  source_config = source.read_text()
  snapshot = cfg.model_copy(
    update={"kernel": cfg.kernel.model_copy(update={"template": "baseline_kernel.py", "reference": "reference.py"})}
  )
  outputs = {
    "source_config.yaml": source_config,
    "baseline_kernel.py": template,
    "reference.py": reference,
    "config.yaml": yaml.safe_dump(snapshot.model_dump(), sort_keys=False),
  }
  destination = Path(run_dir)
  if any((destination / name).exists() for name in outputs):
    raise FileExistsError(f"Run input snapshot already exists in {destination}")
  destination.mkdir(parents=True, exist_ok=True)
  for name, content in outputs.items():
    with (destination / name).open("x") as f:
      f.write(content)
  return snapshot
