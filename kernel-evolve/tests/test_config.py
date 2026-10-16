"""Tests for simplified YAML config parsing and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kernel_evolve.config import EvolveConfig, SessionConfig, load_config, snapshot_run_inputs


@pytest.fixture
def example_config_path():
  return Path(__file__).parent.parent / "examples" / "matmul.yaml"


def test_load_config_from_yaml(example_config_path):
  cfg = load_config(example_config_path)
  assert cfg.kernel.name == "tiled_matmul"
  assert cfg.kernel.evolve_markers.start == "# EVOLVE-BLOCK-START"
  assert len(cfg.shapes) == 2
  assert cfg.shapes[0]["M"] == 1024
  assert cfg.correctness.method == "allclose"
  assert cfg.correctness.rtol == pytest.approx(1e-2)
  assert cfg.tpu.cluster == "REPLACE_WITH_GKE_CLUSTER"
  assert cfg.session.max_iterations == 20
  assert cfg.session.output_dir == "runs/matmul"
  assert cfg.session.github_tracking is False
  assert cfg.session.tracking_repo == ""


def test_ssh_remote_defaults_are_home_relative():
  """Remote paths default under the SSH user's home, not a fixed account's home."""
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    evaluator={"type": "ssh"},
  )
  assert cfg.evaluator.ssh_remote_repo == "~/dracospark"
  assert cfg.evaluator.ssh_python == "~/tpu-venv/bin/python"


def test_config_defaults():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64, "N": 64, "K": 64}],
    tpu={"cluster": "c", "zone": "z"},
  )
  assert cfg.correctness.method == "allclose"
  assert cfg.evaluator.namespace == "default"
  assert cfg.evaluator.poll_interval == 15
  assert cfg.evaluator.timeout == 600
  assert cfg.session.max_iterations == 20
  assert cfg.session.output_dir == "runs/default"


def test_correctness_method_is_validated():
  base = {
    "kernel": {"name": "test", "template": "k.py", "reference": "r.py"},
    "shapes": [{"M": 64}],
  }
  assert EvolveConfig(**base, correctness={"method": "abs"}).correctness.method == "abs"
  with pytest.raises(ValidationError):
    EvolveConfig(**base, correctness={"method": "l2"})


def test_tolerances_must_be_finite_and_nonnegative():
  base = {
    "kernel": {"name": "test", "template": "k.py", "reference": "r.py"},
    "shapes": [{"M": 64}],
  }
  for bad in ({"atol": -1.0}, {"rtol": -1e-9}, {"atol": float("inf")}, {"rtol": float("nan")}):
    with pytest.raises(ValidationError):
      EvolveConfig(**base, correctness=bad)
  assert EvolveConfig(**base, correctness={"atol": 0.0}).correctness.atol == 0.0


def test_config_with_evaluator():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
    evaluator={
      "namespace": "custom-ns",
      "job_template": "custom-template.yaml",
      "repo": "user/repo",
      "branch": "dev",
      "poll_interval": 30,
      "timeout": 1200,
    },
  )
  assert cfg.evaluator.namespace == "custom-ns"
  assert cfg.evaluator.repo == "user/repo"
  assert cfg.evaluator.poll_interval == 30


def test_config_with_session():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
    session={"max_iterations": 50, "output_dir": "runs/custom"},
  )
  assert cfg.session.max_iterations == 50
  assert cfg.session.output_dir == "runs/custom"


def test_session_config_defaults():
  s = SessionConfig()
  assert s.max_iterations == 20
  assert s.output_dir == "runs/default"


def test_config_no_evolution_or_llm_fields():
  """Verify old fields (evolution, llm, logging) are not accepted."""
  with pytest.raises(ValueError):
    EvolveConfig(
      kernel={"name": "t", "template": "k.py", "reference": "r.py"},
      shapes=[{"M": 64}],
      tpu={"cluster": "c", "zone": "z"},
      evolution={"population_size": 25},
    )


def test_config_with_batch():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
    batch={
      "variants_per_round": 5,
      "top_k": 2,
      "max_active_lineages": 4,
    },
  )
  assert cfg.batch.variants_per_round == 5
  assert cfg.batch.top_k == 2
  assert cfg.batch.max_active_lineages == 4


def test_batch_config_defaults():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
  )
  assert cfg.batch.variants_per_round == 1
  assert cfg.batch.top_k == 1
  assert cfg.batch.max_active_lineages == 4


def test_load_matmul_config_has_batch(example_config_path):
  cfg = load_config(example_config_path)
  assert cfg.batch.variants_per_round == 5
  assert cfg.batch.top_k == 2


def test_batch_config_loaded_from_yaml(tmp_path):
  yaml_content = """
kernel:
  name: "test"
  template: "k.py"
  reference: "r.py"
shapes:
  - { M: 64 }
tpu:
  cluster: "c"
  zone: "z"
batch:
  variants_per_round: 3
  top_k: 2
"""
  f = tmp_path / "config.yaml"
  f.write_text(yaml_content)
  cfg = load_config(f)
  assert cfg.batch.variants_per_round == 3
  assert cfg.batch.top_k == 2


def test_tracking_destination_alone_does_not_enable_github():
  session = SessionConfig(tracking_repo="owner/private-repo")
  assert session.github_tracking is False


@pytest.mark.parametrize("repo", ["", " ", "owner", "https://github.com/owner/repo", "owner/repo/extra"])
def test_github_tracking_requires_explicit_repository(repo):
  with pytest.raises(ValidationError, match="explicit tracking_repo"):
    SessionConfig(github_tracking=True, tracking_repo=repo)


@pytest.mark.parametrize("enabled", ["true", "false", 1, 0, None])
def test_github_opt_in_requires_boolean(enabled):
  with pytest.raises(ValidationError):
    SessionConfig(github_tracking=enabled, tracking_repo="owner/private-repo")


def test_tracking_opt_in_survives_config_round_trip(tmp_path):
  config = {
    "kernel": {"name": "test", "template": "k.py", "reference": "r.py"},
    "shapes": [{"M": 64}],
    "session": {"github_tracking": True, "tracking_repo": "owner/private-repo"},
  }
  import yaml

  path = tmp_path / "config.yaml"
  path.write_text(yaml.safe_dump(config))
  session = load_config(path).session
  assert session.github_tracking is True
  assert SessionConfig(**session.model_dump()).tracking_repo == "owner/private-repo"


def test_default_learnings_follow_actual_run_directory(tmp_path):
  session = SessionConfig(output_dir="runs/test")
  first = tmp_path / "test_20261015_100000"
  second = tmp_path / "test_20261015_110000"
  assert session.resolved_agent_md_path(first) == first / "AGENT.md"
  assert session.resolved_agent_md_path(second) == second / "AGENT.md"
  explicit = SessionConfig(agent_md_path=str(tmp_path / "private" / "AGENT.md"))
  assert explicit.resolved_agent_md_path(first) == tmp_path / "private" / "AGENT.md"


@pytest.fixture
def snapshot_source(tmp_path):
  source = tmp_path / "source"
  (source / "kernels").mkdir(parents=True)
  (source / "kernels" / "kernel.py").write_text("# candidate input\n")
  (source / "kernels" / "reference.py").write_text("# frozen reference\n")
  config = source / "input.yaml"
  config.write_text("""# original config comments are retained
kernel:
  name: fixture
  template: kernels/kernel.py
  reference: kernels/reference.py
shapes:
  - {M: 512, N: 512, K: 512}
""")
  return config


def test_run_snapshot_reloads_without_original_sources(snapshot_source, tmp_path):
  import shutil

  original = snapshot_source.read_text()
  run = tmp_path / "actual_run_20261015"
  snapshot_run_inputs(snapshot_source, run)
  shutil.rmtree(snapshot_source.parent)
  reloaded = load_config(run / "config.yaml")
  assert (run / reloaded.kernel.template).read_text() == "# candidate input\n"
  assert (run / reloaded.kernel.reference).read_text() == "# frozen reference\n"
  assert (run / "source_config.yaml").read_text() == original
  assert reloaded.shapes == [{"M": 512, "N": 512, "K": 512}]
  assert reloaded.session.github_tracking is False
  assert reloaded.session.resolved_agent_md_path(run) == run / "AGENT.md"


def test_run_snapshot_refuses_existing_inputs(snapshot_source, tmp_path):
  run = tmp_path / "existing"
  run.mkdir()
  (run / "reference.py").write_text("existing frozen reference")
  with pytest.raises(FileExistsError):
    snapshot_run_inputs(snapshot_source, run)
  assert (run / "reference.py").read_text() == "existing frozen reference"
  assert not (run / "config.yaml").exists()


def test_missing_reference_does_not_create_snapshot(snapshot_source, tmp_path):
  (snapshot_source.parent / "kernels" / "reference.py").unlink()
  run = tmp_path / "missing_source"
  with pytest.raises(FileNotFoundError):
    snapshot_run_inputs(snapshot_source, run)
  assert not run.exists()
