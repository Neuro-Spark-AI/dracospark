"""IR dump lifecycle and candidate binding of profile artifacts in docker/evaluate.py.

The reference runs in the evaluator's process, so its compilations dump into the
same directories as the candidate's. These tests pin that artifacts are taken from
the candidate's own kernels and modules, and that a variant's dumps (~10 GB on
hardware) are removed once its artifacts have been copied.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_evaluate_module():
  module_path = Path(__file__).resolve().parents[1] / "docker" / "evaluate.py"
  spec = importlib.util.spec_from_file_location("kernel_evolve_docker_evaluate_dumps", module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


@pytest.fixture
def evaluate():
  return _load_evaluate_module()


@pytest.fixture
def dump_env(tmp_path, monkeypatch):
  """Point dumps at tmp_path and restore the flag variables _setup_dump_env() edits."""
  monkeypatch.setenv("EVAL_DUMP_DIR", str(tmp_path))
  monkeypatch.setenv("XLA_FLAGS", "")
  monkeypatch.setenv("LIBTPU_INIT_ARGS", "")
  for name in ("VARIANT_ID", "EVAL_KEEP_DUMPS", "EVAL_DISABLE_DUMPS"):
    monkeypatch.delenv(name, raising=False)
  return tmp_path


CANDIDATE = """
def _attn_kernel(q_ref, o_ref):
  pass

def optimized_compute(B, S):
  pass

def _make_test_data(B, S):
  pass
"""

REFERENCE = """
def _make_test_data(B, S):
  pass

def simple_compute(B, S):
  pass
"""


def _mosaic(path: Path, func: str, size: int) -> Path:
  path.write_text(f"module {{\n  func.func @{func}(%arg0: i32) {{\n" + "x" * size + "\n}}\n;; bundle\n")
  return path


def _hlo(path: Path, module: str, size: int) -> Path:
  path.write_text(
    f"HloModule {module}, entry_computation_layout={{()->f32[8]}}\n\nENTRY %main {{\n"
    '  %r = f32[8] custom-call(), custom_call_target="tpu_custom_call"\n' + "#" * size + "\n}\n"
  )
  return path


def _dump_dirs(root: Path) -> tuple[Path, Path, Path]:
  dirs = tuple(root / name for name in ("hlo", "llo", "mosaic"))
  for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
  return dirs


# ── candidate binding ────────────────────────────────────────────────────────


def test_candidate_function_names_exclude_the_reference_definitions(evaluate):
  assert evaluate._candidate_function_names(CANDIDATE, REFERENCE) == {"_attn_kernel", "optimized_compute"}


def test_candidate_function_names_are_empty_for_unparseable_code(evaluate):
  assert evaluate._candidate_function_names("def broken(:", REFERENCE) == set()


def test_llo_pick_prefers_the_candidate_kernel_over_a_larger_reference_dump(evaluate, tmp_path):
  _, _, mosaic = _dump_dirs(tmp_path)
  ref = _mosaic(mosaic / "111-5-mosaic-dump-main-post-lower-to-llo.txt", "main", 20000)
  own = _mosaic(mosaic / "222-5-mosaic-dump-_attn_kernel-post-lower-to-llo.txt", "_attn_kernel", 500)

  result = evaluate.stage_profile_deep(
    {}, [{"B": 1}], dump_dir=str(tmp_path), kernel_names=evaluate._candidate_function_names(CANDIDATE, REFERENCE)
  )

  assert result["_llo_file"] == str(own)
  assert result["llo_kernel"] == "_attn_kernel"
  assert result["artifact_binding"]["llo"] == "candidate"
  # Every kernel's dump still ships, so the reference's IR stays inspectable.
  assert result["_llo_all"]["llo_main.txt"] == str(ref)


def test_llo_pick_without_candidate_names_is_marked_unverified(evaluate, tmp_path):
  _, _, mosaic = _dump_dirs(tmp_path)
  ref = _mosaic(mosaic / "111-5-mosaic-dump-main-post-lower-to-llo.txt", "main", 20000)
  _mosaic(mosaic / "222-5-mosaic-dump-_attn_kernel-post-lower-to-llo.txt", "_attn_kernel", 500)

  result = evaluate.stage_profile_deep({}, [{"B": 1}], dump_dir=str(tmp_path))

  assert result["_llo_file"] == str(ref)
  assert result["artifact_binding"]["llo"] == "unverified"


def test_llo_pick_warns_when_no_dump_belongs_to_the_candidate(evaluate, tmp_path, capsys):
  _, _, mosaic = _dump_dirs(tmp_path)
  ref = _mosaic(mosaic / "111-5-mosaic-dump-main-post-lower-to-llo.txt", "main", 2000)

  result = evaluate.stage_profile_deep({}, [{"B": 1}], dump_dir=str(tmp_path), kernel_names={"_attn_kernel"})

  assert result["_llo_file"] == str(ref)
  assert result["artifact_binding"]["llo"] == "unverified"
  assert "no Mosaic dump matches a candidate kernel" in capsys.readouterr().err


def test_hlo_pick_prefers_the_candidate_module_when_the_reference_also_calls_mosaic(evaluate, tmp_path):
  hlo, _, _ = _dump_dirs(tmp_path)
  _hlo(hlo / "module_0001.jit_simple_compute.after_optimizations.txt", "jit_simple_compute", 20000)
  own = _hlo(hlo / "module_0002.jit_optimized_compute.after_optimizations.txt", "jit_optimized_compute", 200)

  result = evaluate.stage_profile_deep(
    {}, [{"B": 1}], dump_dir=str(tmp_path), kernel_names=evaluate._candidate_function_names(CANDIDATE, REFERENCE)
  )

  assert result["_hlo_file"] == str(own)
  assert result["hlo_module"] == "jit_optimized_compute"
  assert result["artifact_binding"]["hlo"] == "candidate"


CANDIDATE_WITH_HELPER = (
  CANDIDATE
  + """
def compute(x):
  return x
"""
)


def test_a_candidate_helper_name_cannot_claim_the_reference_module(evaluate, tmp_path):
  hlo, _, _ = _dump_dirs(tmp_path)
  ref = _hlo(hlo / "module_0001.jit_simple_compute.after_optimizations.txt", "jit_simple_compute", 20000)
  names = evaluate._candidate_function_names(CANDIDATE_WITH_HELPER, REFERENCE)
  assert "compute" in names

  result = evaluate.stage_profile_deep({}, [{"B": 1}], dump_dir=str(tmp_path), kernel_names=names)

  # Only the reference's module was dumped: it may be the fallback pick, but never "candidate".
  assert result["_hlo_file"] == str(ref)
  assert result["artifact_binding"]["hlo"] == "unverified"


def test_hlo_binding_matches_the_full_jitted_name_with_a_helper_present(evaluate, tmp_path):
  hlo, _, _ = _dump_dirs(tmp_path)
  _hlo(hlo / "module_0001.jit_simple_compute.after_optimizations.txt", "jit_simple_compute", 20000)
  own = _hlo(hlo / "module_0002.jit_optimized_compute.after_optimizations.txt", "jit_optimized_compute", 200)

  result = evaluate.stage_profile_deep(
    {},
    [{"B": 1}],
    dump_dir=str(tmp_path),
    kernel_names=evaluate._candidate_function_names(CANDIDATE_WITH_HELPER, REFERENCE),
  )

  assert result["_hlo_file"] == str(own)
  assert result["artifact_binding"]["hlo"] == "candidate"


def test_schedule_pass_prefers_one_naming_a_candidate_kernel(evaluate, tmp_path, capsys):
  _, llo, mosaic = _dump_dirs(tmp_path)
  _mosaic(mosaic / "222-5-mosaic-dump-_attn_kernel-post-lower-to-llo.txt", "_attn_kernel", 100)
  (llo / "1-main-90-late-finalization.txt").write_text("main\n" + ".mxu0 x\n" * 50 + ".mxu1 y\n" * 50)
  (llo / "2-_attn_kernel-90-late-finalization.txt").write_text("_attn_kernel\n.mxu0 x\n.mxu1 y\n.mxu1 z\n")

  result = evaluate.stage_profile_deep({}, [{"B": 1}], dump_dir=str(tmp_path), kernel_names={"_attn_kernel"})

  assert result["ok"] is True
  assert "schedule pass has 3 .mxuN tags" in capsys.readouterr().err


# ── dump lifecycle ───────────────────────────────────────────────────────────


def test_setup_dump_env_clears_stale_dumps_of_the_same_variant(evaluate, dump_env, monkeypatch):
  monkeypatch.setenv("VARIANT_ID", "v1")
  stale = dump_env / "v1" / "mosaic" / "old-mosaic-dump-main-post-lower-to-llo.txt"
  stale.parent.mkdir(parents=True)
  stale.write_text("func.func @main")

  evaluate._setup_dump_env()

  assert not stale.exists()
  assert all((dump_env / "v1" / name).is_dir() for name in ("hlo", "llo", "mosaic"))


def test_cleanup_removes_the_variant_dump_directory(evaluate, dump_env, monkeypatch):
  monkeypatch.setenv("VARIANT_ID", "v1")
  evaluate._setup_dump_env()
  (dump_env / "v1" / "hlo" / "module.txt").write_text("HloModule x")

  evaluate._cleanup_dumps()

  assert not (dump_env / "v1").exists()


def test_cleanup_without_variant_id_leaves_other_files_in_the_base(evaluate, dump_env):
  (dump_env / "unrelated.txt").write_text("keep me")
  evaluate._setup_dump_env()

  evaluate._cleanup_dumps()

  assert (dump_env / "unrelated.txt").exists()
  assert not any((dump_env / name).exists() for name in ("hlo", "llo", "mosaic"))


def test_keep_dumps_flag_preserves_them(evaluate, dump_env, monkeypatch):
  monkeypatch.setenv("VARIANT_ID", "v1")
  monkeypatch.setenv("EVAL_KEEP_DUMPS", "1")
  evaluate._setup_dump_env()

  evaluate._cleanup_dumps()

  assert (dump_env / "v1" / "mosaic").is_dir()


def test_main_cleans_up_after_an_early_exit(evaluate, dump_env, monkeypatch):
  monkeypatch.setenv("VARIANT_ID", "v1")

  def compile_error_exit():
    (dump_env / "v1" / "mosaic" / "dump.txt").write_text("func.func @k")
    raise SystemExit(0)

  monkeypatch.setattr(evaluate, "_evaluate", compile_error_exit)
  with pytest.raises(SystemExit):
    evaluate.main()

  assert not (dump_env / "v1").exists()


# ── batch dispatch ───────────────────────────────────────────────────────────


def _payload(variant_id: str) -> dict:
  return {
    "batch": True,
    "reference_code": REFERENCE,
    "shapes": [{"B": 1}],
    "variants": [{"variant_id": variant_id, "kernel_code": CANDIDATE}],
  }


def test_batch_removes_a_timed_out_variants_dumps(dump_env):
  from kernel_evolve.docker_evaluate_helpers import batch_dispatch

  (dump_env / "v1-slow" / "llo").mkdir(parents=True)
  (dump_env / "v1-slow" / "llo" / "big.txt").write_text("x")
  (dump_env / "other-variant").mkdir()

  with patch(
    "kernel_evolve.docker_evaluate_helpers.subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd="python", timeout=60),
  ):
    batch_dispatch(_payload("v1-slow"), evaluator_script="/fake/evaluate.py", per_variant_timeout=60)

  assert not (dump_env / "v1-slow").exists()
  assert (dump_env / "other-variant").exists()


def test_unsafe_variant_ids_are_never_used_as_cleanup_paths(dump_env):
  from kernel_evolve.docker_evaluate_helpers import remove_variant_dumps

  outside = dump_env / "hlo"
  outside.mkdir()

  remove_variant_dumps("../" + dump_env.name)
  remove_variant_dumps("")

  assert outside.exists()


def test_a_full_disk_is_reported_as_infrastructure(dump_env):
  from kernel_evolve.docker_evaluate_helpers import batch_dispatch

  crashed = subprocess.CompletedProcess(
    args=[], returncode=1, stdout="", stderr="OSError: [Errno 28] No space left on device: '/tmp/ir_dumps/v1'"
  )
  with patch("kernel_evolve.docker_evaluate_helpers.subprocess.run", return_value=crashed):
    results = batch_dispatch(_payload("v1"), evaluator_script="/fake/evaluate.py", per_variant_timeout=60)

  parsed = json.loads(results[0].split("EVAL_RESULT:", 1)[1])
  assert parsed["error"].startswith("Infrastructure: disk full on the TPU host")
