"""Tests for the TPU pod evaluator helpers."""

from __future__ import annotations

import base64
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _load_evaluate_module():
  module_path = Path(__file__).resolve().parents[1] / "docker" / "evaluate.py"
  spec = importlib.util.spec_from_file_location("kernel_evolve_docker_evaluate", module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def test_stage_benchmark_accepts_reference_entrypoints(tmp_path):
  evaluate = _load_evaluate_module()

  globals_dict = {
    "simple_compute": lambda M=1: M,
    "reference_fn": lambda **kwargs: kwargs["M"],
  }

  mock_lowered = MagicMock()
  mock_compiled = MagicMock()
  mock_compiled.return_value = 42
  mock_compiled.memory_analysis.side_effect = Exception("no TPU")
  mock_lowered.compile.return_value = mock_compiled
  mock_jitted = MagicMock()
  mock_jitted.lower.return_value = mock_lowered

  with patch.object(evaluate, "jax") as mock_jax, patch.object(evaluate, "raw_to_tool_data", None):
    mock_jax.jit.return_value = mock_jitted
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()

    result = evaluate.stage_benchmark(
      globals_dict,
      shapes=[{"M": 4}],
      trace_dir=str(tmp_path),
      warmup=1,
      n_iters=2,
    )

  assert result["ok"] is True
  assert result["latency_ms"] > 0


def test_stage_correctness_uses_reference_entrypoint_names():
  evaluate = _load_evaluate_module()

  kernel_code = """
def optimized_compute(M=1):
  return M + 1
"""
  reference_code = """
def simple_compute(M=1):
  return M + 1
"""
  compiled = evaluate.stage_compile(kernel_code)

  result = evaluate.stage_correctness(
    kernel_code,
    reference_code,
    shapes=[{"M": 3}],
    rtol=1e-2,
    atol=0.0,
    exec_globals=compiled["globals"],
  )

  assert result["ok"] is True
  assert result["max_diff"] == 0.0


def test_stage_correctness_reports_maximum_across_shapes():
  evaluate = _load_evaluate_module()

  kernel_code = """
import numpy as np
def optimized_compute(M=1):
  return np.asarray([M + M * 0.001], dtype=np.float32)
"""
  reference_code = """
import numpy as np
def simple_compute(M=1):
  return np.asarray([M], dtype=np.float32)
"""
  compiled = evaluate.stage_compile(kernel_code)

  result = evaluate.stage_correctness(
    kernel_code,
    reference_code,
    shapes=[{"M": 1}, {"M": 3}, {"M": 2}],
    rtol=0.0,
    atol=0.01,
    exec_globals=compiled["globals"],
  )

  assert result["ok"] is True
  assert np.isclose(result["max_diff"], 0.003)


_METHOD_KERNEL = """
import numpy as np
def optimized_compute(M=1):
  return np.asarray([100.5], dtype=np.float32)
"""

_METHOD_REFERENCE = """
import numpy as np
def simple_compute(M=1):
  return np.asarray([100.0], dtype=np.float32)
"""


def _run_method(evaluate, method):
  compiled = evaluate.stage_compile(_METHOD_KERNEL)
  return evaluate.stage_correctness(
    _METHOD_KERNEL,
    _METHOD_REFERENCE,
    shapes=[{"M": 1}],
    rtol=0.01,
    atol=0.1,
    exec_globals=compiled["globals"],
    method=method,
  )


def test_stage_correctness_allclose_scales_tolerance_by_reference():
  evaluate = _load_evaluate_module()

  result = _run_method(evaluate, "allclose")

  # bound = atol + rtol * |ref| = 0.1 + 1.0 = 1.1, diff = 0.5
  assert result["ok"] is True
  assert np.isclose(result["max_diff"], 0.5)
  assert np.isclose(result["max_scaled_diff"], 0.5 / 1.1)


def test_stage_correctness_abs_method_ignores_reference_magnitude():
  evaluate = _load_evaluate_module()

  result = _run_method(evaluate, "abs")

  # bound = atol = 0.1, diff = 0.5: the same candidate allclose accepts
  assert result["ok"] is False
  assert np.isclose(result["max_diff"], 0.5)
  assert np.isclose(result["max_scaled_diff"], 5.0)
  assert "method=abs" in result["error"]


def test_stage_correctness_rejects_unknown_method():
  evaluate = _load_evaluate_module()

  result = _run_method(evaluate, "l2")

  assert result["ok"] is False
  assert "unknown correctness method" in result["error"]


def test_non_representable_numbers_never_reach_the_result_json():
  evaluate = _load_evaluate_module()

  assert evaluate._json_safe_number(2.5) == 2.5
  assert evaluate._json_safe_number(float("inf")) is None
  assert evaluate._json_safe_number(float("nan")) is None

  # json.dumps would otherwise emit the bare tokens Infinity and NaN, which strict readers reject.
  emitted = json.dumps(
    {
      "max_diff": evaluate._json_safe_number(float("inf")),
      "max_scaled_diff": evaluate._json_safe_number(float("nan")),
    },
    allow_nan=False,
  )
  assert emitted == '{"max_diff": null, "max_scaled_diff": null}'


def test_scaled_diff_requires_exact_match_at_zero_tolerance():
  evaluate = _load_evaluate_module()

  bound = np.zeros(2, dtype=np.float32)
  scaled = evaluate._scaled_diff(np.asarray([0.0, 1e-9], dtype=np.float32), bound)

  assert scaled[0] == 0.0
  assert np.isinf(scaled[1])


def test_integer_difference_survives_the_conversion():
  """float32 collapses 16_777_217 onto 16_777_216 and would call them identical."""
  evaluate = _load_evaluate_module()

  candidate = np.asarray([16_777_217], dtype=np.int32)
  reference = np.asarray([16_777_216], dtype=np.int32)

  max_diff, max_scaled, nonfinite, failed = evaluate._local_correctness_metrics(
    candidate, reference, rtol=0.0, atol=0.0, method="abs"
  )

  assert max_diff == 1.0
  assert failed is True
  assert nonfinite is False
  assert np.isinf(max_scaled)


def test_int64_differences_are_measured_exactly_on_both_sides_of_the_bound():
  """A float64 conversion of the operands would round 2**53 + 1 onto 2**53."""
  evaluate = _load_evaluate_module()

  reference = np.asarray([2**53], dtype=np.int64)

  # difference of 1: refused at zero tolerance, accepted at atol=1
  candidate = np.asarray([2**53 + 1], dtype=np.int64)
  max_diff, _, _, failed = evaluate._local_correctness_metrics(candidate, reference, rtol=0.0, atol=0.0, method="abs")
  assert max_diff == 1.0
  assert failed is True

  max_diff, max_scaled, _, failed = evaluate._local_correctness_metrics(
    candidate, reference, rtol=0.0, atol=1.0, method="abs"
  )
  assert max_diff == 1.0
  assert max_scaled == 1.0
  assert failed is False

  # difference of 5 must not round down to 4 and slip under atol=4
  candidate = np.asarray([2**53 + 5], dtype=np.int64)
  max_diff, _, _, failed = evaluate._local_correctness_metrics(candidate, reference, rtol=0.0, atol=4.0, method="abs")
  assert max_diff == 5.0
  assert failed is True


def test_large_integer_difference_is_compared_without_conversion():
  """Converting the difference would round 2**53 + 1 onto 2**53 and meet the bound."""
  evaluate = _load_evaluate_module()

  candidate = np.asarray([2**53 + 1], dtype=np.int64)
  reference = np.asarray([0], dtype=np.int64)

  _, _, _, failed = evaluate._local_correctness_metrics(candidate, reference, rtol=0.0, atol=float(2**53), method="abs")

  assert failed is True

  # float64 spacing at this magnitude is 2, so 2**53 + 2 is the next representable bound.
  _, _, _, failed = evaluate._local_correctness_metrics(
    candidate, reference, rtol=0.0, atol=float(2**53 + 2), method="abs"
  )

  assert failed is False


def test_integral_candidate_against_a_floating_reference_is_refused():
  """The difference cannot be taken exactly, so 2**53 + 1 would meet a 2**53 bound."""
  evaluate = _load_evaluate_module()

  candidate = np.asarray([2**53 + 1], dtype=np.int64)
  reference = np.asarray([0.0], dtype=np.float64)

  with pytest.raises(ValueError, match="not comparable"):
    evaluate._local_correctness_metrics(candidate, reference, rtol=0.0, atol=float(2**53), method="abs")


def test_stage_correctness_reports_an_incomparable_dtype_pair():
  evaluate = _load_evaluate_module()

  kernel_code = """
import numpy as np
def optimized_compute(M=1):
  return np.asarray([2**53 + 1], dtype=np.int64)
"""
  reference_code = """
import numpy as np
def simple_compute(M=1):
  return np.asarray([0.0], dtype=np.float64)
"""
  compiled = evaluate.stage_compile(kernel_code)
  result = evaluate.stage_correctness(
    kernel_code,
    reference_code,
    shapes=[{"M": 1}],
    rtol=0.0,
    atol=float(2**53),
    exec_globals=compiled["globals"],
    method="abs",
  )

  assert result["ok"] is False
  assert "not comparable" in result["error"]


def test_nan_candidate_fails_and_keeps_the_maxima_finite():
  evaluate = _load_evaluate_module()

  candidate = np.asarray([np.nan], dtype=np.float32)
  reference = np.asarray([1.0], dtype=np.float32)

  max_diff, max_scaled, nonfinite, failed = evaluate._local_correctness_metrics(
    candidate, reference, rtol=0.01, atol=0.1
  )

  assert failed is True
  assert nonfinite is True
  assert math.isfinite(max_diff)
  assert math.isfinite(max_scaled)


def test_infinite_candidate_fails_and_is_reported_in_the_error():
  evaluate = _load_evaluate_module()

  kernel_code = """
import numpy as np
def optimized_compute(M=1):
  return np.asarray([np.inf], dtype=np.float32)
"""
  reference_code = """
import numpy as np
def simple_compute(M=1):
  return np.asarray([1.0], dtype=np.float32)
"""
  compiled = evaluate.stage_compile(kernel_code)
  result = evaluate.stage_correctness(
    kernel_code,
    reference_code,
    shapes=[{"M": 1}],
    rtol=0.01,
    atol=0.1,
    exec_globals=compiled["globals"],
  )

  assert result["ok"] is False
  assert "non-finite" in result["error"]
  assert math.isfinite(result["max_diff"])
  # allow_nan=False makes json.dumps raise on anything it cannot represent.
  emitted = json.dumps(
    {
      "max_diff": evaluate._json_safe_number(result["max_diff"]),
      "max_scaled_diff": evaluate._json_safe_number(result["max_scaled_diff"]),
    },
    allow_nan=False,
  )
  assert json.loads(emitted)["max_diff"] == result["max_diff"]


def test_matching_infinities_are_not_a_mismatch():
  evaluate = _load_evaluate_module()

  values = np.asarray([np.inf, 1.0], dtype=np.float32)

  _, _, nonfinite, failed = evaluate._local_correctness_metrics(values, values.copy(), rtol=0.01, atol=0.1)

  assert nonfinite is False
  assert failed is False


def test_multi_host_aggregation_keeps_the_worst_of_every_field():
  """A mismatch on a non-primary host must not be lost or rounded in the all-gather."""
  evaluate = _load_evaluate_module()

  # 1 + 2**-52 survives float64 and collapses to 1.0 in float32, which is what a
  # plain float64 array would become on a device with x64 disabled.
  remote = np.asarray([1.0 + 2.0**-52, 4.0, 1.0, 1.0], dtype=np.float64)
  captured = {}

  def fake_allgather(words, tiled=False):
    captured["words"] = np.asarray(words)
    captured["tiled"] = tiled
    return np.concatenate([np.asarray(words), np.frombuffer(remote.tobytes(), dtype=np.int32)])

  fake_experimental = SimpleNamespace(multihost_utils=SimpleNamespace(process_allgather=fake_allgather))

  with (
    patch.object(evaluate, "jax", SimpleNamespace(process_count=lambda: 2)),
    patch.dict(sys.modules, {"jax.experimental": fake_experimental}),
  ):
    max_diff, max_scaled, nonfinite, failed = evaluate._global_correctness_metrics(0.5, 0.25, False, False)

  assert captured["words"].dtype == np.int32
  assert captured["words"].shape == (8,)
  assert captured["tiled"] is False
  assert max_diff == 1.0 + 2.0**-52
  assert max_diff != 1.0
  assert max_scaled == 4.0
  assert nonfinite is True
  assert failed is True


def test_single_host_aggregation_passes_the_metrics_through():
  evaluate = _load_evaluate_module()

  with patch.object(evaluate, "jax", SimpleNamespace(process_count=lambda: 1)):
    assert evaluate._global_correctness_metrics(1.5, 2.5, True, False) == (1.5, 2.5, True, False)


def test_screening_skips_trace_artifact_unless_explicitly_kept():
  evaluate = _load_evaluate_module()

  with patch.dict(os.environ, {"SKIP_DEEP_PROFILE": "1"}, clear=True):
    assert evaluate._keep_trace_artifact() is False

  with patch.dict(
    os.environ,
    {"SKIP_DEEP_PROFILE": "1", "EVAL_KEEP_TRACE": "1"},
    clear=True,
  ):
    assert evaluate._keep_trace_artifact() is True

  with patch.dict(os.environ, {}, clear=True):
    assert evaluate._keep_trace_artifact() is True


def test_local_correctness_compares_only_matching_addressable_shards():
  evaluate = _load_evaluate_module()

  class FakeDistributedArray:
    def __init__(self, parts):
      self.addressable_shards = [
        SimpleNamespace(index=(slice(start, stop),), data=np.asarray(data)) for start, stop, data in parts
      ]

    def __array__(self):
      raise RuntimeError("global value is not addressable")

  candidate = FakeDistributedArray(
    [
      (0, 2, [1.0, 2.0]),
      (4, 6, [5.0, 6.0]),
    ]
  )
  reference = FakeDistributedArray(
    [
      (0, 2, [1.0, 2.01]),
      (4, 6, [5.0, 6.0]),
    ]
  )

  max_diff, max_scaled, nonfinite, failed = evaluate._local_correctness_metrics(
    candidate, reference, rtol=0.0, atol=0.02
  )

  assert np.isclose(max_diff, 0.01)
  assert np.isclose(max_scaled, 0.5)
  assert nonfinite is False
  assert failed is False


def test_decode_batch_request():
  """decode_request should handle batch payload format."""
  from kernel_evolve.docker_evaluate_helpers import decode_request

  payload = {
    "batch": True,
    "reference_code": "def ref(): pass",
    "shapes": [{"M": 64}],
    "variants": [
      {"variant_id": "v1", "kernel_code": "def k1(): pass"},
    ],
  }
  b64 = base64.b64encode(json.dumps(payload).encode()).decode()
  result = decode_request(b64)
  assert result["batch"] is True
  assert len(result["variants"]) == 1


def test_batch_dispatch_calls_subprocess():
  """batch_dispatch should spawn subprocesses and collect results."""
  from kernel_evolve.docker_evaluate_helpers import batch_dispatch

  payload = {
    "batch": True,
    "reference_code": "def ref(): pass",
    "shapes": [{"M": 64}],
    "rtol": 0.01,
    "atol": 1.0,
    "variants": [
      {"variant_id": "v1-tiling", "kernel_code": "def k(): pass"},
      {"variant_id": "v2-pipe", "kernel_code": "def k(): pass"},
    ],
  }

  fake_result_json = json.dumps({"status": "SUCCESS", "speedup": 1.5, "latency_ms": 0.5, "fitness": 1.5})
  fake_stdout = f"some output\nEVAL_RESULT:{fake_result_json}\n"
  mock_completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=fake_stdout, stderr="")

  with patch("kernel_evolve.docker_evaluate_helpers.subprocess.run", return_value=mock_completed) as mock_run:
    results = batch_dispatch(payload, evaluator_script="/fake/evaluate.py", per_variant_timeout=60)

  assert mock_run.call_count == 2
  assert len(results) == 2
  assert all("EVAL_RESULT:" in line for line in results)


def test_batch_dispatch_forwards_the_correctness_method():
  """Each per-variant payload must carry the batch's method, not the default."""
  from kernel_evolve.docker_evaluate_helpers import batch_dispatch

  payload = {
    "batch": True,
    "reference_code": "def ref(): pass",
    "shapes": [{"M": 64}],
    "rtol": 0.02,
    "atol": 0.5,
    "method": "abs",
    "variants": [{"variant_id": "v1", "kernel_code": "def k(): pass"}],
  }

  mock_completed = subprocess.CompletedProcess(
    args=[], returncode=0, stdout='EVAL_RESULT:{"status": "SUCCESS"}', stderr=""
  )
  with patch("kernel_evolve.docker_evaluate_helpers.subprocess.run", return_value=mock_completed) as mock_run:
    batch_dispatch(payload, evaluator_script="/fake/evaluate.py", per_variant_timeout=60)

  argv = mock_run.call_args.args[0]
  sent = json.loads(base64.b64decode(argv[argv.index("--eval-payload") + 1]).decode())
  assert sent["method"] == "abs"
  assert sent["rtol"] == 0.02
  assert sent["atol"] == 0.5


def test_batch_dispatch_handles_subprocess_crash():
  """batch_dispatch should emit COMPILE_ERROR for crashed subprocesses."""
  from kernel_evolve.docker_evaluate_helpers import batch_dispatch

  payload = {
    "batch": True,
    "reference_code": "def ref(): pass",
    "shapes": [{"M": 64}],
    "variants": [
      {"variant_id": "v1-crash", "kernel_code": "raise Exception('boom')"},
    ],
  }

  mock_completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Segmentation fault")

  with patch("kernel_evolve.docker_evaluate_helpers.subprocess.run", return_value=mock_completed):
    results = batch_dispatch(payload, evaluator_script="/fake/evaluate.py", per_variant_timeout=60)

  assert len(results) == 1
  parsed = json.loads(results[0].split("EVAL_RESULT:", 1)[1])
  assert parsed["status"] == "COMPILE_ERROR"
  assert parsed["variant_id"] == "v1-crash"


def test_setup_dump_env_uses_variant_id():
  """_get_dump_dir should return variant-specific path when VARIANT_ID is set."""
  evaluate = _load_evaluate_module()

  # Without VARIANT_ID
  with patch.dict(os.environ, {}, clear=True):
    os.environ.pop("VARIANT_ID", None)
    assert evaluate._get_dump_dir() == "/tmp/ir_dumps"

  # With VARIANT_ID
  with patch.dict(os.environ, {"VARIANT_ID": "v1-tiling"}):
    assert evaluate._get_dump_dir() == "/tmp/ir_dumps/v1-tiling"


def test_batch_dispatch_handles_timeout():
  """batch_dispatch should handle subprocess timeout gracefully."""
  from kernel_evolve.docker_evaluate_helpers import batch_dispatch

  payload = {
    "batch": True,
    "reference_code": "def ref(): pass",
    "shapes": [{"M": 64}],
    "variants": [
      {"variant_id": "v1-slow", "kernel_code": "import time; time.sleep(999)"},
    ],
  }

  with patch(
    "kernel_evolve.docker_evaluate_helpers.subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd="python", timeout=60),
  ):
    results = batch_dispatch(payload, evaluator_script="/fake/evaluate.py", per_variant_timeout=60)

  assert len(results) == 1
  parsed = json.loads(results[0].split("EVAL_RESULT:", 1)[1])
  assert parsed["status"] == "COMPILE_ERROR"
  assert "timeout" in parsed["error"].lower()


# ── xprof extraction helper tests ──


def test_is_computation_event_matches():
  evaluate = _load_evaluate_module()
  assert evaluate._is_computation_event({"name": "jit_computation.123"}) is True
  assert evaluate._is_computation_event({"name": "jit(fn)"}) is True
  assert evaluate._is_computation_event({"name": "pallas_call"}) is True
  assert evaluate._is_computation_event({"name": "PALLAS_tpu"}) is True


def test_is_computation_event_rejects():
  evaluate = _load_evaluate_module()
  assert evaluate._is_computation_event({"name": "SyncWait"}) is False
  assert evaluate._is_computation_event({"name": "idle"}) is False
  assert evaluate._is_computation_event({}) is False
  assert evaluate._is_computation_event({"name": ""}) is False


def test_cluster_by_gap_basic():
  """5 events with 4 clear gaps should produce 5 clusters of 1 event each."""
  evaluate = _load_evaluate_module()
  events = [
    {"ts": 0, "dur": 10},
    {"ts": 1000, "dur": 10},
    {"ts": 2000, "dur": 10},
    {"ts": 3000, "dur": 10},
    {"ts": 4000, "dur": 10},
  ]
  clusters = evaluate._cluster_by_gap(events, 5)
  assert clusters is not None
  assert len(clusters) == 5
  assert all(len(c) == 1 for c in clusters)


def test_cluster_by_gap_groups_close_events():
  """Events close together should be grouped; large gaps separate iterations."""
  evaluate = _load_evaluate_module()
  # 2 iterations, each with 3 closely-spaced events, big gap between iterations
  events = [
    {"ts": 0, "dur": 5},
    {"ts": 10, "dur": 5},
    {"ts": 20, "dur": 5},
    # gap of 9975
    {"ts": 10000, "dur": 5},
    {"ts": 10010, "dur": 5},
    {"ts": 10020, "dur": 5},
  ]
  clusters = evaluate._cluster_by_gap(events, 2)
  assert clusters is not None
  assert len(clusters) == 2
  assert len(clusters[0]) == 3
  assert len(clusters[1]) == 3


def test_cluster_by_gap_too_few_events():
  evaluate = _load_evaluate_module()
  events = [{"ts": 0, "dur": 10}]
  assert evaluate._cluster_by_gap(events, 5) is None


def test_cluster_by_gap_empty():
  evaluate = _load_evaluate_module()
  assert evaluate._cluster_by_gap([], 3) is None


def test_extract_iteration_times_basic():
  """Should extract per-iteration times from well-separated computation events."""
  evaluate = _load_evaluate_module()
  tpu_pid = 42
  events = []
  for i in range(3):
    # Each iteration: one computation event of 500us, separated by 10000us gaps
    events.append(
      {
        "pid": tpu_pid,
        "ts": i * 10000,
        "dur": 500,
        "name": "jit_computation.fn",
      }
    )
  times = evaluate._extract_iteration_times(events, tpu_pid, n_iters=3)
  assert times is not None
  assert len(times) == 3
  assert all(abs(t - 0.5) < 0.01 for t in times)  # 500us = 0.5ms


def test_extract_iteration_times_wrong_pid():
  evaluate = _load_evaluate_module()
  events = [{"pid": 99, "ts": 0, "dur": 100, "name": "jit_computation.fn"}]
  assert evaluate._extract_iteration_times(events, tpu_pid=42, n_iters=1) is None


def test_extract_iteration_times_no_computation_events():
  evaluate = _load_evaluate_module()
  events = [{"pid": 42, "ts": 0, "dur": 100, "name": "SyncWait"}]
  assert evaluate._extract_iteration_times(events, tpu_pid=42, n_iters=1) is None


def test_extract_iteration_times_multi_event_iterations():
  """Each iteration may have multiple computation events."""
  evaluate = _load_evaluate_module()
  tpu_pid = 1
  # 2 iterations, each with 2 events, gap between iterations is large
  events = [
    # Iteration 1
    {"pid": tpu_pid, "ts": 0, "dur": 200, "name": "jit_computation.a"},
    {"pid": tpu_pid, "ts": 300, "dur": 200, "name": "pallas_call"},
    # Iteration 2 (gap of 9500)
    {"pid": tpu_pid, "ts": 10000, "dur": 200, "name": "jit_computation.a"},
    {"pid": tpu_pid, "ts": 10300, "dur": 200, "name": "pallas_call"},
  ]
  times = evaluate._extract_iteration_times(events, tpu_pid, n_iters=2)
  assert times is not None
  assert len(times) == 2
  # Each iteration spans from ts=0,dur=200 to ts=300+200=500 => 500us = 0.5ms
  assert abs(times[0] - 0.5) < 0.01
  assert abs(times[1] - 0.5) < 0.01


def test_extract_iteration_times_rejects_incomplete_outer_jit_trace():
  evaluate = _load_evaluate_module()
  events = []
  for i in range(3):
    events.extend(
      [
        {"pid": 1, "ts": i * 10000, "dur": 9000, "name": "jit(timed_compute)"},
        {"pid": 1, "ts": i * 10000 + 10, "dur": 8900, "name": "pallas_call"},
      ]
    )

  assert evaluate._extract_iteration_times(events, 1, n_iters=5) is None


def test_stage_benchmark_returns_benchmark_data(tmp_path):
  """stage_benchmark should return BenchmarkData with timing and profile metrics."""
  evaluate = _load_evaluate_module()

  globals_dict = {"optimized_compute": lambda M=1: M}

  # Mock jax.jit -> lower -> compile chain
  mock_lowered = MagicMock()
  mock_compiled = MagicMock()
  mock_compiled.return_value = 42
  mock_mem = MagicMock()
  mock_mem.peak_memory_in_bytes = 128 * 1024 * 1024  # 128 MB
  mock_compiled.memory_analysis.return_value = mock_mem
  mock_lowered.compile.return_value = mock_compiled
  mock_jitted = MagicMock()
  mock_jitted.lower.return_value = mock_lowered

  # Prepare xprof trace events that simulate 5 iterations
  trace_events = []
  for i in range(5):
    trace_events.append(
      {
        "pid": 1,
        "ts": i * 10000,
        "dur": 500 + i * 10,
        "name": "jit_computation.fn",
      }
    )
  trace_events.append({"args": {"name": "/device:TPU:0"}, "pid": 1})
  trace_events.append(
    {
      "pid": 1,
      "tid": 4294967295,
      "name": "MXU",
      "dur": 100,
      "args": {"% util": "50.0"},
    }
  )

  mock_xprof = MagicMock()
  mock_xprof.xspace_to_tool_data.return_value = (json.dumps({"traceEvents": trace_events}), None)

  # Create dummy .xplane.pb so os.walk finds it and proceeds to xprof parsing
  (tmp_path / "trace.xplane.pb").write_bytes(b"")

  # Patch module-level attributes on the loaded evaluate module
  with patch.object(evaluate, "jax") as mock_jax, patch.object(evaluate, "raw_to_tool_data", mock_xprof):
    mock_jax.jit.return_value = mock_jitted
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()
    mock_jax.profiler.stop_trace.side_effect = lambda: (tmp_path / "trace.xplane.pb").write_bytes(b"")

    result = evaluate.stage_benchmark(
      globals_dict,
      shapes=[{"M": 4}],
      trace_dir=str(tmp_path),
    )

  assert result["ok"] is True
  assert "benchmark" in result
  assert result["latency_ms"] > 0
  assert result["benchmark"]["peak_memory_mb"] == 128.0
  assert result["benchmark"]["timing_source"] == "xprof_clustered"
  assert len(result["benchmark"]["evaluation_times_ms"]) == 5
  assert "compute_ratio" in result
  assert "hw_utilization" in result


def test_stage_benchmark_wallclock_fallback(tmp_path):
  """stage_benchmark should fall back to wallclock timing when xprof fails."""
  evaluate = _load_evaluate_module()

  globals_dict = {"optimized_compute": lambda M=1: M}

  mock_lowered = MagicMock()
  mock_compiled = MagicMock()
  mock_compiled.return_value = 42
  mock_compiled.memory_analysis.side_effect = Exception("no memory_analysis")
  mock_lowered.compile.return_value = mock_compiled
  mock_jitted = MagicMock()
  mock_jitted.lower.return_value = mock_lowered

  with patch.object(evaluate, "jax") as mock_jax, patch.object(evaluate, "raw_to_tool_data", None):
    mock_jax.jit.return_value = mock_jitted
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()

    result = evaluate.stage_benchmark(
      globals_dict,
      shapes=[{"M": 4}],
      trace_dir=str(tmp_path),
    )

  assert result["ok"] is True
  assert result["benchmark"]["timing_source"] == "wallclock"
  assert result["benchmark"]["peak_memory_mb"] is None
  assert result["latency_ms"] > 0


def test_stage_benchmark_incomplete_xprof_falls_back_to_wallclock(tmp_path):
  """An incomplete trace must not be divided by the requested iteration count."""
  evaluate = _load_evaluate_module()

  globals_dict = {"optimized_compute": lambda M=1: M}

  mock_lowered = MagicMock()
  mock_compiled = MagicMock()
  mock_compiled.return_value = 42
  mock_compiled.memory_analysis.side_effect = Exception("no memory")
  mock_lowered.compile.return_value = mock_compiled
  mock_jitted = MagicMock()
  mock_jitted.lower.return_value = mock_lowered

  # Events that can't be clustered into 5 groups (only 2 events, need 5 iters)
  trace_events = [
    {"pid": 1, "ts": 0, "dur": 2500, "name": "jit_computation.fn"},
    {"pid": 1, "ts": 3000, "dur": 2500, "name": "jit_computation.fn"},
    {"args": {"name": "/device:TPU:0"}, "pid": 1},
  ]

  mock_xprof = MagicMock()
  mock_xprof.xspace_to_tool_data.return_value = (json.dumps({"traceEvents": trace_events}), None)

  # Create dummy .xplane.pb so os.walk finds it
  (tmp_path / "trace.xplane.pb").write_bytes(b"")

  with patch.object(evaluate, "jax") as mock_jax, patch.object(evaluate, "raw_to_tool_data", mock_xprof):
    mock_jax.jit.return_value = mock_jitted
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()
    mock_jax.profiler.stop_trace.side_effect = lambda: (tmp_path / "trace.xplane.pb").write_bytes(b"")

    result = evaluate.stage_benchmark(
      globals_dict,
      shapes=[{"M": 4}],
      trace_dir=str(tmp_path),
      n_iters=5,
    )

  assert result["ok"] is True
  assert result["benchmark"]["timing_source"] == "wallclock"


def test_stage_benchmark_no_compute_fn():
  """stage_benchmark should return error when no compute function found."""
  evaluate = _load_evaluate_module()
  result = evaluate.stage_benchmark({}, shapes=[{"M": 4}])
  assert result["ok"] is False
  assert "No compute function" in result["error"]


def test_jax_not_imported_at_module_level():
  """jax must NOT be imported at module level so _setup_dump_env() can
  set XLA_FLAGS/LIBTPU_INIT_ARGS before libtpu reads them (FP5)."""
  evaluate = _load_evaluate_module()
  assert evaluate.jax is None
