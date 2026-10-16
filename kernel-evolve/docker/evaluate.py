"""Three-stage kernel evaluator for TPU. Runs inside a K8s Pod.

Stages: compile -> correctness -> benchmark (+ optional deep profile).
"""

import argparse
import base64
import json
import math
import os
import sys
import time
import traceback
from typing import Any

try:
  from google.cloud import storage
except ImportError:
  storage = None

from pathlib import Path

import numpy as np

# CRITICAL: Do NOT import jax at module level. IR dump flags
# (XLA_FLAGS, LIBTPU_INIT_ARGS) must be set via _setup_dump_env()
# BEFORE jax loads, because importing jax triggers libtpu
# initialization which reads these flags once. main() imports
# jax after calling _setup_dump_env(). See FP5 in AGENT.md.
jax = None

try:
  from xprof.convert import raw_to_tool_data
except ImportError:
  raw_to_tool_data = None


def upload_to_gcs(
  job_name: str,
  artifacts: dict[str, str],
  bucket_name: str = "",
) -> dict[str, Any]:
  """Upload profile artifacts to GCS. Non-fatal — never raises.

  There is no default bucket: when ``bucket_name`` is empty (``GCS_BUCKET`` unset),
  the upload is skipped and ``ok`` is False.
  """
  uploaded: list[str] = []
  if not bucket_name:
    if artifacts:
      print("GCS_BUCKET not set, skipping GCS artifact upload", file=sys.stderr)
    return {"ok": False, "uploaded": uploaded, "gcs_prefix": ""}
  prefix = f"gs://{bucket_name}/{job_name}"
  if not artifacts:
    return {"ok": False, "uploaded": uploaded, "gcs_prefix": prefix}
  if storage is None:
    print("google-cloud-storage not installed, skipping GCS upload", file=sys.stderr)
    return {"ok": False, "uploaded": uploaded, "gcs_prefix": prefix}
  try:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for name, local_path in artifacts.items():
      if not os.path.exists(local_path):
        continue
      try:
        blob = bucket.blob(f"{job_name}/{name}")
        blob.upload_from_filename(local_path)
        uploaded.append(name)
      except Exception as e:
        print(f"GCS upload failed for {name}: {e}", file=sys.stderr)
  except Exception as e:
    print(f"GCS client init failed: {e}", file=sys.stderr)
  return {"ok": len(uploaded) > 0, "uploaded": uploaded, "gcs_prefix": prefix}


def decode_request(b64_payload: str) -> dict[str, Any]:
  return json.loads(base64.b64decode(b64_payload).decode())


def _resolve_compute_fn(
  exec_globals: dict[str, Any],
  *,
  allow_reference: bool = False,
):
  names = ["optimized_compute", "kernel_fn"]
  if allow_reference:
    names.extend(["simple_compute", "reference_fn"])
  for name in names:
    fn = exec_globals.get(name)
    if fn is not None:
      return fn
  return None


def _resolve_timed_pair(exec_globals: dict[str, Any]):
  """The split benchmark protocol: make_inputs(**shape) -> tuple of arrays, and
  timed_compute(*arrays) -> result. When a module defines both, the benchmark stage
  times only timed_compute on pre-materialized inputs -- the legacy protocol jits
  optimized_compute(**shape) whole, which folds test-data RNG and dtype casts into
  the measurement (observed as a ~10 ms constant on some workloads, which
  compresses every fitness ratio toward 1). Correctness still goes through optimized_compute.
  """
  fn = exec_globals.get("timed_compute")
  mk = exec_globals.get("make_inputs")
  if fn is not None and mk is not None:
    return fn, mk
  return None


def _candidate_function_names(kernel_code: str, reference_code: str = "") -> set[str]:
  """Functions the candidate defines and the reference does not.

  The reference runs in the same process, so its compilations dump into the same
  directories. These names tell the candidate's kernels and modules apart from it.
  """
  import ast

  def defined(code):
    try:
      tree = ast.parse(code or "")
    except SyntaxError:
      return set()
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

  return defined(kernel_code) - defined(reference_code)


def _mosaic_kernel_name(path: str):
  """The kernel function a Mosaic dump file was lowered from, or None."""
  import re

  try:
    with open(path, errors="replace") as fh:
      head = fh.read(4096)
  except OSError:
    return None
  names = re.findall(r"func\.func @([A-Za-z_][A-Za-z0-9_]*)", head)
  return next((n for n in names if not n.startswith("transform_")), None)


def _has_tpu() -> bool:
  if jax is None:
    return False
  try:
    devices = jax.devices()
    print(f"JAX devices: {devices}", file=sys.stderr)
    return any(d.platform == "tpu" for d in devices)
  except Exception as e:
    print(f"TPU detection error: {e}", file=sys.stderr)
    return False


def stage_compile(kernel_code: str) -> dict[str, Any]:
  try:
    exec_globals = {}
    exec(kernel_code, exec_globals)
    return {"ok": True, "globals": exec_globals}
  except Exception:
    return {"ok": False, "error": f"Compilation error: {traceback.format_exc()}"}


def _shard_index_key(index) -> tuple:
  """Turn a JAX shard index into a stable, comparable Python key."""
  out = []
  for item in index:
    if isinstance(item, slice):
      out.append((item.start, item.stop, item.step))
    else:
      out.append(item)
  return tuple(out)


CORRECTNESS_METHODS = ("allclose", "abs")


def _correctness_bound(ref_part, rtol, atol, method):
  """Per-element tolerance the absolute difference has to stay under.

  ``allclose`` is the ``np.allclose`` bar ``atol + rtol * |reference|``; ``abs``
  is ``atol`` alone, which keeps a large reference magnitude from buying the
  candidate extra room. An unknown method raises instead of defaulting.
  """
  if method == "abs":
    return np.float64(atol)
  if method != "allclose":
    raise ValueError(f"unknown correctness method {method!r}, expected one of {CORRECTNESS_METHODS}")
  return atol + rtol * np.abs(np.asarray(ref_part, dtype=np.float64))


def _json_safe_number(value):
  """Emit null for a value JSON cannot represent.

  `json.dumps` writes ``Infinity`` and ``NaN`` as bare tokens, which non-Python
  readers reject. The verdict is carried by ``status`` and the error string, so a
  null here loses no decision-relevant information.
  """
  return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def _scaled_diff(diff, bound):
  """Difference measured in units of its own tolerance: <= 1 passes, > 1 fails.

  ``max_diff`` alone cannot say whether a run passed: under ``allclose`` the bar
  varies per element. A zero bound admits only an exact match.
  """
  positive = np.asarray(bound) > 0
  scaled = diff / np.where(positive, bound, 1.0)
  return np.where(positive, scaled, np.where(diff > 0, np.inf, 0.0))


def _absolute_difference(kernel_part, ref_part):
  """``|candidate - reference|`` as (reported, compared).

  Converting operands near ``2**53`` to float64 rounds them onto the same value and
  hides a real difference, so integers are subtracted in integer space. A difference
  that large also cannot be converted for the comparison itself: ``2**53 + 1`` would
  round back onto ``2**53`` and slip under a tolerance of ``2**53``. The verdict
  therefore uses the exact array, and only the reported number is converted.
  """
  if kernel_part.dtype.kind in "iu" and ref_part.dtype.kind in "iu":
    if max(kernel_part.dtype.itemsize, ref_part.dtype.itemsize) <= 4:
      # Every 32-bit difference fits int64 without overflow and float64 exactly.
      exact = np.abs(kernel_part.astype(np.int64) - ref_part.astype(np.int64))
    else:
      # 64-bit integers: Python ints, so neither overflow nor rounding applies, and
      # comparisons against a float bound stay exact. The object array costs roughly
      # 2.5x the memory of the vectorized path, which only 64-bit integer outputs pay.
      exact = np.abs(kernel_part.astype(object) - ref_part.astype(object))
    return np.asarray(exact, dtype=np.float64), exact
  diff = np.abs(np.asarray(kernel_part, dtype=np.float64) - np.asarray(ref_part, dtype=np.float64))
  return diff, diff


def _local_correctness_metrics(kernel_out, ref_out, rtol, atol, method="allclose"):
  """Compare only process-local data when outputs span multiple controllers.

  A global ``jax.Array`` on a multi-host slice cannot be converted to NumPy from
  one controller.  Candidate and reference outputs are required to expose the
  same global shard indices; each controller therefore compares its addressable
  shards and a later host collective combines the two scalar metrics.
  """
  kernel_shards = getattr(kernel_out, "addressable_shards", None)
  ref_shards = getattr(ref_out, "addressable_shards", None)
  if kernel_shards is None and ref_shards is None:
    pairs = [(np.asarray(kernel_out), np.asarray(ref_out))]
  elif kernel_shards is None or ref_shards is None:
    raise ValueError("candidate and reference must use the same distributed output layout")
  else:
    kernel_by_index = {_shard_index_key(shard.index): np.asarray(shard.data) for shard in kernel_shards}
    ref_by_index = {_shard_index_key(shard.index): np.asarray(shard.data) for shard in ref_shards}
    if kernel_by_index.keys() != ref_by_index.keys():
      raise ValueError(
        "candidate and reference have different process-local shard indices: "
        f"candidate={sorted(kernel_by_index)}, reference={sorted(ref_by_index)}"
      )
    pairs = [(kernel_by_index[key], ref_by_index[key]) for key in kernel_by_index]

  local_max = 0.0
  local_scaled = 0.0
  local_nonfinite = False
  local_failed = False
  for kernel_part, ref_part in pairs:
    if kernel_part.shape != ref_part.shape:
      raise ValueError(f"candidate and reference shard shapes differ: {kernel_part.shape} != {ref_part.shape}")
    if (kernel_part.dtype.kind in "iu") != (ref_part.dtype.kind in "iu"):
      # One side integral and the other not: the difference cannot be taken exactly, so
      # a large integer would round onto the reference and meet the bound. A kernel whose
      # output dtype disagrees with the reference this way is a structural mismatch.
      raise ValueError(
        f"candidate and reference output dtypes are not comparable: {kernel_part.dtype} vs {ref_part.dtype}"
      )
    kernel_f = np.asarray(kernel_part, dtype=np.float64)
    ref_f = np.asarray(ref_part, dtype=np.float64)
    diff, compared = _absolute_difference(kernel_part, ref_part)
    finite = np.isfinite(diff)
    bound = _correctness_bound(ref_f, rtol, atol, method)
    if diff.size:
      # The reported maxima stay finite so the result stays serializable; a NaN on
      # either side, or one side infinite against a different value, is reported
      # through the non-finite flag instead. Two identical infinities are equal.
      finite_diff = np.where(finite, diff, 0.0)
      local_max = max(local_max, float(np.max(finite_diff)))
      local_scaled = max(local_scaled, float(np.max(_scaled_diff(finite_diff, bound))))
      mismatched_nonfinite = bool(np.any(~finite & (kernel_f != ref_f)))
      local_nonfinite = local_nonfinite or mismatched_nonfinite
      local_failed = local_failed or mismatched_nonfinite
      # Safety net for any dtype whose conversion is still lossy: a difference that
      # collapses to zero is only a match when the originals are equal exactly.
      local_failed = local_failed or bool(np.any((diff == 0) & (kernel_part != ref_part)))
    local_failed = local_failed or not bool(np.all(np.where(finite, compared <= bound, True)))
  return local_max, local_scaled, local_nonfinite, local_failed


_METRIC_COUNT = 4
_METRIC_WORDS = _METRIC_COUNT * 2


def _global_correctness_metrics(local_max, local_scaled, local_nonfinite, local_failed):
  """All-gather the four scalars so a non-primary host cannot hide a mismatch.

  JAX with x64 disabled turns a float64 array into device float32, which would round
  another host's maxima. The metrics travel as int32 words holding the float64 bits
  instead; every host of a slice runs the same architecture, so the reinterpretation
  is symmetric.
  """
  if jax is None or jax.process_count() <= 1:
    return local_max, local_scaled, local_nonfinite, local_failed
  from jax.experimental import multihost_utils

  metrics = np.asarray([local_max, local_scaled, float(local_nonfinite), float(local_failed)], dtype=np.float64)
  words = np.frombuffer(metrics.tobytes(), dtype=np.int32)
  gathered = np.asarray(multihost_utils.process_allgather(words, tiled=False), dtype=np.int32).reshape(
    -1, _METRIC_WORDS
  )
  values = np.frombuffer(gathered.tobytes(), dtype=np.float64).reshape(-1, _METRIC_COUNT)
  return (
    float(np.max(values[:, 0])),
    float(np.max(values[:, 1])),
    bool(np.any(values[:, 2] != 0)),
    bool(np.any(values[:, 3] != 0)),
  )


def stage_correctness(kernel_code, reference_code, shapes, rtol, atol, exec_globals, method="allclose"):
  """Compare candidate against reference on every shape, stopping at the first failure.

  The reported maxima therefore cover the shapes evaluated up to and including the
  failing one, not the whole list.
  """
  try:
    if method not in CORRECTNESS_METHODS:
      return {
        "ok": False,
        "error": f"unknown correctness method {method!r}, expected one of {CORRECTNESS_METHODS}",
        "max_diff": 0.0,
        "max_scaled_diff": 0.0,
      }
    ref_globals = {}
    exec(reference_code, ref_globals)
    kernel_fn = _resolve_compute_fn(exec_globals)
    ref_fn = _resolve_compute_fn(ref_globals, allow_reference=True)
    if kernel_fn is None or ref_fn is None:
      return {
        "ok": False,
        "error": "Could not find kernel_fn/reference_fn",
        "max_diff": 0.0,
        "max_scaled_diff": 0.0,
      }
    max_diff = 0.0
    max_scaled = 0.0
    for shape in shapes:
      kernel_out = kernel_fn(**shape)
      ref_out = ref_fn(**shape)
      local = _local_correctness_metrics(kernel_out, ref_out, rtol, atol, method)
      shape_max, shape_scaled, nonfinite, failed = _global_correctness_metrics(*local)
      max_diff = max(max_diff, shape_max)
      max_scaled = max(max_scaled, shape_scaled)
      if failed:
        return {
          "ok": False,
          "error": (
            f"Correctness failed for shape {shape}: max_diff={shape_max}, "
            f"max_diff/tolerance={shape_scaled} "
            f"(method={method}, rtol={rtol}, atol={atol})"
            + (
              "; candidate and reference disagree on non-finite values, which the finite maxima above do not measure"
              if nonfinite
              else ""
            )
          ),
          "max_diff": max_diff,
          "max_scaled_diff": max_scaled,
        }
    return {"ok": True, "max_diff": max_diff, "max_scaled_diff": max_scaled}
  except Exception:
    return {
      "ok": False,
      "error": f"Correctness error: {traceback.format_exc()}",
      "max_diff": 0.0,
      "max_scaled_diff": 0.0,
    }


def _is_computation_event(event):
  """Check if a trace event represents a TPU computation."""
  name = event.get("name") or ""
  return "jit_computation" in name or "jit(" in name or "pallas" in name.lower()


def _cluster_by_gap(events, n_clusters):
  """Cluster sorted events into n_clusters groups using largest gaps.

  Computes gaps between consecutive events, finds the (n_clusters - 1)
  largest gaps as iteration boundaries, and splits events at those points.

  Returns list of event lists, or None if clustering fails.
  """
  if len(events) < n_clusters:
    return None

  gaps = []
  for i in range(len(events) - 1):
    gap = events[i + 1]["ts"] - (events[i]["ts"] + events[i]["dur"])
    gaps.append((gap, i))

  gaps.sort(key=lambda x: x[0], reverse=True)
  boundary_indices = sorted(g[1] for g in gaps[: n_clusters - 1])

  clusters = []
  prev = 0
  for idx in boundary_indices:
    clusters.append(events[prev : idx + 1])
    prev = idx + 1
  clusters.append(events[prev:])

  if len(clusters) != n_clusters or any(len(c) == 0 for c in clusters):
    return None

  return clusters


def _extract_iteration_times(events, tpu_pid, n_iters=5):
  """Extract per-iteration device times from xprof trace events.

  xprof trace events use microsecond units for ts and dur fields
  (Chrome Trace Format convention). Returns milliseconds, or None on failure.
  """
  # The device trace normally exposes one outer jit scope per measured call.
  # Prefer those durations directly. A capped trace can contain two or three
  # outer calls plus enough nested jit/pallas events for gap clustering to
  # fabricate five "iterations"; that produced a false 3.91 ms result for a
  # roughly 9.5 ms SR reduce-scatter candidate.
  outer_names = {
    "jit(timed_compute)",
    "jit(optimized_compute)",
    "jit(simple_compute)",
  }
  outer = sorted(
    [e for e in events if e.get("pid") == tpu_pid and "dur" in e and e.get("name") in outer_names],
    key=lambda e: e["ts"],
  )
  if outer:
    if len(outer) < n_iters:
      return None
    return [e["dur"] / 1000.0 for e in outer[-n_iters:]]

  comp_events = sorted(
    [e for e in events if e.get("pid") == tpu_pid and "dur" in e and _is_computation_event(e)], key=lambda e: e["ts"]
  )

  if not comp_events:
    return None

  clusters = _cluster_by_gap(comp_events, n_iters)
  if clusters is None:
    return None

  times_ms = []
  for cluster in clusters:
    start = cluster[0]["ts"]
    end = max(e["ts"] + e["dur"] for e in cluster)
    times_ms.append((end - start) / 1000.0)
  return times_ms


def stage_benchmark(exec_globals, shapes, trace_dir="/tmp/xplane_trace", warmup=3, n_iters=5):
  # A stale trace from a previous variant parses as this variant's latency, and
  # concurrent evaluators on one host race on a shared path -- so the dir is
  # per-process-overridable and always cleared first.
  trace_dir = os.environ.get("TRACE_DIR", trace_dir)
  import shutil

  shutil.rmtree(trace_dir, ignore_errors=True)
  """Merged performance + profile stage using hermetic xprof timing.

  Execution flow:
  1. Resolve kernel_fn
  2. Compile with timing isolation (jax.jit().lower().compile())
  3. Peak memory via memory_analysis()
  4. Warmup (warmup iterations, outside profiler)
  5. Profiled execution (n_iters iterations under xprof)
  6. Parse .xplane.pb for per-iteration device times + hw utilization
  7. Return BenchmarkData + profile metrics

  Falls back to wallclock timing if xprof fails.
  """
  try:
    shape = shapes[0]
    call_args = ()
    pair = _resolve_timed_pair(exec_globals)
    if pair is not None:
      kernel_fn, make_inputs = pair
      call_args = tuple(make_inputs(**shape))
      jax.block_until_ready(call_args)
      jitted = jax.jit(kernel_fn)
    else:
      kernel_fn = _resolve_compute_fn(exec_globals, allow_reference=True)
      if kernel_fn is None:
        return {"ok": False, "error": "No compute function found"}
      # ── Step 2: Compilation isolation ──
      static_names = list(shape.keys())
      jitted = jax.jit(kernel_fn, static_argnames=static_names)

    t0 = time.perf_counter()
    lowered = jitted.lower(*call_args) if call_args else jitted.lower(**shape)
    lower_time_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_time_ms = (time.perf_counter() - t0) * 1000

    # ── Step 3: Peak memory ──
    peak_memory_mb = None
    try:
      mem = compiled.memory_analysis()
      if hasattr(mem, "peak_memory_in_bytes"):
        peak_memory_mb = mem.peak_memory_in_bytes / (1024 * 1024)
    except Exception:
      pass

    # ── Step 4: Warmup ──
    for _ in range(warmup):
      out = compiled(*call_args)
      if hasattr(out, "block_until_ready"):
        out.block_until_ready()

    # ── Step 5: Profiled execution with wallclock fallback ──
    Path(trace_dir).mkdir(parents=True, exist_ok=True)

    wallclock_times = []
    xprof_ok = False
    events = []

    if raw_to_tool_data is not None:
      try:
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.host_tracer_level = 2
        options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

        jax.profiler.start_trace(trace_dir, profiler_options=options)
        try:
          for _ in range(n_iters):
            t0 = time.perf_counter()
            out = compiled(*call_args)
            if hasattr(out, "block_until_ready"):
              out.block_until_ready()
            wallclock_times.append((time.perf_counter() - t0) * 1000)
        finally:
          jax.profiler.stop_trace()
        xprof_ok = True
      except Exception as e:
        print(f"xprof profiling failed: {e}", file=sys.stderr)

    # Wallclock-only fallback if xprof profiling failed
    if not xprof_ok:
      for _ in range(n_iters):
        t0 = time.perf_counter()
        out = compiled(*call_args)
        if hasattr(out, "block_until_ready"):
          out.block_until_ready()
        wallclock_times.append((time.perf_counter() - t0) * 1000)

    # ── Step 6: Parse xprof trace ──
    compute_ratio = None
    memory_transfer_ratio = None
    hw_utilization = None
    profile_diag = {}
    trace_events_path = None
    timing_source = "wallclock"
    eval_times = wallclock_times

    if xprof_ok:
      xplane_path = None
      for root, _dirs, files in os.walk(trace_dir):
        for f in files:
          if f.endswith(".xplane.pb"):
            xplane_path = os.path.join(root, f)
            break
        if xplane_path:
          break

      if xplane_path is not None:
        try:
          tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data([xplane_path], "trace_viewer", {})
          trace_data = json.loads(tool_data_result)
          events = trace_data.get("traceEvents", [])

          trace_events_path = os.path.join(trace_dir, "trace_events.json")
          with open(trace_events_path, "w") as f:
            json.dump(events, f)

          # Find TPU device pid
          tpu_pid = None
          for event in events:
            if "args" in event and event["args"].get("name") == "/device:TPU:0":
              tpu_pid = event.get("pid")
              break
          if tpu_pid is None:
            for event in events:
              if "args" in event:
                name = event["args"].get("name", "")
                if name.startswith("/device:TPU:"):
                  tpu_pid = event.get("pid")
                  break

          if tpu_pid is not None:
            # Extract per-iteration device times
            xprof_times = _extract_iteration_times(events, tpu_pid, n_iters)
            if xprof_times is not None:
              eval_times = xprof_times
              timing_source = "xprof_clustered"
            # If the trace does not contain all measured invocations, keep the
            # synchronized host wallclock samples collected around each
            # compiled call. Dividing an incomplete trace span by n_iters is
            # not a timing fallback.

            # Extract compute_ratio from sync/idle events
            events_for_tpu = [e for e in events if e.get("pid") == tpu_pid]
            computation_events = [e for e in events_for_tpu if "dur" in e and _is_computation_event(e)]
            if len(computation_events) >= 2:
              trace_start = computation_events[0]["ts"]
              trace_end = computation_events[-1]["ts"] + computation_events[-1]["dur"]
              sync_wait_total = 0
              for event in events_for_tpu:
                if "dur" not in event:
                  continue
                evt_start = event["ts"]
                evt_end = evt_start + event["dur"]
                if evt_start >= trace_start and evt_end <= trace_end:
                  name = event.get("name") or ""
                  if "SyncWait" in name or "idle" in name.lower():
                    sync_wait_total += event["dur"]
              total_time = trace_end - trace_start
              if total_time > 0:
                ratio = sync_wait_total / total_time
                compute_ratio = 1.0 - ratio
                memory_transfer_ratio = ratio

            hw_utilization = parse_hw_utilization(events, tpu_pid)

            # Diagnostics
            process_names = {}
            for event in events:
              if "args" in event and "name" in event["args"]:
                pid_val = event.get("pid")
                if pid_val not in process_names:
                  process_names[pid_val] = []
                name_val = event["args"]["name"]
                if name_val not in process_names[pid_val]:
                  process_names[pid_val].append(name_val)

            profile_diag = {
              "process_names": {str(k): v for k, v in process_names.items()},
              "selected_pid": tpu_pid,
              "total_events": len(events),
              "tpu_events": len(events_for_tpu),
              "computation_events": len(computation_events),
              "trace_event_cap_suspected": len(events) >= 1_000_000,
            }
        except Exception as e:
          print(f"xprof trace parsing failed: {e}", file=sys.stderr)
          traceback.print_exc(file=sys.stderr)

    # ── Step 7: Build result ──
    # Local import: kernel_evolve may not be on PYTHONPATH during module load in Docker
    from kernel_evolve.evaluator import BenchmarkData

    benchmark = BenchmarkData(
      lower_time_ms=lower_time_ms,
      compile_time_ms=compile_time_ms,
      evaluation_times_ms=tuple(eval_times),
      peak_memory_mb=peak_memory_mb,
      timing_source=timing_source,
    )

    return {
      "ok": True,
      "benchmark": benchmark.to_dict(),
      "latency_ms": benchmark.median_ms,
      "compute_ratio": compute_ratio,
      "memory_transfer_ratio": memory_transfer_ratio,
      "hw_utilization": hw_utilization,
      "diagnostics": profile_diag,
      "_trace_events_path": trace_events_path,
    }
  except Exception:
    return {"ok": False, "error": f"Benchmark error: {traceback.format_exc()}"}


def parse_hw_utilization(events, tpu_pid):
  """Extract per-hardware-unit utilization from xprof counter events.

  The _counters_ thread (tid=0xFFFFFFFF) contains time-windowed utilization
  percentages for MXU, Scalar ALU, Vector ALU, Vector Load/Store, Vector EUP,
  plus absolute fill/spill counts. Returns a dict of time-weighted averages
  or None if no counter events are found.
  """
  COUNTER_TID = 4294967295  # 0xFFFFFFFF, _counters_ thread
  UTIL_UNITS = {
    "MXU": "mxu_util_pct",
    "Vector ALU": "vector_alu_util_pct",
    "Scalar ALU": "scalar_alu_util_pct",
    "Vector Load": "vector_load_util_pct",
    "Vector Store": "vector_store_util_pct",
    "Vector EUP": "vector_eup_util_pct",
  }
  COUNTER_UNITS = {
    "Vector Fills": ("vector_fills", "fills"),
    "Vector Spills": ("vector_spills", "spills"),
  }

  counter_events = [e for e in events if e.get("pid") == tpu_pid and e.get("tid") == COUNTER_TID and "dur" in e]
  if not counter_events:
    return None

  result = {}
  for xprof_name, key in UTIL_UNITS.items():
    weighted_sum = 0.0
    total_dur = 0.0
    for e in counter_events:
      if e.get("name") == xprof_name:
        dur = e["dur"]
        util = float(e.get("args", {}).get("% util", 0))
        weighted_sum += util * dur
        total_dur += dur
    result[key] = weighted_sum / total_dur if total_dur > 0 else 0.0

  for xprof_name, (key, arg_key) in COUNTER_UNITS.items():
    total = 0
    for e in counter_events:
      if e.get("name") == xprof_name:
        total += int(e.get("args", {}).get(arg_key, 0))
    result[key] = total

  return result


def stage_profile_deep(exec_globals, shapes, dump_dir=None, kernel_names=None):
  """Stage 4b: Deep IR profiling via HLO/LLO/Mosaic dumps.

  Dump flags are set at process startup (before JAX init) via
  _setup_dump_env(). This function only parses the dump files that were
  generated during earlier stages.

  For FLOPs: Pallas GMM matmuls are inside tpu_custom_call and invisible
  to HLO dot-pattern parsing. FLOPs are computed from the eval shapes
  directly (fwd + bwd_gmm + tgmm = 3 matmuls per shape).

  kernel_names: functions defined only by the candidate (see
  _candidate_function_names). When given, LLO and HLO artifacts are taken from the
  candidate's own kernels and modules, not the reference's, and
  ``artifact_binding`` records whether that succeeded.

  Self-contained — does NOT import from kernel_evolve.profiler.
  Non-fatal: returns ok=False on failure without stopping the evaluation pipeline.
  """
  try:
    if dump_dir is None:
      dump_dir = _get_dump_dir()
    import glob
    import re

    _DTYPE_BYTES = {
      "f32": 4,
      "float32": 4,
      "f16": 2,
      "float16": 2,
      "bf16": 2,
      "bfloat16": 2,
      "f8e5m2": 1,
      "f8e4m3fn": 1,
      "s32": 4,
      "int32": 4,
      "s16": 2,
      "int16": 2,
      "s8": 1,
      "int8": 1,
    }

    dump_path = Path(dump_dir)
    hlo_dir = dump_path / "hlo"
    llo_dir = dump_path / "llo"

    # ── Compute FLOPs from eval shapes ──────────────────────────────
    # GMM fwd+bwd has 3 matmul passes per shape:
    #   fwd:      lhs[M,K] @ rhs[G,K,N] -> [M,N]   => 2*M*K*N*G
    #   bwd_gmm:  dlhs_dout[M,N] @ rhs[G,K,N]^T     => 2*M*N*K*G
    #   tgmm:     lhs_t[K,M] @ drhs_dout[M,N]        => 2*K*M*N*G
    # Total per shape: 6 * M * K * N * G
    # Only the first shape is benchmarked for performance.
    flops = None
    if shapes:
      s = shapes[0]
      M = s.get("M", 0)
      K = s.get("K", 0)
      N = s.get("N", 0)
      G = s.get("G", 1)
      if M and K and N:
        flops = 6.0 * M * K * N * G

    # Log dump file counts and top files for debugging
    for label, d in [("hlo", hlo_dir), ("llo", llo_dir), ("mosaic", dump_path / "mosaic")]:
      d_path = Path(d)
      if not d_path.exists():
        print(f"Dump dir [{label}]: does not exist", file=sys.stderr)
        continue
      all_files = [(f.stat().st_size, f) for f in d_path.rglob("*") if f.is_file()]
      all_files.sort(key=lambda x: -x[0])
      print(f"Dump dir [{label}]: {len(all_files)} files", file=sys.stderr)
      for sz, fp in all_files[:5]:
        print(f"  {sz:>10} bytes  {fp.name}", file=sys.stderr)

    # ── Parse LLO dumps ──────────────────────────────────────────────
    # LLO files are named: {hash}-{op_name}-{pass_num}-{pass_name}.txt
    # The actual kernel body is in files with "pallas" in op_name (not
    # "custom-call" which is just the wrapper). We want the final pass
    # with the most instructions (largest file).
    vliw_bundle_count = None
    mxu_utilization = None
    mxu0_count = 0
    mxu1_count = 0

    # On v7x, actual VLIW LLO is in the mosaic/ dump directory
    # (post-lower-to-llo, post-eliminate-llo-extensions), while the
    # llo/ directory contains scheduled HLO and XLA JF passes. Check
    # mosaic/ first for VLIW content, fall back to llo/ if needed.
    mosaic_dir = dump_path / "mosaic"
    llo_search_dirs = [mosaic_dir, llo_dir]

    llo_files = []
    for search_dir in llo_search_dirs:
      llo_files += glob.glob(str(search_dir / "**" / "*.txt"), recursive=True)
      llo_files += glob.glob(str(search_dir / "**" / "*.llo"), recursive=True)

    # Parse filename pattern: {hash}-{op_name}-{pass_num}-{pass_name}.txt
    # Also accepts .llo extension and dotted naming conventions.
    # Mosaic files: {hash}-{pass_num}-mosaic-dump-{name}-{pass_name}.txt
    file_re = re.compile(r"^\d+-(.+?)-(\d+)-(.+)\.(?:txt|llo)$")

    # Strategy: find files containing actual VLIW instructions.
    # Mosaic "post-lower-to-llo" / "post-eliminate-llo-extensions" files
    # contain VLIW LLO. Files starting with "HloModule" are scheduled HLO.
    pallas_candidates = []
    mosaic_candidates = []
    other_candidates = []
    for f in llo_files:
      basename = os.path.basename(f)
      # Mosaic dump files with LLO content
      if "mosaic-dump" in basename and ("lower-to-llo" in basename or "eliminate-llo" in basename):
        mosaic_candidates.append((f, basename, 0, "mosaic"))
        continue
      m = file_re.match(basename)
      if not m:
        other_candidates.append((f, basename, 0, ""))
        continue
      op_name = m.group(1)
      pass_num = int(m.group(2))
      pass_name = m.group(3)
      entry = (f, op_name, pass_num, pass_name)
      if "pallas" in op_name.lower():
        pallas_candidates.append(entry)
      elif "custom-call" in op_name or "custom_call" in op_name:
        other_candidates.append(entry)

    # Priority: mosaic VLIW > pallas candidates > other candidates
    # Pick the largest file in the highest-priority non-empty group.
    # The reference's own Mosaic lowering (XLA can lower jnp code to Mosaic, e.g.
    # attention as func @main) lands here too and is often the largest file, so
    # restrict the pick to kernels the candidate defines when they are known.
    best_file = None
    llo_kernel = None
    llo_binding = "unverified"
    if mosaic_candidates:
      pick_pool = mosaic_candidates
      if kernel_names:
        own = [e for e in mosaic_candidates if _mosaic_kernel_name(e[0]) in kernel_names]
        if own:
          pick_pool = own
          llo_binding = "candidate"
        else:
          print(
            "LLO: WARNING no Mosaic dump matches a candidate kernel "
            f"({sorted(kernel_names)}); the pick may describe the reference",
            file=sys.stderr,
          )
      pick_pool.sort(key=lambda e: os.path.getsize(e[0]), reverse=True)
      best_file = pick_pool[0][0]
      llo_kernel = _mosaic_kernel_name(best_file)
      print(
        f"LLO: picked mosaic candidate: {os.path.basename(best_file)} ({os.path.getsize(best_file)} bytes)",
        file=sys.stderr,
      )
    elif pallas_candidates:
      pallas_candidates.sort(key=lambda e: os.path.getsize(e[0]), reverse=True)
      best_file = pallas_candidates[0][0]
      print(
        f"LLO: picked pallas candidate: {os.path.basename(best_file)} ({os.path.getsize(best_file)} bytes)",
        file=sys.stderr,
      )
    elif other_candidates:
      # Skip files that are actually scheduled HLO (contain 'HloModule')
      real_llo = []
      for entry in other_candidates:
        with open(entry[0]) as fh:
          head = fh.read(256)
        if "HloModule" not in head:
          real_llo.append(entry)
      if real_llo:
        real_llo.sort(key=lambda e: os.path.getsize(e[0]), reverse=True)
        best_file = real_llo[0][0]
        print(
          f"LLO: picked non-HLO other candidate: {os.path.basename(best_file)} ({os.path.getsize(best_file)} bytes)",
          file=sys.stderr,
        )
      else:
        # Last resort: use the largest other candidate even if it's HLO
        other_candidates.sort(key=lambda e: os.path.getsize(e[0]), reverse=True)
        best_file = other_candidates[0][0]
        print(
          f"LLO: picked HLO fallback: {os.path.basename(best_file)} ({os.path.getsize(best_file)} bytes)",
          file=sys.stderr,
        )
    print(
      f"LLO: mosaic={len(mosaic_candidates)}, pallas={len(pallas_candidates)}, other={len(other_candidates)}",
      file=sys.stderr,
    )

    # One mosaic dump exists per pallas_call. Group the candidates by the kernel
    # function they contain and keep the largest pass for each, so a multi-call
    # variant ships IR for every kernel instead of only the representative one.
    # The scalar metrics below still describe ONE kernel: they are static
    # instruction-site counts, and two calls with the same body produce the same
    # counts, so they must not be read as a whole-call total.
    llo_per_kernel = {}
    try:
      seen_best = {}
      for entry in mosaic_candidates:
        fpath = entry[0]
        kname = _mosaic_kernel_name(fpath)
        if kname is None:
          continue
        size = os.path.getsize(fpath)
        if kname not in seen_best or size > seen_best[kname][0]:
          seen_best[kname] = (size, fpath)
      llo_per_kernel = {f"llo_{k}.txt": v[1] for k, v in seen_best.items()}
      if len(llo_per_kernel) > 1:
        print(f"LLO: {len(llo_per_kernel)} kernels dumped: {sorted(seen_best)}", file=sys.stderr)
    except Exception as exc:  # never fail the eval over artifact bookkeeping
      print(f"LLO: per-kernel grouping skipped ({exc})", file=sys.stderr)

    if best_file is not None:
      with open(best_file) as fh:
        llo_text = fh.read()

      # Schedule-level tags (.mxu0/.mxu1 per-unit assignment, DMA, double-buffer)
      # only appear in the post-auto-mxu-assigner "late-finalization" LLO pass,
      # NOT in best_file (a mosaic/pre-assigner pass). Reading best_file alone made
      # dual_ratio always 0 (no .mxuN tags -> vmatmul fallback hard-codes mxu1=0).
      # Pick the finalization pass with the most .mxuN tags as the authoritative
      # assigned schedule; keep llo_text for VMEM/bundle-density (parse from mosaic).
      sched_text = llo_text
      try:
        fin = [f for f in glob.glob(os.path.join(dump_dir, "llo", "*.txt")) if "finaliz" in os.path.basename(f).lower()]
        fin.sort(key=os.path.getsize, reverse=True)
        # Prefer a pass that names the candidate kernel the LLO pick bound to, so a
        # larger reference schedule does not stand in for the candidate's. Every
        # candidate-only def would be too loose: a short helper name can hit reference text.
        own_kernels = {llo_kernel} if llo_binding == "candidate" and llo_kernel else set()
        name_res = [re.compile(r"\b" + re.escape(k) + r"\b") for k in own_kernels]
        best_sched, best_key = None, (False, 0)
        for fp in fin[:8]:
          t = open(fp, errors="replace").read()
          n = t.count(".mxu0") + t.count(".mxu1")
          key = (any(r.search(t) for r in name_res), n)
          if n > 0 and key > best_key:
            best_sched, best_key = t, key
        best_tags = best_key[1]
        if best_sched is not None:
          sched_text = best_sched
          print(f"LLO: schedule pass has {best_tags} .mxuN tags", file=sys.stderr)
      except Exception as _e:
        print(f"LLO: schedule-pass scan skipped: {_e}", file=sys.stderr)

      # Log first line and VLIW pattern presence for debugging
      first_line = llo_text.split("\n", 1)[0][:120]
      has_hlomod = "HloModule" in llo_text[:256]
      has_vliw = ";;" in llo_text or ".mxu" in llo_text
      print(
        f"LLO: first_line={first_line!r}, is_hlo={has_hlomod}, has_vliw={has_vliw}, len={len(llo_text)}",
        file=sys.stderr,
      )
      # Count VLIW bundles — multiple formats:
      #   Classic: separated by `;;`
      #   v7x libtpu: numbered entries like `  N  :  { ... }`
      #   v7x alt: `bundle N` markers
      bundle_count = llo_text.count(";;")
      if bundle_count == 0:
        bundle_count = len(re.findall(r"^\s*\d+\s*:\s*\{", llo_text, re.MULTILINE))
      if bundle_count == 0:
        bundle_count = len(re.findall(r"\bbundle\s+\d+", llo_text, re.IGNORECASE))
      if bundle_count == 0:
        # Last resort: count instruction-like lines (register assignments)
        bundle_count = len(
          re.findall(
            r"^\s*%\w+\s*=",
            llo_text,
            re.MULTILINE,
          )
        )
      if bundle_count > 0:
        vliw_bundle_count = bundle_count
      # Count MXU operations per unit from the assigned schedule pass. The
      # `.mxu0`/`.mxu1` tags are the post-auto-mxu-assigner ground truth.
      mxu0_count = len(re.findall(r"\.mxu0\b", sched_text))
      mxu1_count = len(re.findall(r"\.mxu1\b", sched_text))
      # Last resort for LLO formats without per-unit tags: count total matmul
      # ops (no unit split available -> report as mxu0).
      if mxu0_count + mxu1_count == 0:
        vmatmul_count = len(re.findall(r"\bllo\.vmatmul\b", sched_text)) or len(
          re.findall(r"\bllo\.vmatmul\b", llo_text)
        )
        if vmatmul_count > 0:
          mxu0_count = vmatmul_count
          mxu1_count = 0

    total_mxu = mxu0_count + mxu1_count
    if total_mxu > 0:
      max_mxu = max(mxu0_count, mxu1_count)
      min_mxu = min(mxu0_count, mxu1_count)
      mxu_utilization = {
        "mxu0": mxu0_count,
        "mxu1": mxu1_count,
        "total": total_mxu,
        "dual_ratio": min_mxu / max_mxu if max_mxu > 0 else 0.0,
        "distribution": {
          "mxu0_pct": mxu0_count / total_mxu * 100,
          "mxu1_pct": mxu1_count / total_mxu * 100,
        },
      }

    # ── New LLO metrics ─────────────────────────────────────────────
    vmem_allocation = None
    bundle_density = None
    dma_analysis = None
    special_units = None

    if best_file is not None:
      # VMEM allocations — classic format
      alloc_re = re.compile(
        r"#allocation\d+\s*=\s*\w+\[[^\]]*\]"
        r"(?:,\s*space=(\w+))?"
        r",\s*size=0x([0-9a-fA-F]+)"
      )
      vmem_b = 0
      smem_b = 0
      alloc_count = 0
      for am in alloc_re.finditer(llo_text):
        space = am.group(1)
        sz = int(am.group(2), 16)
        if space == "smem":
          smem_b += sz
        else:
          vmem_b += sz
        alloc_count += 1

      # v7x MLIR LLO format: parse memref types from function signature
      # e.g. memref<1x1x128x128xf32, #tpu.tiled<...>, #tpu.memory_space<vmem>>
      # Match each memref with its shape, dtype, and memory space.
      if alloc_count == 0:
        # Find all memref declarations with their memory space
        memref_re = re.compile(
          r"memref<([\dx]+)x(\w+),"  # shape x dtype
          r"(?:(?!memref<).)*?"  # anything except another memref<
          r"memory_space<(\w+)>",  # memory space
          re.DOTALL,
        )
        for mm in memref_re.finditer(llo_text):
          shape_str, dtype, space = mm.group(1), mm.group(2), mm.group(3)
          elems = 1
          for d in shape_str.split("x"):
            if d:
              elems *= int(d)
          byte_sz = elems * _DTYPE_BYTES.get(dtype, 4)
          if space == "smem":
            smem_b += byte_sz
          else:
            vmem_b += byte_sz
          alloc_count += 1

      if alloc_count > 0:
        vmem_allocation = {
          "vmem_bytes": vmem_b,
          "smem_bytes": smem_b,
          "allocation_count": alloc_count,
        }

      # Bundle density (ILP)
      segments = llo_text.split(";;")
      if len(segments) > 1:
        op_re = re.compile(r"^\s*%\w+\s*=", re.MULTILINE)
        ops_list = []
        for seg in segments[:-1]:
          cnt = len(op_re.findall(seg))
          if cnt > 0:
            ops_list.append(cnt)
        if ops_list:
          bundle_density = {
            "total_bundles": len(ops_list),
            "avg_ops_per_bundle": sum(ops_list) / len(ops_list),
            "max_ops_per_bundle": max(ops_list),
          }
      else:
        # v7x MLIR LLO: no ;; separators. Count total llo.* operations
        # and total %var assignments for an ops-per-assignment ratio.
        llo_ops = len(re.findall(r"\bllo\.\w+", llo_text))
        assign_count = vliw_bundle_count or 0
        if assign_count > 0 and llo_ops > 0:
          bundle_density = {
            "total_bundles": assign_count,
            "avg_ops_per_bundle": llo_ops / assign_count,
            "max_ops_per_bundle": 0,  # not available in MLIR format
          }

      # DMA analysis — read the assigned schedule pass (dma.* / double-buffer
      # markers are schedule-level, like .mxuN). Fall back to llo_text.
      _dma_src = sched_text if re.search(r"\bdma\.\w+", sched_text) else llo_text
      dma_cnt = len(re.findall(r"\bdma\.\w+", _dma_src))
      if dma_cnt == 0:
        # v7x MLIR: count llo.vector_load/llo.vector_store as memory ops
        dma_cnt = len(re.findall(r"\bllo\.vector_load\b", _dma_src)) + len(
          re.findall(r"\bllo\.vector_store\b", _dma_src)
        )
      if dma_cnt > 0:
        dma_analysis = {
          "dma_count": dma_cnt,
          "dma_sync_count": len(re.findall(r"\bdma\.done\.wait\b", _dma_src)),
          # NOTE: double-buffering marker is heuristic/unverified for v6e LLO.
          "double_buffering": bool(re.search(r"\bsand\.u32\s+1\b", _dma_src)),
        }

      # Special hardware units
      xlane = len(re.findall(r"\b\w+\.xlane\b", llo_text))
      eup = len(re.findall(r"\bvpow2\b", llo_text)) + len(re.findall(r"\bvpop\.eup\b", llo_text))
      nops = len(re.findall(r"^\s*nop\b", llo_text, re.MULTILINE))
      # v7x MLIR: check for llo.vxpose (cross-lane), llo.vbcast (broadcast)
      if xlane == 0:
        xlane = len(re.findall(r"\bllo\.vxpose\b", llo_text))
      if eup == 0:
        eup = len(re.findall(r"\bllo\.vdwg\b", llo_text))  # exp/pow via vdwg
      if xlane + eup + nops > 0:
        special_units = {
          "xlane_ops": xlane,
          "eup_ops": eup,
          "nop_count": nops,
        }

    # ── Parse HLO dumps (for HBM bandwidth) ────────────────────────
    hbm_bytes = None

    hlo_files = glob.glob(str(hlo_dir / "**" / "*.txt"), recursive=True)
    hlo_files += glob.glob(str(hlo_dir / "**" / "*.hlo"), recursive=True)

    # Prefer files with 'after' in name (optimized HLO). Among those, prefer a
    # module that actually contains the Pallas kernel: picking by size alone
    # returns the REFERENCE module whenever the reference is the bigger program
    # (observed in practice: rounds shipped a reference HLO labelled as the
    # candidate's, which silently voided every HLO-based conclusion for the round).
    # A candidate module always calls tpu_custom_call; a pure-XLA reference never does.
    def _is_candidate_module(path):
      # Three conditions, each earned by a wrong pick: it must be an HLO module
      # (buffer-assignment and liveness dumps also carry "after" in their names and
      # mention tpu_custom_call), it must have an ENTRY computation, and it must
      # contain the Pallas call (a pure-XLA reference module never does).
      try:
        with open(path) as fh:
          text = fh.read()
      except OSError:
        return False
      return "HloModule" in text and "ENTRY" in text and "tpu_custom_call" in text

    def _module_name(path):
      try:
        with open(path, errors="replace") as fh:
          m = re.match(r"\s*HloModule\s+([^\s,]+)", fh.read(512))
      except OSError:
        return None
      return m.group(1) if m else None

    after_files = [f for f in hlo_files if "after" in os.path.basename(f).lower()]
    pool = after_files or hlo_files
    chosen_hlo = None
    hlo_binding = "unverified"
    if pool:
      pool = sorted(pool, key=lambda f: os.path.getsize(f), reverse=True)
      with_call = [f for f in pool if _is_candidate_module(f)]

      # On TPU the reference can also contain a tpu_custom_call (XLA's own Mosaic
      # lowering), so a Pallas call alone does not identify the candidate. Prefer
      # the module jitted from a candidate-only function (e.g. jit_optimized_compute).
      # Exact jitted-function names only: a substring test would let a candidate-only
      # helper such as `compute` claim the reference's jit_simple_compute module.
      def _is_own_module(path):
        m = _module_name(path) or ""
        return any(m == f"jit_{k}" or m.startswith(f"jit_{k}.") for k in (kernel_names or ()))

      own = [f for f in with_call if _is_own_module(f)]
      if own:
        hlo_binding = "candidate"
      chosen_hlo = (own or with_call or pool)[0]
      if with_call:
        print(
          f"HLO: {len(with_call)}/{len(pool)} files are candidate HLO modules; picked {os.path.basename(chosen_hlo)}",
          file=sys.stderr,
        )
      else:
        print(
          "HLO: WARNING no candidate HLO module found (no file with HloModule + "
          "ENTRY + tpu_custom_call); the pick is a fallback and HLO conclusions "
          "are void",
          file=sys.stderr,
        )

    if chosen_hlo is not None:
      print(
        f"HLO: picked {os.path.basename(chosen_hlo)} ({os.path.getsize(chosen_hlo)} bytes) from {len(hlo_files)} files",
        file=sys.stderr,
      )
      with open(chosen_hlo) as fh:
        hlo_text = fh.read()

      # Extract HBM bandwidth from custom_call with tpu_custom_call
      shape_re = re.compile(r"(\w+)\[([\d,]+)\]")
      cc_pat = re.compile(
        r"(%\S+)\s*=\s*(\S+)\s+custom-call\(([^)]+)\)"
        r"[^\"]*custom_call_target=\"tpu_custom_call\""
      )
      cc_match = cc_pat.search(hlo_text)
      if cc_match:
        hbm_bytes = 0
        # Parse output shape
        out_shape = cc_match.group(2)  # e.g., "bf16[8,2048,128]"
        out_m = shape_re.match(out_shape)
        if out_m and out_m.group(1) in _DTYPE_BYTES:
          elems = 1
          for d in out_m.group(2).split(","):
            elems *= int(d)
          hbm_bytes += elems * _DTYPE_BYTES[out_m.group(1)]

        # Parse input operand names and look up their shapes
        input_names = re.findall(r"(%\S+)", cc_match.group(3))
        for name in input_names:
          # Find parameter definition: %name = dtype[dims] parameter(N)
          param_pat = re.compile(re.escape(name) + r"\s*=\s*(\w+\[\d[\d,]*\])")
          param_m = param_pat.search(hlo_text)
          if param_m:
            inp_m = shape_re.match(param_m.group(1))
            if inp_m and inp_m.group(1) in _DTYPE_BYTES:
              elems = 1
              for d in inp_m.group(2).split(","):
                elems *= int(d)
              hbm_bytes += elems * _DTYPE_BYTES[inp_m.group(1)]

    # HLO fusion analysis
    fusion_analysis = None
    if chosen_hlo is not None:
      fc = len(re.findall(r"^fused_computation", hlo_text, re.MULTILINE))
      cpf = bool(re.search(r"cross_program_prefetch_index=", hlo_text))
      fusion_analysis = {
        "fusion_count": fc,
        "has_cross_program_prefetch": cpf,
      }

    arithmetic_intensity = flops / hbm_bytes if flops and hbm_bytes else None

    # VMEM utilization: vmem_bytes vs physical VMEM capacity.
    # Default is TPU v6e (Trillium); override with VMEM_CAPACITY_MIB.
    vmem_utilization_pct = None
    if vmem_allocation is not None and vmem_allocation["vmem_bytes"] > 0:
      vmem_capacity_bytes = int(float(os.environ.get("VMEM_CAPACITY_MIB", 128)) * 1024 * 1024)
      vmem_utilization_pct = vmem_allocation["vmem_bytes"] / vmem_capacity_bytes * 100.0

    return {
      "ok": True,
      "vliw_bundle_count": vliw_bundle_count,
      "mxu_utilization": mxu_utilization,
      "hbm_bandwidth_bytes": hbm_bytes,
      "flops": flops,
      "arithmetic_intensity": arithmetic_intensity,
      "vmem_allocation": vmem_allocation,
      "vmem_utilization_pct": vmem_utilization_pct,
      "bundle_density": bundle_density,
      "dma_analysis": dma_analysis,
      "fusion_analysis": fusion_analysis,
      "special_units": special_units,
      "_hlo_file": chosen_hlo,
      "_llo_file": best_file,
      "llo_kernel": llo_kernel,
      "hlo_module": _module_name(chosen_hlo) if chosen_hlo else None,
      # "candidate" = taken from a kernel/module only the candidate defines;
      # "unverified" = the largest dump, which may be the reference's.
      "artifact_binding": {"llo": llo_binding, "hlo": hlo_binding},
      "_llo_all": llo_per_kernel,
      "llo_kernels": sorted(llo_per_kernel),
      "llo_representative": os.path.basename(best_file) if best_file else None,
      "llo_metrics_scope": ("one kernel of %d" % len(llo_per_kernel))
      if len(llo_per_kernel) > 1
      else "the single kernel",
    }
  except Exception:
    return {"ok": False, "error": f"Deep profile error: {traceback.format_exc()}"}


_DUMP_SUBDIRS = ("hlo", "llo", "mosaic")


def _get_dump_dir() -> str:
  """Return variant-specific dump dir when VARIANT_ID is set, else default.

  A variant's IR dump is ~10 GB; main() removes it once its artifacts are copied
  or uploaded (set EVAL_KEEP_DUMPS=1 to keep it for debugging). Override the base
  with EVAL_DUMP_DIR, e.g. a RAM-backed tmpfs like /dev/shm.
  """
  base = os.environ.get("EVAL_DUMP_DIR", "/tmp/ir_dumps")
  variant_id = os.environ.get("VARIANT_ID", "")
  if variant_id:
    return f"{base}/{variant_id}"
  return base


def _setup_dump_env():
  """Set XLA/LIBTPU dump flags before JAX init so libtpu reads them.

  Must be called before any JAX import or compilation. Dump files are
  generated during all subsequent compilations and parsed later by
  stage_profile_deep().

  When VARIANT_ID env var is set (batch subprocess mode), dumps go to
  a variant-specific subdirectory to avoid cross-variant file conflicts.

  Set EVAL_DISABLE_DUMPS=1 to skip IR dumping entirely (no HLO/Mosaic/LLO on
  disk). Deep profiling then returns ok:false (no files); correctness and
  benchmark latency are unaffected. Use this on small VMs where the per-variant
  IR dump (~15 GB) can fill the disk — a full disk aborts later variants with
  RESOURCE_EXHAUSTED and can hang the primary past its timeout.
  """
  if os.environ.get("EVAL_DISABLE_DUMPS", "") == "1":
    return

  dump_dir = _get_dump_dir()
  # Dumps left by an earlier run with the same variant id would be parsed as this run's.
  _clear_dump_subdirs(dump_dir)
  os.makedirs(f"{dump_dir}/hlo", exist_ok=True)
  os.makedirs(f"{dump_dir}/llo", exist_ok=True)
  os.makedirs(f"{dump_dir}/mosaic", exist_ok=True)

  os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}/hlo " + os.environ.get("XLA_FLAGS", "")
  os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={dump_dir}/llo "
    "--xla_jf_dump_hlo_text=true "
    "--xla_jf_dump_llo_text=true "
    "--xla_jf_emit_annotations=true "
    f"--xla_mosaic_dump_to={dump_dir}/mosaic "
    "--xla_mosaic_enable_llo_source_annotations=true "
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true " + os.environ.get("LIBTPU_INIT_ARGS", "")
  )


def _clear_dump_subdirs(dump_dir: str) -> None:
  """Remove only the dump subdirectories _setup_dump_env() creates under dump_dir."""
  import shutil

  for name in _DUMP_SUBDIRS:
    shutil.rmtree(os.path.join(dump_dir, name), ignore_errors=True)


def _cleanup_dumps() -> None:
  """Delete this run's IR dumps after artifacts are copied or uploaded.

  Each variant dumps ~10 GB of HLO/LLO/Mosaic text, on every host of a slice.
  Without this, a few rounds fill a TPU VM's disk. EVAL_KEEP_DUMPS=1 keeps them.
  """
  if os.environ.get("EVAL_DISABLE_DUMPS") == "1" or os.environ.get("EVAL_KEEP_DUMPS") == "1":
    return
  dump_dir = _get_dump_dir()
  _clear_dump_subdirs(dump_dir)
  if os.environ.get("VARIANT_ID"):
    try:
      os.rmdir(dump_dir)  # the per-variant directory, only if now empty
    except OSError:
      pass


def _keep_trace_artifact() -> bool:
  """Keep XProf JSON for deep-profile runs, or when screening asks explicitly."""
  return os.environ.get("SKIP_DEEP_PROFILE") != "1" or os.environ.get("EVAL_KEEP_TRACE") == "1"


def main():
  _setup_dump_env()
  try:
    _evaluate()
  finally:
    _cleanup_dumps()


def _evaluate():
  global jax
  try:
    import jax as _jax

    jax = _jax
  except ImportError:
    pass

  parser = argparse.ArgumentParser()
  parser.add_argument("--eval-payload", required=False)
  parser.add_argument("--eval-payload-file", required=False, help="Path to file containing base64 payload")
  args = parser.parse_args()
  if args.eval_payload_file:
    with open(args.eval_payload_file) as f:
      b64_payload = f.read().strip()
  elif args.eval_payload:
    b64_payload = args.eval_payload
  else:
    parser.error("--eval-payload or --eval-payload-file is required")
  request = decode_request(b64_payload)
  job_name = request.get("variant_id", "unknown")

  if not _has_tpu():
    print("ERROR: No TPU detected. This evaluator requires a TPU device.", file=sys.stderr)
    sys.exit(1)

  compile_result = stage_compile(request["kernel_code"])
  if not compile_result["ok"]:
    err = {"status": "COMPILE_ERROR", "variant_id": job_name, "error": compile_result["error"]}
    print(f"EVAL_RESULT:{json.dumps(err)}")
    sys.exit(0)

  correct_result = stage_correctness(
    request["kernel_code"],
    request["reference_code"],
    request["shapes"],
    request.get("rtol", 1e-2),
    request.get("atol", 1e-2),
    compile_result["globals"],
    request.get("method", "allclose"),
  )
  if not correct_result["ok"]:
    result_data = {
      "status": "INCORRECT",
      "variant_id": job_name,
      "error": correct_result["error"],
      "max_diff": _json_safe_number(correct_result["max_diff"]),
      "max_scaled_diff": _json_safe_number(correct_result["max_scaled_diff"]),
    }
    print(f"EVAL_RESULT:{json.dumps(result_data)}")
    sys.exit(0)

  # Stage 3+4 merged: Benchmark (performance + profile)
  bench_result = stage_benchmark(
    compile_result["globals"],
    request["shapes"],
    trace_dir="/tmp/xplane_trace/kernel",
  )
  if not bench_result["ok"]:
    err = {"status": "COMPILE_ERROR", "variant_id": job_name, "error": bench_result["error"]}
    print(f"EVAL_RESULT:{json.dumps(err)}")
    sys.exit(0)

  ref_compile = stage_compile(request["reference_code"])
  ref_bench = stage_benchmark(
    ref_compile.get("globals", {}),
    request["shapes"],
    trace_dir="/tmp/xplane_trace/reference",
  )
  ref_latency = ref_bench.get("latency_ms", bench_result["latency_ms"])
  speedup = ref_latency / bench_result["latency_ms"] if bench_result["latency_ms"] > 0 else 0.0

  compute_ratio = bench_result.get("compute_ratio")
  memory_transfer_ratio = bench_result.get("memory_transfer_ratio")
  hw_utilization = bench_result.get("hw_utilization")
  profile_diag = bench_result.get("diagnostics", {})

  # Stage 5: Deep profile (non-fatal) — parse dumps generated by earlier stages
  if os.environ.get("SKIP_DEEP_PROFILE") == "1":
    # Screening mode: latency + correctness decide top-K; the LLO/HLO/trace dumps
    # (tens of GB and several minutes per variant) are only worth it for finalists.
    deep_profile = {"ok": False, "skipped": True}
  else:
    deep_profile = stage_profile_deep(
      compile_result["globals"],
      request["shapes"],
      kernel_names=_candidate_function_names(request["kernel_code"], request["reference_code"]),
    )
  if not deep_profile["ok"]:
    print(
      f"Deep profile skipped: {deep_profile.get('error', 'unknown')}",
      file=sys.stderr,
    )

  # Compute efficiency from deep profile FLOPs and measured latency
  # TPU v6e (Trillium) per-chip peak: ~918 TFLOPS bf16. Override with PEAK_FLOPS.
  peak_flops = float(os.environ.get("PEAK_FLOPS", 918e12))
  if deep_profile.get("flops") and bench_result["latency_ms"] > 0:
    actual_fps = deep_profile["flops"] / (bench_result["latency_ms"] / 1000.0)
    deep_profile["compute_efficiency_pct"] = (actual_fps / peak_flops) * 100.0

  # HBM bandwidth utilization (TPU v6e peak: ~1759 GB/s). Override with PEAK_HBM_BW.
  peak_hbm_bw = float(os.environ.get("PEAK_HBM_BW", 1759e9))
  if deep_profile.get("hbm_bandwidth_bytes") and bench_result["latency_ms"] > 0:
    actual_bw = deep_profile["hbm_bandwidth_bytes"] / (bench_result["latency_ms"] / 1000.0)
    deep_profile["hbm_bandwidth_utilization_pct"] = (actual_bw / peak_hbm_bw) * 100.0

  # HBM capacity utilization (TPU v6e: 32 GB per chip). Override with HBM_CAPACITY_GB.
  peak_memory_mb = bench_result.get("benchmark", {}).get("peak_memory_mb")
  if peak_memory_mb is not None and peak_memory_mb > 0:
    hbm_capacity_mb = float(os.environ.get("HBM_CAPACITY_GB", 32)) * 1024  # GB -> MB
    deep_profile["hbm_capacity_utilization_pct"] = peak_memory_mb / hbm_capacity_mb * 100.0

  # ── Upload profile artifacts to GCS (non-fatal) ──
  artifacts = {}
  if bench_result.get("ok") and _keep_trace_artifact():
    trace_path = bench_result.get("_trace_events_path")
    if trace_path:
      artifacts["trace_events.json"] = trace_path
  if deep_profile.get("ok"):
    hlo_path = deep_profile.get("_hlo_file")
    llo_path = deep_profile.get("_llo_file")
    if hlo_path:
      artifacts["hlo_post_opt.txt"] = hlo_path
    if llo_path:
      artifacts["llo_final.txt"] = llo_path
    # A variant with more than one pallas_call has one mosaic dump PER kernel, and
    # the representative pick above returns exactly one of them -- for a two-call
    # kernel (e.g. compute + RDMA) it can pick the half without the RDMA, leaving
    # the interesting half with no IR at all. Ship every distinct kernel's dump and
    # let the reader see which is which.
    for extra_name, extra_path in (deep_profile.get("_llo_all") or {}).items():
      if extra_path != llo_path:
        artifacts[extra_name] = extra_path

  # ── Copy artifacts to a local dir for non-GCS transports (e.g. SSH). ──
  # When ARTIFACTS_LOCAL_DIR is set, each artifact is copied to
  # {ARTIFACTS_LOCAL_DIR}/{variant_id}/ so an SSH-based evaluator can scp
  # them back without any GCS bucket.
  artifacts_local_dir_env = os.environ.get("ARTIFACTS_LOCAL_DIR")
  copied_local = []
  local_variant_dir = ""
  if artifacts_local_dir_env and artifacts:
    import shutil

    local_variant_dir = os.path.join(artifacts_local_dir_env, job_name)
    os.makedirs(local_variant_dir, exist_ok=True)
    for name, local_path in artifacts.items():
      try:
        if local_path and os.path.exists(local_path):
          shutil.copy(local_path, os.path.join(local_variant_dir, name))
          copied_local.append(name)
      except Exception as e:
        print(f"Local artifact copy failed for {name}: {e}", file=sys.stderr)
    if copied_local:
      print(f"Copied artifacts {copied_local} to {local_variant_dir}", file=sys.stderr)

  # No default bucket: unset GCS_BUCKET skips the upload (SSH transport uses scp).
  _gcs_bucket = os.environ.get("GCS_BUCKET", "")
  gcs_result = (
    upload_to_gcs(job_name, artifacts, bucket_name=_gcs_bucket)
    if artifacts
    else {"ok": False, "uploaded": [], "gcs_prefix": ""}
  )
  if gcs_result["ok"]:
    print(f"Uploaded artifacts: {gcs_result['uploaded']} to {gcs_result['gcs_prefix']}", file=sys.stderr)

  # Strip internal fields from deep_profile before including in result
  clean_deep_profile = {k: v for k, v in deep_profile.items() if not k.startswith("_")}

  result = {
    "status": "SUCCESS",
    "variant_id": job_name,
    "fitness": speedup,
    "latency_ms": bench_result["latency_ms"],
    "speedup": speedup,
    "max_diff": _json_safe_number(correct_result["max_diff"]),
    "max_scaled_diff": _json_safe_number(correct_result["max_scaled_diff"]),
    "flops": clean_deep_profile.get("flops", 0.0) or 0.0,
    "compute_ratio": compute_ratio,
    "memory_transfer_ratio": memory_transfer_ratio,
    "metadata": {
      "reference_latency_ms": ref_latency,
      "reference_perf_ok": ref_bench.get("ok", False),
      "profile_diagnostics": profile_diag,
      "hw_utilization": hw_utilization,
      "profile": clean_deep_profile,
      "benchmark": bench_result.get("benchmark"),
      "reference_benchmark": ref_bench.get("benchmark"),
      **({"artifacts_gcs_prefix": gcs_result["gcs_prefix"]} if gcs_result.get("ok") else {}),
      **({"artifacts_local_dir": local_variant_dir} if copied_local else {}),
    },
  }
  print(f"EVAL_RESULT:{json.dumps(result)}")


if __name__ == "__main__":
  # Parse payload to check for batch mode BEFORE setting up dump env.
  # In batch mode, subprocesses each set their own dump env.
  _parser = argparse.ArgumentParser()
  _parser.add_argument("--eval-payload", required=False)
  _parser.add_argument("--eval-payload-file", required=False, help="Path to file containing base64 payload")
  _args = _parser.parse_args()
  if _args.eval_payload_file:
    with open(_args.eval_payload_file) as f:
      _b64 = f.read().strip()
  elif _args.eval_payload:
    _b64 = _args.eval_payload
  else:
    _parser.error("--eval-payload or --eval-payload-file is required")
  _request = decode_request(_b64)

  if _request.get("batch"):
    # Batch mode: skip TPU check in parent — each subprocess checks
    # independently. Calling _has_tpu() here locks libtpu for PID 1,
    # preventing subprocesses from accessing the TPU.
    from kernel_evolve.docker_evaluate_helpers import batch_dispatch

    result_lines = batch_dispatch(
      _request,
      evaluator_script=__file__,
      per_variant_timeout=1500,
    )
    for line in result_lines:
      print(line)
  else:
    # Single-variant mode: original path
    main()
