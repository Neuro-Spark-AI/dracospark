"""Capture a controlled Pallas tile experiment, not a model performance claim.

Launch on every worker of an idle TPU slice with the SAME --out and arguments.
Only process zero benchmarks one local chip; other ranks join the final barrier.
Run fresh processes repeatedly. The fixture changes the matmul/attention row tile from
128 to 16, then restores it. No sleep, fake timings, or altered output semantics
are used to manufacture a slowdown. A tile change is not guaranteed to regress.

Raw unprofiled sidecars are compatible with kernel_evolve.regression. Trace
captures are separate, with selectors verified for these particular fixtures.
Other workloads need their own inspected selectors and coverage contract. Ground-truth
interventions live in experiment.json, not in the diagnostic input sidecars.
Requires JAX TPU, NumPy and XProf. Does not install anything or launch SSH jobs.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import time
from collections import Counter
from pathlib import Path


def write_json(path, data):
  path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def paired_samples(left, right, *, synchronize, samples, clock=time.perf_counter_ns):
  """Alternate AB/BA, synchronizing the complete output tree each time."""
  results = [[], []]
  order = []
  for index in range(samples):
    pair = (0, 1) if index % 2 == 0 else (1, 0)
    for arm in pair:
      start = clock()
      synchronize((left, right)[arm]())
      results[arm].append((clock() - start) / 1e6)
      order.append(arm)
  return results, order


def fixture_trace_spec(events, repetitions):
  """Verify one named device operation per fixture invocation, on the device clock.

  This is NOT a general iteration finder. Its exact event-count contract is
  specific to the two one-Pallas-call fixtures below. A host annotation may
  use a shifted clock and must not silently clip device events.
  """
  devices = [
    e["pid"]
    for e in events
    if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name") == "/device:TPU:0"
  ]
  if len(devices) != 1:
    raise ValueError("missing or ambiguous fixture TPU process")
  pid = devices[0]
  threads = {
    e["args"]["name"]: e["tid"]
    for e in events
    if e.get("ph") == "M" and e.get("name") == "thread_name" and e["pid"] == pid
  }
  if "XLA Ops" not in threads or "Tensor Core Sync Flag" not in threads:
    raise ValueError("fixture trace lacks required device tracks")
  pattern = r"^diagnostic_(matmul|attention)\."
  kernels = [
    e
    for e in events
    if e.get("ph") == "X"
    and e.get("pid") == pid
    and e.get("tid") == threads["XLA Ops"]
    and re.search(pattern, e.get("name", ""))
  ]
  if len(kernels) != repetitions:
    raise ValueError(f"expected {repetitions} complete fixture operations, got {len(kernels)}")
  return {
    "window_us": [min(e["ts"] for e in kernels), max(e["ts"] + e["dur"] for e in kernels)],
    "window_clock": "XProf device timestamps; exact fixture operation coverage",
    "repetitions": repetitions,
    "selectors": {
      "kernel": {"pid": pid, "tid": threads["XLA Ops"], "name_regex": pattern, "expected_events_per_repetition": 1},
      "wait": {"pid": pid, "tid": threads["Tensor Core Sync Flag"], "name_regex": r"^SyncWait:"},
    },
  }


def capture(fn, directory, repetitions):
  import jax
  from xprof.convert import raw_to_tool_data

  directory.mkdir()
  options = jax.profiler.ProfileOptions()
  options.python_tracer_level = 0
  options.host_tracer_level = 2
  options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}
  jax.profiler.start_trace(str(directory), profiler_options=options)
  try:
    with jax.profiler.TraceAnnotation("dracospark_capture"):
      for _ in range(repetitions):
        jax.block_until_ready(fn())
  finally:
    jax.profiler.stop_trace()
  xplanes = sorted(directory.rglob("*.xplane.pb"))
  if len(xplanes) != 1:
    raise RuntimeError(f"expected one local XPlane, got {xplanes}")
  data, _ = raw_to_tool_data.xspace_to_tool_data([str(xplanes[0])], "trace_viewer", {})
  events = json.loads(data)["traceEvents"]
  write_json(directory / "events.json", events)
  windows = [e for e in events if e.get("ph") == "X" and e.get("name") == "dracospark_capture"]
  if len(windows) != 1:
    raise RuntimeError("missing or ambiguous explicit capture window")
  window = windows[0]
  inventory = Counter((e.get("pid"), e.get("tid"), e.get("name")) for e in events if e.get("ph") == "X")
  write_json(
    directory / "inventory.json",
    {
      "metadata": [e for e in events if e.get("ph") == "M"],
      "events": [
        {"pid": pid, "tid": tid, "name": name, "count": count} for (pid, tid, name), count in inventory.most_common()
      ],
    },
  )
  return {
    **fixture_trace_spec(events, repetitions),
    "path": str(directory.name + "/events.json"),
    "host_annotation_window_us": [window["ts"], window["ts"] + window["dur"]],
  }


def benchmark(args):
  import jax
  import jax.numpy as jnp
  import ml_dtypes
  import numpy as np
  from jax.experimental import pallas as pl

  outdir = args.out
  outdir.mkdir(parents=True, exist_ok=False)
  source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  rng = np.random.default_rng(20260920)
  size = args.size
  with jax.default_device(jax.local_devices()[0]):
    attention = args.workload == "attention"
    shape = (args.heads, size, args.dim) if attention else (size, size)
    inputs = [
      jax.device_put(rng.normal(0, 0.125, shape).astype(np.float32)).astype(jnp.bfloat16)
      for _ in range(3 if attention else 2)
    ]
    jax.block_until_ready(inputs)

    def build(block_m):
      def kernel(x_ref, y_ref, out_ref):
        out_ref[...] = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

      def compute(a, b):
        return pl.pallas_call(
          kernel,
          name="diagnostic_matmul",
          out_shape=jax.ShapeDtypeStruct((size, size), jnp.float32),
          grid=(size // block_m, size // 128),
          in_specs=[pl.BlockSpec((block_m, size), lambda i, j: (i, 0)), pl.BlockSpec((size, 128), lambda i, j: (0, j))],
          out_specs=pl.BlockSpec((block_m, 128), lambda i, j: (i, j)),
        )(a, b)

      def attention_kernel(q_ref, k_ref, v_ref, out_ref):
        scores = jnp.dot(q_ref[...], k_ref[...].T, preferred_element_type=jnp.float32) * (args.dim**-0.5)
        probs = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
        out_ref[...] = jnp.dot(probs.astype(jnp.bfloat16), v_ref[...], preferred_element_type=jnp.float32)

      def attention_compute(q, k, v):
        return pl.pallas_call(
          attention_kernel,
          name="diagnostic_attention",
          out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
          grid=(args.heads, size // block_m),
          in_specs=[
            pl.BlockSpec((None, block_m, args.dim), lambda h, i: (h, i, 0)),
            pl.BlockSpec((None, size, args.dim), lambda h, i: (h, 0, 0)),
            pl.BlockSpec((None, size, args.dim), lambda h, i: (h, 0, 0)),
          ],
          out_specs=pl.BlockSpec((None, block_m, args.dim), lambda h, i: (h, i, 0)),
        )(q, k, v)

      start = time.perf_counter()
      compiled = jax.jit(attention_compute if attention else compute).lower(*inputs).compile()
      compile_seconds = time.perf_counter() - start
      (outdir / f"tile-{block_m}.hlo.txt").write_text(compiled.as_text())
      return lambda: compiled(*inputs), compile_seconds

    functions_and_compile = [build(128), build(16)]
    functions = [f for f, _ in functions_and_compile]
    # Same BF16 operands with FP64 host accumulation, outside every timed region.
    host_inputs = [np.asarray(x, dtype=np.float64) for x in inputs]
    if attention:
      q, k, v = host_inputs
      scores = q @ k.swapaxes(-1, -2) * (args.dim**-0.5)
      probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
      probs /= probs.sum(axis=-1, keepdims=True)
      expected = probs.astype(ml_dtypes.bfloat16).astype(np.float64) @ v
      reference = "NumPy FP64 dense attention, BF16 probability rounding before PV"
    else:
      expected = host_inputs[0] @ host_inputs[1]
      reference = "NumPy FP64 matmul of the same BF16 operands"
    correctness = []
    for fn in functions:
      actual = np.asarray(jax.block_until_ready(fn()))
      np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
      correctness.append(
        {"max_abs_error": float(np.max(np.abs(actual - expected))), "reference": reference, "atol": 2e-4, "rtol": 2e-4}
      )
    # Fixed warmup after explicit compilation; retain samples to inspect stability.
    warmup = []
    for fn in functions:
      times, _ = paired_samples(fn, fn, synchronize=jax.block_until_ready, samples=10)
      warmup.append(times)

    context = {
      "workload": {
        "shapes": {name: list(shape) for name in (("q", "k", "v") if attention else ("x", "y"))},
        "dtypes": {"input": "bf16", "accumulation": "f32", "output": "f32"},
        "layout": "HSD" if attention else "row-major matrices",
        "semantics": (
          {
            "attention": "dense noncausal MHA",
            "padding": False,
            "scale": args.dim**-0.5,
            "probabilities": "BF16 before PV",
            "rope": "none",
            "dropout": False,
            "bias": False,
          }
          if attention
          else "matmul, no transpose or bias"
        ),
        "input_id": "numpy.default_rng seed=20260920; normal(0,0.125); cast BF16",
        "concurrency": 1,
      },
      "hardware": {
        "device": jax.local_devices()[0].device_kind,
        "global_chips": jax.device_count(),
        "processes": jax.process_count(),
        "active_benchmark_chips": 1,
        "device_id": jax.local_devices()[0].id,
      },
      "runtime": {
        "python": platform.python_version(),
        **{p: metadata.version(p) for p in ("jax", "jaxlib", "libtpu", "numpy", "xprof")},
      },
      "timing": {
        "scope": "single-chip Pallas caller: dispatch + complete-output synchronization",
        "clock": "time.perf_counter_ns",
        "synchronized": True,
        "steady_state": True,
        "profiled": False,
        "order": "alternating AB/BA",
      },
    }
    # The analyzer sees only opaque IDs, measurements and inspected trace selectors.
    experiments = [("aa", 0, 0), ("ab", 0, 1), ("ba", 1, 0)]
    for name, left, right in experiments:
      samples, order = paired_samples(
        functions[left], functions[right], synchronize=jax.block_until_ready, samples=args.samples
      )
      for arm, fn_index in enumerate((left, right)):
        record = {
          "revision": f"sha256:{source_hash}:implementation-{fn_index}",
          "context": context,
          "samples_ms": samples[arm],
          "correctness_passed": True,
          "correctness": correctness[fn_index],
        }
        write_json(outdir / f"{name}-{arm}.json", record)
      write_json(outdir / f"{name}-order.json", order)
    # Only now enable profiling. These durations never enter samples_ms.
    captures = [capture(fn, outdir / f"trace-{i}", 5) for i, fn in enumerate(functions)]
    write_json(outdir / "captures.json", captures)
    for name, left, right in experiments:
      for arm, index in enumerate((left, right)):
        record = json.loads((outdir / f"{name}-{arm}.json").read_text())
        record["trace"] = captures[index]
        write_json(outdir / f"{name}-{arm}-with-trace.json", record)
    write_json(
      outdir / "experiment.json",
      {
        "fixture": "controlled row-tile intervention; NOT a production model regression reproduction",
        "source_sha256": source_hash,
        "tiles": [128, 16],
        "pairs": experiments,
        "compile_seconds": [t for _, t in functions_and_compile],
        "warmup_samples_ms": warmup,
        "correctness": correctness,
        "noise_threshold_pct": 5,
        "capture_separate_from_timing": True,
      },
    )
    print(f"Captured measurements in {outdir}", flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--size", type=int, default=1024)
  parser.add_argument("--samples", type=int, default=60)
  parser.add_argument("--workload", choices=("matmul", "attention"), default="matmul")
  parser.add_argument("--heads", type=int, default=5)
  parser.add_argument("--dim", type=int, choices=(64, 128), default=128)
  args = parser.parse_args()
  if args.size <= 0 or args.size % 128 or args.samples < 30:
    parser.error("size must be a positive multiple of 128 and samples >= 30")
  if args.heads <= 0:
    parser.error("heads must be positive")
  import jax
  from jax.experimental import multihost_utils

  jax.distributed.initialize(initialization_timeout=120)
  if not all(d.platform == "tpu" for d in jax.devices()):
    raise RuntimeError("TPU required; no CPU fallback")
  if jax.process_index() == 0:
    benchmark(args)
  multihost_utils.sync_global_devices("dracospark-validation-finished")
  jax.distributed.shutdown()


if __name__ == "__main__":
  main()
