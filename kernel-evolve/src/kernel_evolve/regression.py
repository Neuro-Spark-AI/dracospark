"""Compare paired caller measurements and trace evidence, without inventing causality.

Run with ``python -m kernel_evolve.regression control.json candidate.json
--noise-pct 1 --out regression.json``. See docs/attention-optimization-policy.md.
Only the standard library is needed; this tool never launches a hardware job.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

EXPERIMENTS = {
  "custom_call": "Keep tensor geometry and mask fixed; compare per-range custom calls with one grouped call. "
  "Check queue materializations and intervening ATen work in both traces.",
  "wait": "Correlate waits with host dispatch, queue drains and collectives. Change one boundary at a time; "
  "SyncWait alone does not identify HBM traffic.",
  "layout_copy": "Compare caller HLO and tensor layouts; isolate one copy/transpose or padding conversion "
  "without changing attention visibility.",
  "collective": "Hold rank mapping, payload, parallel degrees and concurrency fixed; test placement/overlap "
  "separately from the compute kernel.",
  "kernel": "Replay the exact local Q/K/V shapes and mask; compare HLO/LLO, spills and tile choices. "
  "Retune one parameter and rerun in the caller.",
  "compile": "Separate cold start from steady state; record specialization/cache misses and repeat with "
  "the same warmed shapes before diagnosing execution latency.",
}


def _finite(value):
  return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _union_us(intervals):
  total = 0.0
  end = -math.inf
  for start, stop in sorted(intervals):
    total += max(0.0, stop - max(start, end))
    end = max(end, stop)
  return total


def summarize_trace(events, spec):
  """Union overlapping spans per category inside an explicit window (microseconds).

  Counts are matched trace events, NOT inferred kernel launches. Categories may
  overlap each other. Missing events and unclassified time are not device idle.
  Explicit pid/tid/name selectors avoid confusing host scopes with device work.
  """
  start, stop = spec["window_us"]
  repetitions = spec["repetitions"]
  if not (_finite(start) and _finite(stop) and stop > start):
    raise ValueError("trace window_us must be two finite increasing timestamps")
  if type(repetitions) is not int or repetitions <= 0:
    raise ValueError("trace repetitions must be a positive integer")
  if not isinstance(events, list) or not spec.get("selectors"):
    raise ValueError("trace must contain an event list and explicit selectors")
  if any(not isinstance(event, dict) for event in events):
    raise ValueError("each trace event must be an object")
  categories = {}
  for category, selector in spec["selectors"].items():
    if "pid" not in selector or not selector.get("name_regex"):
      raise ValueError("each trace selector requires pid and name_regex")
    pattern = re.compile(selector["name_regex"])
    intervals = []
    clipped = False
    for event in events:
      if (
        event.get("ph") != "X"
        or event.get("pid") != selector["pid"]
        or ("tid" in selector and event.get("tid") != selector["tid"])
        or not pattern.search(str(event.get("name", "")))
      ):
        continue
      ts, dur = event.get("ts"), event.get("dur")
      if not (_finite(ts) and _finite(dur) and dur > 0 and _finite(ts + dur)):
        continue
      left, right = max(start, ts), min(stop, ts + dur)
      if right > left:
        intervals.append((left, right))
        clipped |= left != ts or right != ts + dur
    expected = selector.get("expected_events_per_repetition")
    if expected is not None:
      if type(expected) is not int or expected < 0:
        raise ValueError("expected_events_per_repetition must be a nonnegative integer")
      if len(intervals) != expected * repetitions or clipped:
        raise ValueError(
          f"{category}: incomplete declared event coverage: matched {len(intervals)}, "
          f"expected {expected * repetitions}, clipped={clipped}; verify window and clocks"
        )
    categories[category] = {
      "matched_events": len(intervals),
      "events_per_repetition": len(intervals) / repetitions,
      "union_ms_per_repetition": _union_us(intervals) / repetitions / 1000,
    }
  return {"window_ms_per_repetition": (stop - start) / repetitions / 1000, "categories": categories}


def _measurement_issues(run, label):
  issues = []
  context = run.get("context", {})
  for field in ("workload", "hardware", "runtime", "timing"):
    if not isinstance(context.get(field), dict) or not context[field]:
      issues.append(f"{label}: missing context.{field}")
  workload = context.get("workload", {})
  for field in ("shapes", "dtypes", "layout", "semantics", "input_id", "concurrency"):
    if field not in workload or workload[field] is None:
      issues.append(f"{label}: missing workload.{field}")
  timing = context.get("timing", {})
  for field in ("scope", "clock"):
    if not timing.get(field):
      issues.append(f"{label}: missing timing.{field}")
  for field, expected in (("synchronized", True), ("steady_state", True), ("profiled", False)):
    if timing.get(field) is not expected:
      issues.append(f"{label}: timing.{field} must be {expected}")
  samples = run.get("samples_ms")
  if not isinstance(samples, list) or len(samples) < 3 or any(not _finite(x) or x <= 0 for x in samples):
    issues.append(f"{label}: require at least three finite positive unprofiled latency samples_ms")
  if not run.get("revision"):
    issues.append(f"{label}: missing revision identifying the implementation")
  return issues


def _trace(run, base_dir):
  spec = run.get("trace")
  if spec is None:
    return None, "missing trace; cannot localize the timing difference"
  path = Path(spec["path"])
  if not path.is_absolute():
    path = base_dir / path
  data = json.loads(path.read_text())
  events = data.get("traceEvents") if isinstance(data, dict) else data
  return summarize_trace(events, spec), None


def compare_runs(control, candidate, *, noise_pct, control_dir=Path("."), candidate_dir=Path(".")):
  """Report observations and falsifiable hypotheses, never an automatic promotion.

  noise_pct is a caller-supplied practical threshold, not a statistical test.
  A complete record and trace still need an ablation to establish root cause.
  """
  if not _finite(noise_pct) or noise_pct < 0:
    raise ValueError("noise_pct must be finite and nonnegative")
  issues = _measurement_issues(control, "control") + _measurement_issues(candidate, "candidate")
  if control.get("context") != candidate.get("context"):
    issues.append("contexts differ: compare the same workload, hardware, runtime and timing protocol")
  report = {
    "comparison": "inconclusive" if issues else "comparable",
    "issues": issues,
    "noise_tolerance_pct": noise_pct,
    "root_cause": "unproven",
    "promotion": "not_evaluated",
    "correctness_passed": control.get("correctness_passed") is True and candidate.get("correctness_passed") is True,
    "trace_deltas": {},
    "hypotheses": [],
    "missing_evidence": [],
    "limitations": [
      "Profile categories can overlap; their union times must not be summed into a latency budget.",
      "Matched trace-event counts are not proof of runtime launch counts or queue materialization.",
      "Medians and the supplied noise threshold do not establish statistical significance or causality.",
    ],
  }
  if issues:
    return report
  before, after = statistics.median(control["samples_ms"]), statistics.median(candidate["samples_ms"])
  delta_pct = 100 * (after / before - 1)
  report["latency"] = {
    "control_median_ms": before,
    "candidate_median_ms": after,
    "delta_ms": after - before,
    "delta_pct": delta_pct,
    "control_samples": len(control["samples_ms"]),
    "candidate_samples": len(candidate["samples_ms"]),
  }
  report["observation"] = (
    "regression" if delta_pct > noise_pct else "improvement" if delta_pct < -noise_pct else "within_threshold"
  )
  if not report["correctness_passed"]:
    report["missing_evidence"].append("correctness is not confirmed for both arms; do not promote")
  traces = []
  for label, run, directory in (("control", control, control_dir), ("candidate", candidate, candidate_dir)):
    try:
      trace, error = _trace(run, directory)
      if error:
        report["missing_evidence"].append(f"{label}: {error}")
      traces.append(trace)
    except (OSError, ValueError, KeyError, TypeError, re.error) as exc:
      traces.append(None)
      report["missing_evidence"].append(f"{label}: invalid trace: {exc}")
  if any(trace is None for trace in traces):
    return report
  for category in sorted(set(traces[0]["categories"]) | set(traces[1]["categories"])):
    if any(category not in trace["categories"] for trace in traces):
      report["missing_evidence"].append(f"{category}: selector missing from one trace")
      continue
    selectors = [run["trace"]["selectors"][category] for run in (control, candidate)]
    if selectors[0]["name_regex"] != selectors[1]["name_regex"] or ("tid" in selectors[0]) != ("tid" in selectors[1]):
      report["missing_evidence"].append(f"{category}: different event-name selectors or thread scope; compare manually")
      continue
    left, right = (trace["categories"][category] for trace in traces)
    change = right["union_ms_per_repetition"] - left["union_ms_per_repetition"]
    count_change = right["events_per_repetition"] - left["events_per_repetition"]
    report["trace_deltas"][category] = {
      "control": left,
      "candidate": right,
      "delta_union_ms_per_repetition": change,
      "delta_events_per_repetition": count_change,
    }
    if not left["matched_events"] or not right["matched_events"]:
      report["missing_evidence"].append(f"{category}: zero matches in an arm; verify selector and trace coverage")
    # More short waits need not mean more waiting. Call/compile counts can
    # motivate boundary/cache experiments, but other categories need a time increase.
    increased = change > 0 or (count_change > 0 and category in ("custom_call", "compile"))
    if report["observation"] == "regression" and increased and category in EXPERIMENTS:
      report["hypotheses"].append(
        {"category": category, "confidence": "hypothesis_only", "next_experiment": EXPERIMENTS[category]}
      )
  if report["observation"] == "regression" and not report["hypotheses"]:
    report["missing_evidence"].append(
      "no increased selected category explains the regression; inspect "
      "caller boundaries, trace coverage and critical-path overlap"
    )
  return report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("control", type=Path)
  parser.add_argument("candidate", type=Path)
  parser.add_argument("--noise-pct", type=float, required=True)
  parser.add_argument("--out", type=Path)
  args = parser.parse_args(argv)
  try:
    report = compare_runs(
      json.loads(args.control.read_text()),
      json.loads(args.candidate.read_text()),
      noise_pct=args.noise_pct,
      control_dir=args.control.parent,
      candidate_dir=args.candidate.parent,
    )
  except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
    parser.error(str(exc))
  rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
  if args.out:
    args.out.write_text(rendered)
  else:
    print(rendered, end="")
  return 2 if report["comparison"] == "inconclusive" else 0


if __name__ == "__main__":
  raise SystemExit(main())
