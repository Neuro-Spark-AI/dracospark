"""Synthetic evidence tests; none of these fixtures is a TPU performance result."""

import copy
import json

import pytest

from kernel_evolve.regression import compare_runs, main, summarize_trace


def run(samples=(10, 10, 10)):
  return {
    "revision": "synthetic-control",
    "context": {
      "workload": {
        "shapes": {"q": [5, 8192, 128]},
        "dtypes": "bf16",
        "layout": "HSD",
        "semantics": "dense",
        "input_id": "fixture",
        "concurrency": 1,
      },
      "hardware": {"device": "synthetic"},
      "runtime": {"compiler": "fixture"},
      "timing": {
        "scope": "caller",
        "clock": "wallclock",
        "synchronized": True,
        "steady_state": True,
        "profiled": False,
      },
    },
    "samples_ms": list(samples),
    "correctness_passed": True,
  }


def event(name, start, dur, pid=1, tid=1):
  return dict(name=name, ts=start, dur=dur, pid=pid, tid=tid, ph="X")


def spec(**changes):
  return {
    "window_us": [0, 10000],
    "repetitions": 1,
    "selectors": {"custom_call": {"pid": 1, "tid": 1, "name_regex": "^custom$"}},
  } | changes


def test_overlaps_are_unioned_and_clipped_not_summed():
  summary = summarize_trace(
    [event("custom", -100, 1100), event("custom", 500, 1000), event("custom", 9900, 200)], spec()
  )
  category = summary["categories"]["custom_call"]
  assert category["matched_events"] == 3
  assert category["union_ms_per_repetition"] == pytest.approx(1.6)


def test_filters_other_devices_threads_counters_and_bad_timestamps():
  events = [
    event("custom", 0, 100),
    event("custom", 0, 5000, pid=2),
    event("custom", 0, 4000, tid=2),
    event("other", 0, 5000),
    event("custom", 0, float("nan")),
    event("custom", 0, -1),
    event("custom", 0, 9000) | {"ph": "C"},
  ]
  category = summarize_trace(events, spec())["categories"]["custom_call"]
  assert category["matched_events"] == 1
  assert category["union_ms_per_repetition"] == pytest.approx(0.1)


def test_counts_normalized_by_repetitions_without_guessing_iterations():
  events = [event("custom", 0, 1000), event("custom", 2000, 1000)]
  category = summarize_trace(events, spec(repetitions=2))["categories"]["custom_call"]
  assert category["events_per_repetition"] == 1
  assert category["union_ms_per_repetition"] == 1


@pytest.mark.parametrize("events", [[], [event("custom", -100, 1000)], [event("custom", 9000, 2000)]])
def test_declared_event_coverage_rejects_missing_or_clipped_spans(events):
  selection = {"custom_call": {"pid": 1, "tid": 1, "name_regex": "^custom$", "expected_events_per_repetition": 1}}
  with pytest.raises(ValueError, match="incomplete declared event coverage"):
    summarize_trace(events, spec(selectors=selection))


def test_declared_event_coverage_accepts_exact_complete_spans():
  selection = {"kernel": {"pid": 1, "tid": 1, "name_regex": "^custom$", "expected_events_per_repetition": 1}}
  result = summarize_trace(
    [event("custom", 0, 1000), event("custom", 2000, 1000)], spec(selectors=selection, repetitions=2)
  )
  assert result["categories"]["kernel"]["matched_events"] == 2


@pytest.mark.parametrize("expected", [-1, True, 1.5])
def test_bad_declared_event_coverage_rejected(expected):
  selection = {"kernel": {"pid": 1, "name_regex": "^custom$", "expected_events_per_repetition": expected}}
  with pytest.raises(ValueError, match="nonnegative integer"):
    summarize_trace([], spec(selectors=selection))


@pytest.mark.parametrize(
  "changes",
  [
    {"window_us": [0, 0]},
    {"window_us": [0, float("inf")]},
    {"repetitions": 0},
    {"repetitions": True},
    {"selectors": {"kernel": {"name_regex": "custom"}}},
  ],
)
def test_bad_trace_contract_rejected(changes):
  with pytest.raises(ValueError):
    summarize_trace([], spec(**changes))


@pytest.mark.parametrize("samples", [(12, 12, 12), (8, 8, 8), (10, 10, 10)])
def test_latency_observation_does_not_claim_root_cause_or_promote(samples):
  report = compare_runs(run(), run(samples), noise_pct=1)
  expected = {12: "regression", 8: "improvement", 10: "within_threshold"}[samples[0]]
  assert report["observation"] == expected
  assert report["root_cause"] == "unproven"
  assert report["promotion"] == "not_evaluated"
  assert report["missing_evidence"]


@pytest.mark.parametrize("field", ["workload", "hardware", "runtime", "timing"])
def test_unmatched_context_does_not_compare(field):
  candidate = run((12, 12, 12))
  candidate["context"][field]["changed"] = True
  report = compare_runs(run(), candidate, noise_pct=1)
  assert report["comparison"] == "inconclusive"
  assert "latency" not in report


@pytest.mark.parametrize("samples", [[], [112.1], [1, 2, float("nan")], [1, 2, float("inf")], [1, 2, -1]])
def test_aggregate_or_invalid_samples_not_promoted_into_measurement(samples):
  assert compare_runs(run(), run(samples), noise_pct=1)["comparison"] == "inconclusive"


@pytest.mark.parametrize("field,value", [("profiled", True), ("synchronized", False), ("steady_state", False)])
def test_profiled_cold_or_unsynchronized_timings_inconclusive(field, value):
  before = run()
  before["context"]["timing"][field] = value
  assert compare_runs(before, before, noise_pct=1)["comparison"] == "inconclusive"


def test_unknown_shape_contract_fails_closed():
  before = run()
  del before["context"]["workload"]["semantics"]
  assert compare_runs(before, before, noise_pct=1)["comparison"] == "inconclusive"


def test_incorrect_candidate_can_be_diagnosed_but_not_accepted():
  candidate = run((12, 12, 12))
  candidate["correctness_passed"] = False
  report = compare_runs(run(), candidate, noise_pct=1)
  assert report["observation"] == "regression"
  assert report["correctness_passed"] is False
  assert "correctness" in " ".join(report["missing_evidence"])


def test_paired_trace_identifies_hypothesis_and_specific_next_experiment(tmp_path):
  before, after = run(), run((12, 12, 12))
  (tmp_path / "before.json").write_text(json.dumps([event("custom", 0, 100)]))
  (tmp_path / "after.json").write_text(json.dumps({"traceEvents": [event("custom", i * 200, 100) for i in range(22)]}))
  before["trace"] = spec(path="before.json")
  after["trace"] = spec(path="after.json")
  report = compare_runs(before, after, noise_pct=1, control_dir=tmp_path, candidate_dir=tmp_path)
  assert report["trace_deltas"]["custom_call"]["delta_events_per_repetition"] == 21
  hypothesis = report["hypotheses"][0]
  assert hypothesis["confidence"] == "hypothesis_only"
  assert "queue materializations" in hypothesis["next_experiment"]
  assert report["root_cause"] == "unproven"


def test_bad_or_missing_trace_keeps_valid_latency_observation(tmp_path):
  candidate = run((12, 12, 12))
  candidate["trace"] = spec(path="absent.json")
  report = compare_runs(run(), candidate, noise_pct=1, candidate_dir=tmp_path)
  assert report["observation"] == "regression"
  assert "invalid trace" in " ".join(report["missing_evidence"])


def test_incomplete_trace_keeps_timing_but_cannot_generate_hypotheses(tmp_path):
  path = tmp_path / "trace.json"
  path.write_text(json.dumps([event("custom", 0, 100)]))
  before, after = run(), run((12, 12, 12))
  for record in (before, after):
    record["trace"] = spec(path=str(path))
    record["trace"]["selectors"]["custom_call"]["expected_events_per_repetition"] = 2
  report = compare_runs(before, after, noise_pct=1)
  assert report["observation"] == "regression"
  assert not report["hypotheses"]
  assert "incomplete declared event coverage" in str(report["missing_evidence"])


def test_more_wait_events_without_more_wait_time_do_not_suggest_wait_bottleneck(tmp_path):
  before, after = run(), run((12, 12, 12))
  for name, record, events in (
    ("before", before, [event("wait", 0, 100)]),
    ("after", after, [event("wait", i * 100, 10) for i in range(5)]),
  ):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(events))
    record["trace"] = spec(path=str(path), selectors={"wait": {"pid": 1, "name_regex": "^wait$"}})
  report = compare_runs(before, after, noise_pct=1)
  assert report["trace_deltas"]["wait"]["delta_events_per_repetition"] == 4
  assert report["trace_deltas"]["wait"]["delta_union_ms_per_repetition"] < 0
  assert not report["hypotheses"]


def test_mismatched_selectors_not_compared(tmp_path):
  path = tmp_path / "trace.json"
  path.write_text(json.dumps([event("custom", 0, 100)]))
  before = run()
  before["trace"] = spec(path=str(path))
  after = copy.deepcopy(before)
  after["trace"]["selectors"]["custom_call"]["name_regex"] = "pallas"
  report = compare_runs(before, after, noise_pct=1)
  assert not report["trace_deltas"]
  assert "different event-name selectors" in " ".join(report["missing_evidence"])


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_invalid_noise_threshold(threshold):
  with pytest.raises(ValueError):
    compare_runs(run(), run(), noise_pct=threshold)


def test_cli_uses_relative_trace_paths_and_writes_report(tmp_path):
  before, after = tmp_path / "control.json", tmp_path / "candidate.json"
  trace_path = tmp_path / "trace.json"
  trace_path.write_text(json.dumps([event("custom", 0, 100)]))
  control, candidate = run(), run((12, 12, 12))
  control["trace"] = spec(path="trace.json")
  candidate["trace"] = spec(path="trace.json")
  before.write_text(json.dumps(control))
  after.write_text(json.dumps(candidate))
  output = tmp_path / "report.json"
  assert main([str(before), str(after), "--noise-pct", "1", "--out", str(output)]) == 0
  report = json.loads(output.read_text())
  assert report["observation"] == "regression"
  assert report["trace_deltas"]["custom_call"]["control"]["matched_events"] == 1
  after.write_text(json.dumps(run((112.1,))))
  assert main([str(before), str(after), "--noise-pct", "1", "--out", str(output)]) == 2


def test_malformed_trace_event_is_reported_as_missing_evidence(tmp_path):
  path = tmp_path / "trace.json"
  path.write_text(json.dumps(["not an event"]))
  candidate = run((12, 12, 12))
  candidate["trace"] = spec(path=str(path))
  report = compare_runs(run(), candidate, noise_pct=1)
  assert report["observation"] == "regression"
  assert "each trace event must be an object" in " ".join(report["missing_evidence"])


def test_different_thread_scope_is_not_compared(tmp_path):
  path = tmp_path / "trace.json"
  path.write_text(json.dumps([event("custom", 0, 100)]))
  control = run()
  control["trace"] = spec(path=str(path))
  candidate = copy.deepcopy(control)
  del candidate["trace"]["selectors"]["custom_call"]["tid"]
  report = compare_runs(control, candidate, noise_pct=1)
  assert not report["trace_deltas"]
  assert "thread scope" in " ".join(report["missing_evidence"])
