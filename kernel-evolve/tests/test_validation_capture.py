"""CPU tests of timing protocol; no fixture is a TPU measurement."""

import importlib.util
from pathlib import Path

import pytest


def module():
  path = Path(__file__).parents[1] / "scripts" / "validate_regression_tpu.py"
  spec = importlib.util.spec_from_file_location("capture_validation", path)
  capture = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(capture)
  return capture


def test_paired_samples_alternates_and_synchronizes_whole_output():
  capture = module()
  events = []
  outputs = [{"nested": [object()]}, {"nested": [object()]}]
  ticks = iter(range(0, 16_000_000, 1_000_000))

  def arm(index):
    def invoke():
      events.append(index)
      return outputs[index]

    return invoke

  seen = []
  samples, order = capture.paired_samples(arm(0), arm(1), synchronize=seen.append, samples=4, clock=lambda: next(ticks))
  assert order == events == [0, 1, 1, 0, 0, 1, 1, 0]
  assert samples == [[1.0] * 4, [1.0] * 4]
  assert all(value is outputs[index] for value, index in zip(seen, order, strict=True))


def fixture_events():
  return [
    {"ph": "M", "name": "process_name", "pid": 3, "args": {"name": "/device:TPU:0"}},
    {"ph": "M", "name": "thread_name", "pid": 3, "tid": 4, "args": {"name": "XLA Ops"}},
    {"ph": "M", "name": "thread_name", "pid": 3, "tid": 5, "args": {"name": "Tensor Core Sync Flag"}},
    {"ph": "X", "name": "diagnostic_matmul.1", "pid": 3, "tid": 4, "ts": 10, "dur": 20},
    {"ph": "X", "name": "diagnostic_matmul.1", "pid": 3, "tid": 4, "ts": 50, "dur": 20},
    {"ph": "X", "name": "dracospark_capture", "pid": 8, "tid": 1, "ts": 40, "dur": 100},
  ]


def test_fixture_uses_verified_device_clock_not_shifted_host_window():
  spec = module().fixture_trace_spec(fixture_events(), 2)
  assert spec["window_us"] == [10, 70]
  assert spec["selectors"]["kernel"]["expected_events_per_repetition"] == 1


def test_fixture_does_not_guess_missing_device_iterations():
  with pytest.raises(ValueError, match="expected 3 complete"):
    module().fixture_trace_spec(fixture_events(), 3)


def test_fixture_does_not_guess_device_or_track_identity():
  with pytest.raises(ValueError, match="missing or ambiguous"):
    module().fixture_trace_spec([], 2)
  with pytest.raises(ValueError, match="required device tracks"):
    module().fixture_trace_spec(fixture_events()[:1], 2)
