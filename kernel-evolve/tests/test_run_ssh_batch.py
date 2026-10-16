"""Tests for the payload the SSH batch driver writes per variant."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from kernel_evolve.evaluator import EvalResult


def _load_runner():
  path = Path(__file__).resolve().parents[1] / "scripts" / "run_ssh_batch.py"
  spec = importlib.util.spec_from_file_location("kernel_evolve_run_ssh_batch", path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def test_saved_payload_uses_the_status_name_the_analyze_skill_reads():
  runner = _load_runner()

  payload = runner.result_payload("v1", EvalResult.incorrect(max_diff=0.5, error="too far", max_scaled_diff=5.0))
  saved = json.loads(json.dumps(payload, allow_nan=False))

  assert saved["variant_id"] == "v1"
  assert saved["status"] == "INCORRECT"
  assert saved["max_diff"] == 0.5
  assert saved["max_scaled_diff"] == 5.0


def test_saved_payload_carries_nulls_for_numbers_json_cannot_hold():
  runner = _load_runner()

  payload = runner.result_payload("v2", EvalResult.incorrect(max_diff=None, error="non-finite", max_scaled_diff=None))
  saved = json.loads(json.dumps(payload, allow_nan=False))

  assert saved["max_diff"] is None
  assert saved["max_scaled_diff"] is None


def test_missing_numbers_print_as_not_available():
  runner = _load_runner()

  assert runner._fmt(None) == "n/a"
  assert runner._fmt(0.5) == "0.5"
