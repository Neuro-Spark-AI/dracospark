"""Synthetic transfer records: these tests contain no measured TPU claims."""

import copy
import hashlib
import json

import pytest

from kernel_evolve.transfer import REQUIRED_CHECKS, assess_transfer, binding, main, validate_plan


def context(scope):
  return {
    "workload": {
      "shapes": {"heads": 14, "tokens": 1024},
      "dtypes": "bf16",
      "layout": "HSD",
      "semantics": {"mask": "dense", "padding": False},
      "input_id": "synthetic",
      "concurrency": 1,
    },
    "hardware": {"device": "synthetic-tpu"},
    "runtime": {"compiler": "synthetic-v1"},
    "timing": {"scope": scope, "clock": "monotonic", "synchronized": True, "steady_state": True, "profiled": False},
  }


def plan():
  return {
    "schema_version": 1,
    "id": "synthetic-transfer",
    "method": {
      "name": "head tiling",
      "description": "Retune local head tiles while preserving visibility",
      "source_revision": "source-rev",
      "source_context": {"heads": 5, "model": "source-example"},
    },
    "candidate": {
      "revision": "candidate-rev",
      "parameters": {"heads_per_tile": 7},
      "adaptation": "Retune head tile from 5 to 7; no semantic changes",
      "guards": [
        {
          "scope": "kernel",
          "path": ["workload", "shapes", "heads"],
          "op": "divisible_by",
          "value": 7,
          "reason": "This implementation has no head-tail handling",
        },
        {
          "scope": "kernel",
          "path": ["workload", "semantics", "mask"],
          "op": "one_of",
          "value": ["dense"],
          "reason": "This implementation supports dense visibility only",
        },
        {
          "scope": "kernel",
          "path": ["workload", "semantics", "padding"],
          "op": "equals",
          "value": False,
          "reason": "This implementation has no padding mask",
        },
      ],
    },
    "target": {
      "name": "target-example",
      "scopes": {
        scope: {"context": context(scope), "incumbent_revision": "control-rev"} for scope in ("kernel", "caller")
      },
    },
    "policy": {
      "objective_scope": "caller",
      "required_scopes": ["kernel", "caller"],
      "min_fresh_runs": 2,
      "noise_pct": 5,
    },
    "checks": {},
    "experiments": [],
    "interventions": [],
  }


def artifact(tmp_path, filename, data):
  path = tmp_path / filename
  path.write_text(json.dumps(data))
  return {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def add_runs(tmp_path, record, scope, factor=0.8, count=2, lifecycle=True):
  for index in range(count):
    tag = f"{scope}-{index}"
    samples = [10 + index * 0.1, 10.2 + index * 0.1, 9.8 + index * 0.1]
    measurement = {
      "context": record["target"]["scopes"][scope]["context"],
      "correctness_passed": True,
      "revision": "control-rev",
      "samples_ms": samples,
    }
    control = artifact(tmp_path, f"{tag}-control.json", measurement)
    candidate = artifact(
      tmp_path,
      f"{tag}-candidate.json",
      {
        **measurement,
        "revision": record["candidate"]["revision"],
        "samples_ms": [x * factor for x in samples],
      },
    )
    record["experiments"].append(
      {
        "id": tag,
        "scope": scope,
        "fresh_process_id": tag,
        "binding": binding(record),
        "control": control,
        "candidate": candidate,
        "lifecycle": {"passed": lifecycle, "evidence": artifact(tmp_path, f"{tag}-exit.json", {"exit_code": 0})},
      }
    )


def add_checks(tmp_path, record):
  for name in REQUIRED_CHECKS:
    record["checks"][name] = {
      "passed": True,
      "binding": binding(record),
      "evidence": artifact(tmp_path, name + ".json", {"synthetic_receipt": name}),
    }


def complete(tmp_path, *, caller_factor=0.8):
  record = plan()
  add_runs(tmp_path, record, "kernel")
  add_runs(tmp_path, record, "caller", factor=caller_factor)
  add_checks(tmp_path, record)
  return record


def assess(record, tmp_path):
  return assess_transfer(record, base_dir=tmp_path)


def test_empty_evidence_is_eligible_but_unvalidated(tmp_path):
  result = assess(plan(), tmp_path)
  assert result["decision"] == "needs_validation"
  assert all(guard["status"] == "pass" for guard in result["guards"])
  assert result["scopes"]["caller"]["fresh_validated_runs"] == 0


@pytest.mark.parametrize("name", ["video-model", "language-model", "arbitrary-new-model", "no-model-operator"])
def test_guards_are_shape_and_semantics_based_not_model_based(tmp_path, name):
  record = plan()
  record["target"]["name"] = name
  record["candidate"]["guards"][0]["value"] = 5
  result = assess(record, tmp_path)
  assert result["decision"] == "ineligible"
  assert result["guards"][0]["observed"] == 14


@pytest.mark.parametrize(
  "value,expected", [(None, "needs_contract"), (True, "ineligible"), (0, "ineligible"), (False, "needs_validation")]
)
def test_unknown_padding_fails_closed_and_booleans_are_not_numbers(tmp_path, value, expected):
  record = plan()
  record["target"]["scopes"]["kernel"]["context"]["workload"]["semantics"]["padding"] = value
  assert assess(record, tmp_path)["decision"] == expected


def test_missing_guard_path_is_unknown(tmp_path):
  record = plan()
  del record["target"]["scopes"]["kernel"]["context"]["workload"]["shapes"]["heads"]
  assert assess(record, tmp_path)["decision"] == "needs_contract"


def test_full_evidence_only_advances_to_review_not_deployment(tmp_path):
  result = assess(complete(tmp_path), tmp_path)
  assert result["decision"] == "ready_for_review"
  assert result["production_promotion"] == "not_performed"
  assert result["root_cause"] == "unproven"
  assert result["evidence"][5]["sha256"]
  assert sum("measurement" in entry for entry in result["evidence"]) == 8


@pytest.mark.parametrize("factor,decision", [(1.01, "no_demonstrated_gain"), (1.2, "regressed_in_target")])
def test_kernel_win_cannot_override_caller_outcome(tmp_path, factor, decision):
  result = assess(complete(tmp_path, caller_factor=factor), tmp_path)
  assert result["scopes"]["kernel"]["status"] == "improved"
  assert result["decision"] == decision


def test_kernel_only_evidence_cannot_fill_caller_scope(tmp_path):
  record = plan()
  add_runs(tmp_path, record, "kernel")
  add_checks(tmp_path, record)
  assert assess(record, tmp_path)["decision"] == "needs_validation"


@pytest.mark.parametrize("change", ["compiler", "shape", "semantics", "caller", "parameters", "revision", "policy"])
def test_changed_target_or_implementation_invalidates_prior_evidence(tmp_path, change):
  record = complete(tmp_path)
  ctx = record["target"]["scopes"]["caller"]["context"]
  if change == "compiler":
    ctx["runtime"]["compiler"] = "v2"
  elif change == "shape":
    ctx["workload"]["shapes"]["tokens"] = 2048
  elif change == "semantics":
    ctx["workload"]["semantics"]["mask"] = "causal"
  elif change == "caller":
    ctx["runtime"]["caller_revision"] = "new-caller"
  elif change == "parameters":
    record["candidate"]["parameters"]["heads_per_tile"] = 14
  elif change == "revision":
    record["candidate"]["revision"] = "candidate-v2"
  else:
    record["policy"]["noise_pct"] = 3
  result = assess(record, tmp_path)
  assert result["decision"] == "needs_validation"
  assert all(check["status"] == "stale" for check in result["checks"].values())
  assert all("stale experiment binding" in run["issues"] for run in result["experiments"])


@pytest.mark.parametrize("mutation", ["checksum", "missing", "wrong_revision", "wrong_context", "malformed"])
def test_bad_artifacts_cannot_be_promoted(tmp_path, mutation):
  record = complete(tmp_path)
  ref = record["experiments"][-1]["candidate"]
  path = tmp_path / ref["path"]
  if mutation == "missing":
    path.unlink()
  elif mutation == "checksum":
    path.write_text("{}")
  else:
    data = json.loads(path.read_text())
    if mutation == "wrong_revision":
      data["revision"] = "wrong"
    elif mutation == "wrong_context":
      data["context"]["runtime"]["compiler"] = "wrong"
    else:
      data["context"] = None
    record["experiments"][-1]["candidate"] = artifact(tmp_path, ref["path"], data)
  result = assess(record, tmp_path)
  assert result["decision"] == "needs_validation"
  assert result["experiments"][-1]["issues"]


@pytest.mark.parametrize("lifecycle", [False, None])
def test_lifecycle_failure_does_not_erase_timing_but_blocks_acceptance(tmp_path, lifecycle):
  record = complete(tmp_path)
  record["experiments"][-1]["lifecycle"]["passed"] = lifecycle
  result = assess(record, tmp_path)
  assert result["experiments"][-1]["comparison"]["observation"] == "improvement"
  assert result["decision"] == "needs_validation"


def test_duplicate_samples_with_new_run_ids_do_not_count_as_repetitions(tmp_path):
  record = complete(tmp_path)
  first, second = record["experiments"][-2:]
  second["control"] = first["control"]
  second["candidate"] = first["candidate"]
  result = assess(record, tmp_path)
  assert result["decision"] == "needs_validation"
  assert result["scopes"]["caller"]["fresh_validated_runs"] == 1


def test_multiple_pairs_from_same_process_do_not_count_as_fresh_runs(tmp_path):
  record = complete(tmp_path)
  record["experiments"][-1]["fresh_process_id"] = record["experiments"][-2]["fresh_process_id"]
  assert assess(record, tmp_path)["scopes"]["caller"]["fresh_validated_runs"] == 1
  assert assess(record, tmp_path)["decision"] == "needs_validation"


def test_incorrect_fast_candidate_is_not_a_winner(tmp_path):
  record = complete(tmp_path)
  ref = record["experiments"][-1]["candidate"]
  data = json.loads((tmp_path / ref["path"]).read_text())
  data["correctness_passed"] = False
  record["experiments"][-1]["candidate"] = artifact(tmp_path, ref["path"], data)
  assert assess(record, tmp_path)["decision"] == "validation_failed"


@pytest.mark.parametrize("check", REQUIRED_CHECKS)
def test_missing_application_memory_or_other_required_receipt_blocks_review(tmp_path, check):
  record = complete(tmp_path)
  del record["checks"][check]
  assert assess(record, tmp_path)["decision"] == "needs_validation"


def test_failed_acceptance_receipt_is_preserved(tmp_path):
  record = complete(tmp_path)
  record["checks"]["application_correctness"]["passed"] = False
  assert assess(record, tmp_path)["decision"] == "validation_failed"


@pytest.mark.parametrize(
  "field,value",
  [
    ("noise_pct", float("nan")),
    ("noise_pct", float("inf")),
    ("noise_pct", True),
    ("min_fresh_runs", 1),
    ("min_fresh_runs", True),
    ("required_scopes", ["kernel"]),
    ("objective_scope", "absent"),
  ],
)
def test_invalid_policy_rejected(field, value):
  record = plan()
  record["policy"][field] = value
  with pytest.raises(ValueError):
    validate_plan(record)


@pytest.mark.parametrize(
  "op,value", [("eval", "print(1)"), ("divisible_by", 0), ("divisible_by", True), ("one_of", []), ("equals", None)]
)
def test_guard_language_does_not_execute_arbitrary_expressions(op, value):
  record = plan()
  record["candidate"]["guards"][0].update(op=op, value=value)
  with pytest.raises(ValueError):
    validate_plan(record)


def test_interventions_record_contradicted_hypotheses_without_establishing_cause(tmp_path):
  record = complete(tmp_path)
  record["interventions"] = [
    {
      "hypothesis": "A layout copy erases the win",
      "changed_variable": "remove one copy",
      "held_fixed": "shapes, mask, runtime, seed",
      "prediction": "caller recovers",
      "conclusion": "contradicted",
      "evidence": [artifact(tmp_path, "ablation.json", {"delta_pct": 0.1})],
    }
  ]
  result = assess(record, tmp_path)
  assert result["plan"]["interventions"][0]["conclusion"] == "contradicted"
  assert result["root_cause"] == "unproven"


def test_cli_records_snapshots_and_never_overwrites_history(tmp_path, capsys):
  record = complete(tmp_path)
  plan_path, output = tmp_path / "plan.json", tmp_path / "record.json"
  plan_path.write_text(json.dumps(record))
  assert main([str(plan_path), "--binding"]) == 0
  assert capsys.readouterr().out.strip() == binding(record)
  assert main([str(plan_path), "--out", str(output)]) == 0
  original = output.read_bytes()
  assert json.loads(original)["decision"] == "ready_for_review"
  with pytest.raises(SystemExit) as exc:
    main([str(plan_path), "--out", str(output)])
  assert exc.value.code == 2
  assert output.read_bytes() == original


def test_unknown_fields_are_errors_not_silently_ignored_policy():
  record = copy.deepcopy(plan())
  record["policy"]["min_fresh_run"] = 100
  with pytest.raises(ValueError, match="unknown"):
    validate_plan(record)
