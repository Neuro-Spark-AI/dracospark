"""Assess and archive a scoped optimization transfer, without deploying it.

``python -m kernel_evolve.transfer plan.json --out record.json`` reads local,
checksum-pinned evidence only. See docs/optimization-transfer.md for the contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kernel_evolve.regression import compare_runs

REQUIRED_CHECKS = (
  "compilation",
  "operator_correctness",
  "structural_correctness",
  "application_correctness",
  "memory",
)


def _canonical(value):
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
  return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _keys(value, required, optional=()):
  if not isinstance(value, dict):
    raise ValueError("expected an object")
  missing = set(required) - value.keys()
  extra = value.keys() - set(required) - set(optional)
  if missing or extra:
    raise ValueError(f"invalid fields: missing={sorted(missing)}, unknown={sorted(extra)}")


def _text(value):
  if not isinstance(value, str) or not value.strip():
    raise ValueError("expected a nonempty string")


def _strings(value, *, nonempty=True):
  if not isinstance(value, list) or (nonempty and not value):
    raise ValueError("expected a nonempty list" if nonempty else "expected a list")
  for item in value:
    _text(item)
  if len(set(value)) != len(value):
    raise ValueError("duplicate list entries")


def _checksum(value):
  if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
    raise ValueError("expected a lowercase SHA256")


def _no_constant(value):
  raise ValueError(f"non-finite JSON constant: {value}")


def _json(data):
  return json.loads(data, parse_constant=_no_constant)


def binding(plan):
  """Bind acceptance receipts to implementation, target contexts and policy.

  Model names are provenance, not capability predicates. Changed shapes, compiler,
  caller, parameters, guards or policy invalidate previously bound receipts.
  """
  return _digest({key: plan[key] for key in ("method", "candidate", "target", "policy")})


def validate_plan(plan):
  _keys(
    plan,
    ("schema_version", "id", "method", "candidate", "target", "policy"),
    ("checks", "experiments", "interventions"),
  )
  if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
    raise ValueError("unsupported schema_version")
  _text(plan["id"])
  _keys(plan["method"], ("name", "description", "source_revision", "source_context"))
  for key in ("name", "description", "source_revision"):
    _text(plan["method"][key])
  if not isinstance(plan["method"]["source_context"], dict) or not plan["method"]["source_context"]:
    raise ValueError("method requires source_context provenance")
  candidate = plan["candidate"]
  _keys(candidate, ("revision", "parameters", "adaptation", "guards"))
  _text(candidate["revision"])
  _text(candidate["adaptation"])
  if not isinstance(candidate["parameters"], dict):
    raise ValueError("candidate parameters must be an object")
  target = plan["target"]
  _keys(target, ("name", "scopes"))
  _text(target["name"])
  if not isinstance(target["scopes"], dict) or not target["scopes"]:
    raise ValueError("target requires named scopes")
  for name, scope in target["scopes"].items():
    _text(name)
    _keys(scope, ("context", "incumbent_revision"))
    _text(scope["incumbent_revision"])
    _keys(scope["context"], ("workload", "hardware", "runtime", "timing"))
    if any(not isinstance(value, dict) or not value for value in scope["context"].values()):
      raise ValueError("scope context sections must be nonempty objects")
  guards = candidate["guards"]
  if not isinstance(guards, list) or not guards:
    raise ValueError("declare capability guards; an empty contract cannot establish eligibility")
  for guard in guards:
    _keys(guard, ("scope", "path", "op", "value", "reason"))
    if guard["scope"] not in target["scopes"]:
      raise ValueError("guard references an unknown scope")
    if not isinstance(guard["path"], list) or not guard["path"]:
      raise ValueError("guard path must be a nonempty list")
    for component in guard["path"]:
      _text(component)
    _text(guard["reason"])
    op, value = guard["op"], guard["value"]
    if op not in ("equals", "one_of", "divisible_by"):
      raise ValueError("unknown guard operation; expressions are never executed")
    if value is None:
      raise ValueError("guard requirements cannot be unknown")
    if op == "divisible_by" and (type(value) is not int or value <= 0):
      raise ValueError("divisor must be a positive integer")
    if op == "one_of" and (not isinstance(value, list) or not value or None in value):
      raise ValueError("one_of requires nonempty known alternatives")
  policy = plan["policy"]
  _keys(policy, ("objective_scope", "required_scopes", "min_fresh_runs", "noise_pct"))
  _strings(policy["required_scopes"])
  if (
    policy["objective_scope"] not in policy["required_scopes"]
    or not set(policy["required_scopes"]) <= target["scopes"].keys()
  ):
    raise ValueError("objective and required scopes must be declared target scopes")
  if type(policy["min_fresh_runs"]) is not int or policy["min_fresh_runs"] < 2:
    raise ValueError("min_fresh_runs must be an integer >= 2; single runs remain exploratory")
  if type(policy["noise_pct"]) not in (float, int) or policy["noise_pct"] < 0:
    raise ValueError("noise_pct must be finite and nonnegative")
  checks = plan.get("checks", {})
  if not isinstance(checks, dict) or checks.keys() - set(REQUIRED_CHECKS):
    raise ValueError("unknown acceptance checks")
  for check in checks.values():
    _keys(check, ("passed", "binding", "evidence"))
    if check["passed"] is not None and type(check["passed"]) is not bool:
      raise ValueError("check passed must be boolean or null")
    _checksum(check["binding"])
  experiments = plan.get("experiments", [])
  if not isinstance(experiments, list):
    raise ValueError("experiments must be a list")
  ids = []
  for experiment in experiments:
    _keys(experiment, ("id", "scope", "fresh_process_id", "binding", "control", "candidate", "lifecycle"))
    for key in ("id", "fresh_process_id"):
      _text(experiment[key])
    ids.append(experiment["id"])
    _checksum(experiment["binding"])
    if experiment["scope"] not in target["scopes"]:
      raise ValueError("experiment references an unknown scope")
    _keys(experiment["lifecycle"], ("passed", "evidence"))
    if experiment["lifecycle"]["passed"] is not None and type(experiment["lifecycle"]["passed"]) is not bool:
      raise ValueError("lifecycle passed must be boolean or null")
  if len(set(ids)) != len(ids):
    raise ValueError("experiment IDs must be unique")
  interventions = plan.get("interventions", [])
  if not isinstance(interventions, list):
    raise ValueError("interventions must be a list")
  for intervention in interventions:
    _keys(intervention, ("hypothesis", "changed_variable", "held_fixed", "prediction", "conclusion", "evidence"))
    for key in ("hypothesis", "changed_variable", "held_fixed", "prediction"):
      _text(intervention[key])
    if intervention["conclusion"] not in ("supported", "contradicted", "unresolved"):
      raise ValueError("invalid intervention conclusion")
    if not isinstance(intervention["evidence"], list) or not intervention["evidence"]:
      raise ValueError("interventions require evidence")
  _canonical(plan)  # Reject NaN/Infinity anywhere, including nested context/parameters.


def evaluate_guards(plan):
  results = []
  for guard in plan["candidate"]["guards"]:
    observed = plan["target"]["scopes"][guard["scope"]]["context"]
    for component in guard["path"]:
      observed = observed.get(component) if isinstance(observed, dict) else None
    if observed is None:
      status = "unknown"
    else:
      op, required = guard["op"], guard["value"]
      if op == "divisible_by":
        passed = type(observed) is int and observed > 0 and observed % required == 0
      elif op == "equals":
        passed = _canonical(observed) == _canonical(required)
      else:
        passed = any(_canonical(observed) == _canonical(choice) for choice in required)
      status = "pass" if passed else "fail"
    results.append({**guard, "observed": observed, "status": status})
  return results


def assess_transfer(plan, *, base_dir=Path(".")):
  """Return an auditable record. Passed receipts are producer claims, not proof.

  No network, subprocess, dispatch mutation or automated root-cause inference.
  Missing/stale evidence cannot advance a record to ready_for_review.
  """
  validate_plan(plan)
  current_binding = binding(plan)
  evidence = []

  def artifact(ref, *, measurement=False):
    _keys(ref, ("path", "sha256"))
    _text(ref["path"])
    _checksum(ref["sha256"])
    path = Path(ref["path"])
    if not path.is_absolute():
      path = base_dir / path
    digest = hashlib.sha256()
    raw = None
    if measurement:
      raw = path.read_bytes()
      digest.update(raw)
    else:
      with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
          digest.update(chunk)
    if digest.hexdigest() != ref["sha256"]:
      raise ValueError(f"artifact checksum mismatch: {ref['path']}")
    saved = {"path": str(path.resolve()), "sha256": digest.hexdigest()}
    data = None
    if measurement:
      data = _json(raw)
      if not isinstance(data, dict):
        raise ValueError("measurement must be an object")
      saved["measurement"] = data
    evidence.append(saved)
    return data, path

  guards = evaluate_guards(plan)
  checks = {}
  for name in REQUIRED_CHECKS:
    check = plan.get("checks", {}).get(name)
    status = "missing"
    error = None
    if check:
      try:
        artifact(check["evidence"])
        status = (
          "stale"
          if check["binding"] != current_binding
          else "pass"
          if check["passed"] is True
          else "fail"
          if check["passed"] is False
          else "unknown"
        )
      except (OSError, ValueError, TypeError) as exc:
        status, error = "invalid", str(exc)
    checks[name] = {"status": status, "error": error}

  experiments = []
  seen_measurements = set()
  for experiment in plan.get("experiments", []):
    result = {
      "id": experiment["id"],
      "scope": experiment["scope"],
      "fresh_process_id": experiment["fresh_process_id"],
      "issues": [],
      "lifecycle_passed": False,
    }
    try:
      control, control_path = artifact(experiment["control"], measurement=True)
      candidate, candidate_path = artifact(experiment["candidate"], measurement=True)
      scope = plan["target"]["scopes"][experiment["scope"]]
      if experiment["binding"] != current_binding:
        result["issues"].append("stale experiment binding")
      if _canonical(control.get("context")) != _canonical(scope["context"]) or _canonical(
        candidate.get("context")
      ) != _canonical(scope["context"]):
        result["issues"].append("measurement context does not match the declared target scope")
      if (
        control.get("revision") != scope["incumbent_revision"]
        or candidate.get("revision") != plan["candidate"]["revision"]
      ):
        result["issues"].append("implementation revisions do not match the declared incumbent/candidate")
      pair_id = (
        experiment["scope"],
        _digest([{key: run.get(key) for key in ("context", "revision", "samples_ms")} for run in (control, candidate)]),
      )
      if pair_id in seen_measurements:
        result["issues"].append("duplicate measurement content is not a fresh-process repetition")
      seen_measurements.add(pair_id)
      if not result["issues"]:
        result["comparison"] = compare_runs(
          control,
          candidate,
          noise_pct=plan["policy"]["noise_pct"],
          control_dir=control_path.parent,
          candidate_dir=candidate_path.parent,
        )
      lifecycle = experiment["lifecycle"]
      if lifecycle["evidence"] is not None:
        artifact(lifecycle["evidence"])
        result["lifecycle_passed"] = lifecycle["passed"] is True
      if not result["lifecycle_passed"]:
        result["issues"].append("runtime lifecycle acceptance is missing or failed")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
      result["issues"].append(str(exc))
    experiments.append(result)

  scopes = {}
  for name in plan["target"]["scopes"]:
    runs = [run for run in experiments if run["scope"] == name]
    comparable = [run for run in runs if run.get("comparison", {}).get("comparison") == "comparable"]
    valid = [run for run in comparable if not run["issues"] and run["comparison"]["correctness_passed"]]
    fresh = {run["fresh_process_id"] for run in valid}
    observations = [run["comparison"]["observation"] for run in comparable]
    status = (
      "incorrect"
      if any(not run["comparison"]["correctness_passed"] for run in comparable)
      else "regressed"
      if "regression" in observations
      else "needs_validation"
      if len(fresh) < plan["policy"]["min_fresh_runs"] or len(valid) != len(runs)
      else "improved"
      if all(value == "improvement" for value in observations)
      else "no_demonstrated_gain"
    )
    scopes[name] = {"status": status, "fresh_validated_runs": len(fresh), "observations": observations}

  for intervention in plan.get("interventions", []):
    for ref in intervention["evidence"]:
      artifact(ref)
  required = [scopes[name]["status"] for name in plan["policy"]["required_scopes"]]
  objective = scopes[plan["policy"]["objective_scope"]]["status"]
  decision = (
    "ineligible"
    if any(g["status"] == "fail" for g in guards)
    else "needs_contract"
    if any(g["status"] == "unknown" for g in guards)
    else "validation_failed"
    if "incorrect" in required or any(c["status"] == "fail" for c in checks.values())
    else "regressed_in_target"
    if "regressed" in required
    else "needs_validation"
    if "needs_validation" in required or any(c["status"] != "pass" for c in checks.values())
    else "no_demonstrated_gain"
    if objective != "improved"
    else "ready_for_review"
  )
  return {
    "schema_version": 1,
    "id": plan["id"],
    "binding": current_binding,
    "decision": decision,
    "production_promotion": "not_performed",
    "root_cause": "unproven",
    "guards": guards,
    "checks": checks,
    "scopes": scopes,
    "experiments": experiments,
    "plan": plan,
    "evidence": evidence,
    "limitations": [
      "Eligibility applies only to the declared predicates; guard completeness requires code review.",
      "Contexts, independent process IDs and acceptance receipts are producer declarations, not attestations.",
      "Each scope is compared separately; kernel wins are never substituted for caller or request evidence.",
      "Ready for review means the declared objective passed, not deployment approval or a proven root cause.",
      "Keep raw traces and compiler artifacts; measurement snapshots do not embed their large external files.",
    ],
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("plan", type=Path)
  destination = parser.add_mutually_exclusive_group(required=True)
  destination.add_argument("--out", type=Path, help="new immutable JSON record; existing files are never overwritten")
  destination.add_argument("--binding", action="store_true", help="print target binding before collecting evidence")
  args = parser.parse_args(argv)
  try:
    plan = _json(args.plan.read_text())
    validate_plan(plan)
    if args.binding:
      print(binding(plan))
      return 0
    report = assess_transfer(plan, base_dir=args.plan.parent)
    with args.out.open("x") as stream:
      stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['id']}: {report['decision']} (production unchanged)")
    return 0
  except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
    parser.error(str(exc))


if __name__ == "__main__":
  raise SystemExit(main())
