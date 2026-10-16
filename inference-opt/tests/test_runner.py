import json
import subprocess
import sys
from pathlib import Path

import pytest

from inference_opt import runner
from inference_opt.evidence import digest, file_hash, lock_plan, read, source_closure, verify_bundle


def make_plan(tmp_path):
    code = tmp_path / "code/adapter.py"
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("# fixture: not GPU evidence\n")
    plan = {
        "protocol": 1,
        "adapter_entrypoint": str(code),
        "adapter_source_root": str(code.parent),
        "campaign_id": "cpu-contract",
        "attempt_ledger": str(tmp_path / "ledger"),
        "concurrency": 1,
        "world_size": 1,
        "adapter": [sys.executable, str(code)],
        "sealed_files": source_closure(sys.executable, code.parent),
        "runtime_identity_sha256": digest({"kind": "fixture", "world_size": 1, "concurrency": 1}),
        "arms": {"baseline": {"cache": False}, "candidate": {"cache": True}},
        "treatment_fields": ["cache"],
        "warmups": 1,
        "request_seed": 42,
        "block_seeds": list(range(10, 18)),
        "confirmation_block_seeds": list(range(20, 28)),
        "baseline_repetitions": 2,
        "schedule": [["baseline", "candidate"], ["candidate", "baseline"]] * 4,
        "confirmation_schedule": [["candidate", "baseline"], ["baseline", "candidate"]] * 4,
        "statistics": {
            "rule": "paired_exact_sign_practical_gain",
            "minimum_blocks": 8,
            "practical_gain_fraction": 0.02,
            "confidence": 0.95,
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def fake_call(self, action, **data):
    if action == "inspect":
        return {"identity": {"kind": "fixture", "world_size": 1, "concurrency": 1}}
    if action == "configure":
        return {"quiescent": True}
    if action == "request":
        self.serial += 1
        seconds = 10 if data["arm"] == "baseline" else 8
        return {
            "record": {
                **data,
                "status": "completed",
                "elapsed_s": seconds,
                "start_monotonic": self.serial * 20,
                "end_monotonic": self.serial * 20 + seconds,
                "device_gate": True,
            }
        }
    if action == "validate":
        return {
            "validations": [{"client_request_key": r["client_request_key"], "passed": True} for r in data["records"]]
        }
    if action == "analyze":
        return {"analysis": {"coverage": {"complete_model": data["record"]["arm"] == "baseline"}}}
    if action == "compare_outputs":
        return {"numerical": {"passed": True, "independent": True}}
    raise ValueError(action)


def test_reprofile_failure_blocks_candidate_and_review(tmp_path, monkeypatch):
    plan_path = make_plan(tmp_path)
    monkeypatch.setattr(runner.Adapter, "call", fake_call)
    baseline = tmp_path / "baseline"
    runner.baseline(plan_path, baseline)
    proposal = {
        "baseline_manifest_sha256": file_hash(baseline / "evidence.manifest.json"),
        "treatment": {"cache": True},
        "hypothesis": "fixture",
        "falsifier": "fixture",
    }
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal))
    campaign = tmp_path / "campaign"
    runner.campaign(baseline, proposal_path, campaign)
    verify_bundle(campaign)
    assert read(campaign / "outcome.json")["status"] == "rejected"
    assert any("reprofile" in s for s in read(campaign / "comparison.json")["reasons"])
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"evidence_manifest_sha256": file_hash(campaign / "evidence.manifest.json"), "verdict": "pass"})
    )
    assert runner.review(campaign, review, baseline_root=baseline)["accepted"] is False
    with pytest.raises(ValueError, match="qualifi"):
        runner.campaign(baseline, proposal_path, tmp_path / "confirmation", confirmation_root=campaign)


def test_adapter_timeout_records_unknown_and_does_not_retry(tmp_path, monkeypatch):
    source = make_plan(tmp_path)
    root = tmp_path / "run"
    plan = lock_plan(source, root)
    calls = []

    def timeout(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired("fixture", 1)

    monkeypatch.setattr(runner, "run_adapter", timeout)
    adapter = runner.Adapter(plan, root)
    with pytest.raises(RuntimeError, match="submission_unknown"):
        adapter.call("request", client_request_key="durable-before-POST")
    with pytest.raises(RuntimeError, match="reconcile"):
        adapter.configure("baseline")
    assert len(calls) == 1
    assert read(root / "operations/0001-request.request.json")["client_request_key"] == "durable-before-POST"
    assert read(root / "operations/0001-request.result.json")["status"] == "submission_unknown"


def test_raw_artifacts_are_copied_and_hashed(tmp_path, monkeypatch):
    source = make_plan(tmp_path)
    root = tmp_path / "run"
    plan = lock_plan(source, root)
    raw = tmp_path / "native.trace.json"
    raw.write_text('{"traceEvents":[]}')
    reply = {
        "protocol": 1,
        "status": "ok",
        "artifacts": [{"name": "native.trace.json", "path": str(raw), "sha256": file_hash(raw)}],
    }
    monkeypatch.setattr(
        runner, "run_adapter", lambda *a, **k: subprocess.CompletedProcess(["fixture"], 0, json.dumps(reply), "")
    )
    runner.Adapter(plan, root).call("analyze")
    assert file_hash(root / "artifacts/native.trace.json") == file_hash(raw)
    assert Path(root / "operations/0001-analyze.result.json").exists()


def test_declared_u4_cannot_relabel_a_u1_runtime():
    identity = {"world_size": 1, "concurrency": 1}
    plan = {"runtime_identity_sha256": digest(identity), "world_size": 4, "concurrency": 1}
    with pytest.raises(RuntimeError, match="topology"):
        runner.validate_identity(plan, identity)


def test_same_basename_attempts_have_unique_request_keys(tmp_path, monkeypatch):
    source = make_plan(tmp_path)
    monkeypatch.setattr(runner.Adapter, "call", fake_call)
    roots = [tmp_path / "loop-one/baseline", tmp_path / "loop-two/baseline"]
    keys = []
    for root in roots:
        plan = lock_plan(source, root)
        keys.append(runner.Adapter(plan, root).request("baseline", "benchmark", 0)["client_request_key"])
    assert keys[0] != keys[1]


def test_confirmation_uses_its_own_longer_seed_schedule(tmp_path, monkeypatch):
    source = make_plan(tmp_path)
    plan = read(source)
    plan["block_seeds"] = list(range(10, 18))
    plan["confirmation_schedule"] = [["candidate", "baseline"], ["baseline", "candidate"]] * 5
    plan["confirmation_block_seeds"] = list(range(20, 30))
    source.write_text(json.dumps(plan))
    root = tmp_path / "run"
    locked = lock_plan(source, root)
    monkeypatch.setattr(runner.Adapter, "call", fake_call)
    adapter = runner.Adapter(locked, root)
    pair = [adapter.request(arm, "benchmark", 0, block=5, confirmation=True) for arm in ("candidate", "baseline")]
    assert [record["seed"] for record in pair] == [25, 25]
    assert adapter.request("baseline", "benchmark", 0, block=3)["seed"] == 13


def test_adapter_cannot_ignore_the_frozen_seed(tmp_path, monkeypatch):
    source = make_plan(tmp_path)
    root = tmp_path / "run"
    plan = lock_plan(source, root)

    def wrong_seed(self, action, **data):
        result = fake_call(self, action, **data)
        result["record"]["seed"] = 7
        return result

    monkeypatch.setattr(runner.Adapter, "call", wrong_seed)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner.Adapter(plan, root).request("baseline", "benchmark", 0)


def test_baseline_requires_literal_validation_pass(tmp_path, monkeypatch):
    source = make_plan(tmp_path)

    def nonboolean(self, action, **data):
        result = fake_call(self, action, **data)
        if action == "validate":
            for validation in result["validations"]:
                validation["passed"] = "false"
        return result

    monkeypatch.setattr(runner.Adapter, "call", nonboolean)
    root = tmp_path / "baseline"
    runner.baseline(source, root)
    assert read(root / "outcome.json")["status"] == "failed"


def test_rollback_failure_blocks_confirmation_and_reviewer_override(tmp_path, monkeypatch):
    source = make_plan(tmp_path)
    monkeypatch.setattr(runner.Adapter, "call", fake_call)
    baseline = tmp_path / "baseline"
    runner.baseline(source, baseline)
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps(
            {
                "baseline_manifest_sha256": file_hash(baseline / "evidence.manifest.json"),
                "treatment": {"cache": True},
                "hypothesis": "fixture",
                "falsifier": "fixture",
            }
        )
    )

    def complete_profile(self, action, **data):
        result = fake_call(self, action, **data)
        if action == "analyze":
            result["analysis"]["coverage"]["complete_model"] = True
        return result

    original = runner.Adapter.configure
    calls = 0

    def fail_last(self, arm):
        nonlocal calls
        calls += 1
        if calls == 20:
            raise RuntimeError("fixture rollback transport failure")
        original(self, arm)

    monkeypatch.setattr(runner.Adapter, "call", complete_profile)
    monkeypatch.setattr(runner.Adapter, "configure", fail_last)
    target = tmp_path / "search"
    runner.campaign(baseline, proposal, target)
    assert read(target / "rollback.json")["baseline_restored"] is False
    assert read(target / "outcome.json")["status"] == "rejected"
    verify_bundle(target)
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"evidence_manifest_sha256": file_hash(target / "evidence.manifest.json"), "verdict": "pass"})
    )
    assert runner.review(target, review, baseline_root=baseline)["accepted"] is False
