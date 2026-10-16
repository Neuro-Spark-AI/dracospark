import json
import math
import sys
from pathlib import Path

import pytest
from test_runner import fake_call, make_plan

from inference_opt import runner
from inference_opt.evidence import file_hash, lock_plan, read, seal
from inference_opt.statistics_policy import assessment, exact_tail


@pytest.mark.parametrize("n", [8, 10, 20])
def test_exact_null_error_never_exceeds_predeclared_tail(n):
    # Exhaust the binomial sign counts, not a noisy Monte Carlo assertion.
    rate = sum(math.comb(n, wins) for wins in range(n + 1) if exact_tail(wins, n) <= 0.025) / 2**n
    assert rate <= 0.025


def test_small_samples_ties_and_mean_tail_regressions_do_not_pass():
    policy = {
        "rule": "paired_exact_sign_practical_gain",
        "minimum_blocks": 8,
        "practical_gain_fraction": 0.02,
        "confidence": 0.95,
    }
    assert not assessment([-0.1] * 2, policy)["passed"]
    assert not assessment([-0.02] * 8, policy)["passed"]
    assert not assessment([-0.1] * 9 + [2.0], policy)["passed"]
    assert assessment([-0.1] * 8, policy)["passed"]


@pytest.mark.parametrize("field,value", [("minimum_blocks", 2), ("practical_gain_fraction", 0), ("confidence", 0.9)])
def test_invalid_acceptance_policy_fails_before_attempt(tmp_path, field, value):
    source = make_plan(tmp_path)
    plan = read(source)
    plan["statistics"][field] = value
    source.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        lock_plan(source, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_unsealed_adapter_helper_and_wrong_entrypoint_are_rejected(tmp_path):
    source = make_plan(tmp_path)
    plan = read(source)
    helper = Path(plan["adapter_source_root"]) / "hidden.py"
    helper.write_text('raise RuntimeError("unsealed timing producer")')
    with pytest.raises(ValueError, match="sealed"):
        lock_plan(source, tmp_path / "unsealed")
    helper.unlink()
    plan["adapter"] = [sys.executable, "-c", 'print("fake timings")', plan["adapter_entrypoint"]]
    source.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="entrypoint"):
        lock_plan(source, tmp_path / "wrong-command")


def prepare_search(tmp_path, monkeypatch):
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
    return source, baseline, proposal


def test_retry_is_consumed_even_after_failed_search(tmp_path, monkeypatch):
    _, baseline, proposal = prepare_search(tmp_path, monkeypatch)
    runner.campaign(baseline, proposal, tmp_path / "rejected")
    with pytest.raises(ValueError, match="already attempted"):
        runner.campaign(baseline, proposal, tmp_path / "try-until-pass")


def test_failed_confirmation_cannot_be_rerun_with_new_output(tmp_path, monkeypatch):
    _, baseline, proposal = prepare_search(tmp_path, monkeypatch)

    def complete(self, action, **data):
        result = fake_call(self, action, **data)
        if action == "analyze":
            result["analysis"]["coverage"]["complete_model"] = True
        return result

    monkeypatch.setattr(runner.Adapter, "call", complete)
    search = tmp_path / "search"
    runner.campaign(baseline, proposal, search)

    def failed(self, action, **data):
        result = complete(self, action, **data)
        if action == "compare_outputs":
            result["numerical"]["passed"] = False
        return result

    monkeypatch.setattr(runner.Adapter, "call", failed)
    root = tmp_path / "failed-confirmation"
    runner.campaign(baseline, proposal, root, confirmation_root=search)
    with pytest.raises(ValueError, match="already attempted"):
        runner.campaign(baseline, proposal, tmp_path / "confirmation-again", confirmation_root=search)
    # Even if an outcome file claims acceptance, review recomputes the failed
    # numerical gate. Change the self-consistent manifest/receipt to exercise
    # recomputation in addition to the independent ledger-binding check.
    (root / "outcome.json").write_text('{"status":"eligible_for_review"}')
    (root / "evidence.manifest.json").unlink()
    seal(root)
    plan = read(root / "plan.lock.json")
    finish = Path(plan["attempt_ledger"]) / plan["campaign_id"] / "confirmation.finished.json"
    receipt = read(finish)
    receipt["manifest_sha256"] = file_hash(root / "evidence.manifest.json")
    finish.write_text(json.dumps(receipt))
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"verdict": "pass", "evidence_manifest_sha256": file_hash(root / "evidence.manifest.json")})
    )
    assert runner.review(root, review, baseline_root=baseline, search_root=search)["accepted"] is False


def test_resealed_outcome_is_rejected_by_external_ledger(tmp_path, monkeypatch):
    _, baseline, proposal = prepare_search(tmp_path, monkeypatch)
    root = tmp_path / "search"
    runner.campaign(baseline, proposal, root)
    (root / "outcome.json").write_text('{"status":"eligible_for_review"}')
    (root / "evidence.manifest.json").unlink()
    seal(root)
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"verdict": "pass", "evidence_manifest_sha256": file_hash(root / "evidence.manifest.json")})
    )
    with pytest.raises(ValueError, match="ledger"):
        runner.review(root, review, baseline_root=baseline)
