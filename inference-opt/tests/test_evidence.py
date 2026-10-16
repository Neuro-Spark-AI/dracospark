import json
import sys

import pytest

from inference_opt.evidence import (
    digest,
    lock_plan,
    read,
    seal,
    source_closure,
    verify_bundle,
    verify_plan,
)


def plan(tmp_path):
    code = tmp_path / "code/adapter.py"
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("# pinned adapter\n")
    return {
        "protocol": 1,
        "adapter_entrypoint": str(code),
        "adapter_source_root": str(code.parent),
        "campaign_id": "cpu-contract",
        "attempt_ledger": str(tmp_path / "ledger"),
        "concurrency": 1,
        "world_size": 1,
        "adapter": [sys.executable, str(code)],
        "sealed_files": source_closure(sys.executable, code.parent),
        "arms": {"baseline": {"cache": False}, "candidate": {"cache": True}},
        "treatment_fields": ["cache"],
        "warmups": 2,
        "request_seed": 42,
        "block_seeds": list(range(10, 18)),
        "confirmation_block_seeds": list(range(20, 28)),
        "schedule": [["baseline", "candidate"], ["candidate", "baseline"]] * 4,
        "confirmation_schedule": [["candidate", "baseline"], ["baseline", "candidate"]] * 4,
        "statistics": {
            "rule": "paired_exact_sign_practical_gain",
            "minimum_blocks": 8,
            "practical_gain_fraction": 0.02,
            "confidence": 0.95,
        },
    }


def test_sealed_plan_checks_code_closure(tmp_path):
    value = plan(tmp_path)
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(value))
    root = tmp_path / "run"
    lock_plan(source, root)
    assert digest(verify_plan(root)) == digest(value)
    (tmp_path / "code/adapter.py").write_text("# changed evaluator\n")
    with pytest.raises(ValueError, match="executable"):
        verify_plan(root)


def test_bundle_detects_added_removed_and_changed_artifacts(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "raw.json").write_text('{"status":"failed"}')
    seal(root)
    verify_bundle(root)
    (root / "added.txt").write_text("unregistered output")
    with pytest.raises(ValueError, match="file set"):
        verify_bundle(root)
    (root / "added.txt").unlink()
    (root / "raw.json").write_text('{"status":"passed"}')
    with pytest.raises(ValueError, match="drift"):
        verify_bundle(root)


def test_nonfinite_inputs_fail(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text('{"time":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        read(source)


def test_treatment_cannot_change_undeclared_fields(tmp_path):
    value = plan(tmp_path)
    value["arms"]["candidate"]["frames"] = 1
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="treatment"):
        lock_plan(source, tmp_path / "run")


def test_evidence_traversal_rejected(tmp_path):
    (tmp_path / "evidence.manifest.json").write_text(json.dumps({"files": [{"path": "../escape", "sha256": "0" * 64}]}))
    with pytest.raises(ValueError, match="unsafe"):
        verify_bundle(tmp_path)


def test_nested_manifest_is_ordinary_evidence(tmp_path):
    (tmp_path / "artifacts").mkdir()
    nested = tmp_path / "artifacts/evidence.manifest.json"
    nested.write_text("original")
    seal(tmp_path)
    verify_bundle(tmp_path)
    nested.write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        verify_bundle(tmp_path)


def test_offline_code_closure_survives_original_path_loss(tmp_path):
    value = plan(tmp_path)
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(value))
    root = tmp_path / "run"
    lock_plan(source, root)
    (tmp_path / "code/adapter.py").unlink()
    assert verify_plan(root, execution=False) == value


def test_explicit_eight_rank_plan_is_admitted_without_relabeling_runtime(tmp_path):
    value = plan(tmp_path)
    value["world_size"] = 8
    source = tmp_path / "eight-rank.json"
    source.write_text(json.dumps(value))
    locked = lock_plan(source, tmp_path / "eight-rank")
    assert locked["world_size"] == 8
    value["world_size"] = 16
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="topology"):
        lock_plan(source, tmp_path / "sixteen-rank")
