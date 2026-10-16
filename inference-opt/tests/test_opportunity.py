import copy
import json

import pytest

from inference_opt.evidence import file_hash, seal
from inference_opt.opportunity import assess, opportunities


def metrics():
    return {"duration_s": 4, "algorithmic_flops": 220e12, "precision": "bf16", "compute_peak_flop_s": 100e12}


def envelope():
    return {"target_flop_s": 60e12, "operator": "attention", "shape": [128, 64], "precision": "bf16"}


def context():
    return {
        "gpu": "test-gpu",
        "partition": "full",
        "backend": "test-backend",
        "layout": "test-layout",
        "clock_policy": "fixed-test",
        "power_policy": "fixed-test",
        "operator": "attention",
        "shape": [128, 64],
    }


def test_attention_55_to_60_is_local_8_percent_not_45_percent_e2e():
    result = assess(metrics(), envelope())
    assert result["compute_peak_fraction"] == pytest.approx(0.55)
    assert result["attainable_fraction"] == pytest.approx(55 / 60)
    assert result["local_saving_s"] == pytest.approx(1 / 3)
    assert result["e2e_saving_s"] is None
    assert "scenario" in result["reference_status"]
    assert "Near reference" in result["guidance"]
    assert "still be compute-bound" in result["guidance"]


def test_missing_peak_and_envelope_do_not_invent_device_ceiling():
    row = metrics() | {"compute_peak_flop_s": None}
    result = assess(row)
    assert result["compute_peak_fraction"] is None
    assert result["local_saving_s"] is None
    assert result["reference_status"] == "unavailable"


@pytest.mark.parametrize("target", [50e12, 101e12])
def test_inconsistent_target_has_no_rankable_headroom(target):
    result = assess(metrics(), envelope() | {"target_flop_s": target})
    assert result["local_saving_s"] is None
    assert "inconsistent" in result["reference_status"]


@pytest.mark.parametrize("target", [0, -1, float("inf"), float("nan"), True])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError):
        assess(metrics(), envelope() | {"target_flop_s": target})


def calibrated(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "context": context() | {"precision": "bf16"},
                "samples_flop_s": [59e12, 60e12, 61e12],
                "flops_definition": "algorithmic useful work",
            }
        )
    )
    seal(tmp_path)
    return envelope() | {"calibration": {"artifact": path.name, "sha256": file_hash(path)}}


def test_sealed_calibration_matches_exact_context(tmp_path):
    ref = calibrated(tmp_path)
    assert assess(metrics(), ref, context(), tmp_path)["reference_status"].startswith("matched")
    for key in context():
        changed = context() | {key: "different"}
        assert "mismatch" in assess(metrics(), ref, changed, tmp_path)["reference_status"]
    assert "incomplete" in assess(metrics(), ref, {}, tmp_path)["reference_status"]


def test_outside_manifest_or_changed_calibration_is_rejected(tmp_path):
    ref = calibrated(tmp_path)
    (tmp_path / "evidence.manifest.json").write_text('{"protocol":1,"files":[]}')
    with pytest.raises(ValueError, match="outside"):
        assess(metrics(), ref, context(), tmp_path)
    ref["calibration"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        assess(metrics(), ref, context(), tmp_path)


def test_rank_local_groups_keep_missing_roof_and_avoid_precision_or_shape_match():
    rank = {"rank": 0, "operators": [{"operator": "attention", "shape": [128, 64], "metrics": metrics()}]}
    original = copy.deepcopy(rank)
    result = opportunities(rank, [envelope()])
    assert result["opportunities"][0]["local_saving_s"] == pytest.approx(1 / 3)
    assert result["opportunities"][0]["roofline_efficiency"] is None
    assert rank == original
    for ref in [envelope() | {"precision": "fp16"}, envelope() | {"shape": [1, 2]}]:
        assert opportunities(rank, [ref])["opportunities"][0]["local_saving_s"] is None
    with pytest.raises(ValueError, match="ambiguous"):
        opportunities(rank, [envelope(), envelope()])


def test_priority_uses_local_saving_not_timeline_share_and_keeps_counter_roofs():
    attention = {"operator": "attention", "shape": [128, 64], "metrics": metrics()}
    gemm_metrics = metrics() | {
        "duration_s": 2,
        "algorithmic_flops": 20e12,
        "roofline_efficiency": 0.5,
        "roof_flop_s": 20e12,
        "bound": "roofline predicts HBM; causal validation required",
        "memory": {"HBM": {"status": "measured", "bytes": 1e9}},
    }
    gemm = {"operator": "gemm", "shape": [128, 64], "metrics": gemm_metrics}
    refs = [envelope(), envelope() | {"operator": "gemm", "target_flop_s": 20e12}]
    result = opportunities({"rank": 3, "operators": [attention, gemm]}, refs)
    rows = result["opportunities"]
    assert rows[0]["operator"] == "gemm"  # 1s possible versus 0.333s, despite less timeline time.
    assert rows[0]["counter_roofs_per_call"][0]["roofline_efficiency"] == 0.5
    assert all(row["e2e_saving_s"] is None for row in rows)


def test_calibration_traversal_and_precision_mismatch(tmp_path):
    ref = calibrated(tmp_path)
    assert "mismatch" in assess(metrics() | {"precision": "fp16"}, ref, context(), tmp_path)["reference_status"]
    ref["calibration"]["artifact"] = "../calibration.json"
    with pytest.raises(ValueError, match="path"):
        assess(metrics(), ref, context(), tmp_path)


def test_zero_headroom_never_falls_back_to_large_duration_for_priority():
    rows = [
        {"operator": "attention", "shape": [128, 64], "metrics": metrics() | {"algorithmic_flops": 240e12}},
        {"operator": "gemm", "shape": [128, 64], "metrics": metrics() | {"duration_s": 1, "algorithmic_flops": 30e12}},
    ]
    refs = [envelope(), envelope() | {"operator": "gemm"}]
    result = opportunities({"rank": 0, "operators": rows}, refs)["opportunities"]
    assert result[0]["operator"] == "gemm"
    assert result[1]["local_saving_s"] == 0
