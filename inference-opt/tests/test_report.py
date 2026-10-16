import json

import pytest

from inference_opt import report
from inference_opt.evidence import seal


def test_report_renders_advisory_headroom_without_claiming_e2e(tmp_path, monkeypatch):
    # Report-only synthetic evidence; this is not an execution/acceptance fixture.
    root = tmp_path / "synthetic"
    root.mkdir()
    (root / "outcome.json").write_text('{"status":"synthetic_fixture"}')
    profile = {
        "coverage": {"complete_model": False},
        "performance_envelopes": [
            {"operator": "attention", "shape": [128, 64], "precision": "bf16", "target_flop_s": 60e12}
        ],
        "ranks": [
            {
                "rank": 0,
                "scope_prefixes": {"step": "decode.step.", "block": "decode.layer.", "stage": "decode.stage."},
                "scopes": [
                    {"name": "decode.step.0", "ancestry": ["decode.step.0"], "device_busy_us": 9000, "kernel_count": 9},
                    {"name": "attention", "ancestry": ["decode.step.0"], "device_busy_us": 4000, "kernel_count": 2},
                ],
                "operators": [
                    {
                        "operator": "attention",
                        "shape": [128, 64],
                        "metrics": {
                            "duration_s": 4,
                            "algorithmic_flops": 220e12,
                            "precision": "bf16",
                            "compute_peak_flop_s": 100e12,
                        },
                    }
                ],
            }
        ],
    }
    (root / "profile-candidate.json").write_text(json.dumps(profile))
    seal(root)
    before = (root / "evidence.manifest.json").read_bytes()
    monkeypatch.setattr(report, "verify_plan", lambda *a, **k: {"world_size": 1, "concurrency": 1})
    output = tmp_path / "report"
    report.render(root, output)
    text = (output / "report.md").read_text()
    assert "55.000%" in text and "91.67%" in text and "333.333" in text
    assert "scenario; uncalibrated" in text and "not model MFU" in text
    # Adapter-declared step/block scopes are per-step totals, not module rows.
    assert "| attention | 4.000 | 2 |" in text and "decode.step.0" not in text
    advice = json.loads((output / "opportunities.json").read_text())[0]
    assert advice["opportunities"][0]["e2e_saving_s"] is None
    assert before == (root / "evidence.manifest.json").read_bytes()
    assert (output / "index.html").exists()
    with pytest.raises(FileExistsError):
        report.render(root, output)
