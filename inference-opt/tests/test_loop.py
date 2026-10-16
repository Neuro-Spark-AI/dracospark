import json
import subprocess
import sys
from pathlib import Path

from inference_opt.evidence import read, verify_bundle
from inference_opt.report import render


def test_subprocess_cycle_confirms_and_reviews_without_gpu_or_model(tmp_path):
    driver = Path(__file__).parents[1] / "examples/run_cpu_fixture.py"
    result = subprocess.run(
        [sys.executable, str(driver), "--output", str(tmp_path / "fixture")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["gpu_actions"] is False
    assert receipt["evidence_kind"] == "synthetic_fixture"
    assert receipt["outcome"]["status"] == "candidate_qualified"
    assert receipt["outcome"]["result"]["production_adoption"] is False
    root = Path(receipt["root"])
    for name in ("baseline", "search", "confirmation"):
        verify_bundle(root / "cycle" / name)
    seeds = {row["seed"] for row in read(root / "cycle/confirmation/validated.records.json")}
    assert seeds == set(range(20, 28))
    assert read(root / "cycle/confirmation/rollback.json")["baseline_restored"] is True
    before = (root / "cycle/confirmation/evidence.manifest.json").read_bytes()
    render(root / "cycle/confirmation", root / "second-report")
    assert before == (root / "cycle/confirmation/evidence.manifest.json").read_bytes()
    assert (root / "second-report/opportunities.json").is_file()
    assert "Roofline distance and optimization guidance" in (root / "second-report/report.md").read_text()
