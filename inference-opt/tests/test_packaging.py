import json
import subprocess
import sys
from pathlib import Path

from inference_opt import analysis
from inference_opt.codex_role import response_schema
from inference_opt.registry import provenance


def test_packaged_registry_resolves_without_source_checkout():
    result = provenance(analysis.ANALYZER, analysis.__file__, "analyze_trace")
    assert result["key"].startswith("dracospark.inference-opt.")
    assert len(result["registry_sha256"]) == 64


def test_agent_schema_admits_only_the_locked_treatment():
    from jsonschema import ValidationError, validate

    treatment = {
        "reference_cache": False,
        "attention_plain": True,
        "attention_skip_padding": False,
        "fusion_block_heads": 4,
    }
    schema = response_schema({"role": "optimize", "allowed_treatment": treatment})
    response = {
        "baseline_manifest_sha256": "a" * 64,
        "treatment": treatment,
        "hypothesis": "reduce launch waste",
        "falsifier": "no E2E improvement",
        "expected_metric_changes": [],
    }
    validate(response, schema)
    response["treatment"] = {**treatment, "reference_cache": True}
    try:
        validate(response, schema)
    except ValidationError:
        pass
    else:
        raise AssertionError("an unregistered treatment must be rejected")


def test_cli_manifest_seals_the_installed_registry_and_modules(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "inference_opt", "manifest"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    manifest = json.loads(result.stdout)
    assert manifest["protocol"] == 1
    files = {Path(row["path"]).name for row in manifest["sealed_files"]}
    assert {"analyzers.json", "runner.py", "agents.py", "registry.py"} <= files
