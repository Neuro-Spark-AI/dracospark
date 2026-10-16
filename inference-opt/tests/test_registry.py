import json

import pytest

from inference_opt.evidence import file_hash
from inference_opt.registry import provenance

KEY = "example-org.model-adapter.counter-match"
REPO = "example-org/model-adapter"
SOURCE = "adapters/example/analyze.py"


def adapter_registry(tmp_path, code_sha256):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "protocol": 1,
                "analyzers": [
                    {
                        "key": KEY,
                        "status": "ACTIVE",
                        "code_sha256": code_sha256,
                        "source": {"repo": REPO, "path": SOURCE, "symbol": "one"},
                    }
                ],
            }
        )
    )
    return registry


def test_cross_repo_analyzer_must_match_active_registered_source(tmp_path):
    code = tmp_path / SOURCE
    code.parent.mkdir(parents=True)
    code.write_text("# CPU provenance fixture, not GPU evidence\n")
    registry = adapter_registry(tmp_path, file_hash(code))
    args = {"registry_path": registry, "source_repo": REPO, "source_path": SOURCE}
    result = provenance(KEY, code, "one", **args)
    assert len(result["code_sha256"]) == len(result["registry_sha256"]) == 64
    with pytest.raises(ValueError, match="repository mismatch"):
        provenance(KEY, code, "one", **{**args, "source_repo": "foreign/repo"})
    with pytest.raises(ValueError, match="executable path"):
        provenance(KEY, code, "one", **{**args, "source_path": "foreign.py"})
    with pytest.raises(ValueError, match="both repo and path"):
        provenance(KEY, code, "one", source_repo=REPO, registry_path=registry)


def test_external_analyzer_requires_user_supplied_registry(tmp_path):
    code = tmp_path / SOURCE
    code.parent.mkdir(parents=True)
    code.write_text("# CPU provenance fixture, not GPU evidence\n")
    with pytest.raises(ValueError, match="user-supplied registry_path"):
        provenance(KEY, code, "one", source_repo=REPO, source_path=SOURCE)
    # The packaged registry holds only built-in analyzers.
    with pytest.raises(ValueError, match="absent"):
        provenance(KEY, code, "one")


def test_matching_path_cannot_hide_changed_analyzer_bytes(tmp_path):
    code = tmp_path / SOURCE
    code.parent.mkdir(parents=True)
    code.write_text("# registered analyzer\n")
    registry = adapter_registry(tmp_path, file_hash(code))
    code.write_text("# modified analyzer")
    with pytest.raises(ValueError, match="code hash"):
        provenance(KEY, code, "one", source_repo=REPO, source_path=SOURCE, registry_path=registry)
