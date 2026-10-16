"""Analyzer registry; source, symbol and code bytes must match exactly.

Built-in analyzers resolve against the packaged ``analyzers.json``. An adapter's own (external)
analyzer calls ``provenance`` with ``source_repo``, ``source_path`` and ``registry_path`` naming an
adapter-owned registry file with the same protocol-1 schema; seal that file in the plan closure.
"""

from pathlib import Path

from .evidence import file_hash, read


def provenance(key, code, symbol, *, source_repo=None, source_path=None, registry_path=None):
    package = Path(__file__).resolve().parent
    if (source_repo is None) != (source_path is None):
        raise ValueError("external analyzer requires both repo and path")
    if source_path is not None and registry_path is None:
        raise ValueError("external analyzer requires a user-supplied registry_path")
    registry = Path(registry_path) if registry_path is not None else package / "analyzers.json"
    data = read(registry)
    if data.get("protocol") != 1:
        raise ValueError("unsupported analyzer registry protocol")
    matches = [row for row in data["analyzers"] if row["key"] == key]
    if len(matches) != 1:
        raise ValueError("analyzer absent or ambiguous in registry")
    row = matches[0]
    if row["status"] != "ACTIVE":
        raise ValueError("analyzer is not active")
    code = Path(code).resolve()
    if source_path is None:
        if code.parent != package:
            raise ValueError("internal analyzer must be an installed package module")
        expected = "inference-opt/src/inference_opt/" + code.name
        source_repo = "Neuro-Spark-AI/dracospark"
    else:
        if (
            Path(source_path).is_absolute()
            or ".." in Path(source_path).parts
            or not str(code).endswith("/" + source_path)
        ):
            raise ValueError("external analyzer executable path differs from declared source")
        expected = source_path
    source = row["source"]
    if source["repo"] != source_repo:
        raise ValueError("analyzer registry repository mismatch")
    if source["path"] != expected or source["symbol"] != symbol:
        raise ValueError("analyzer registry source mismatch")
    if row.get("code_sha256") != file_hash(code):
        raise ValueError("analyzer code hash differs from registered source")
    return {"key": key, "version": 1, "code_sha256": file_hash(code), "registry_sha256": file_hash(registry)}
