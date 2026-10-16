"""Optional read-only Codex CLI bridge for the JSON agent protocol.

The user's configured model is retained. This wrapper never invokes GPU/SSH
commands itself and never changes the evaluator or measured evidence.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def response_schema(packet):
    if packet["role"] == "optimize":
        treatment = packet["allowed_treatment"]
        controls = {}
        types = {bool: "boolean", int: "integer", float: "number", str: "string", type(None): "null"}
        for name, value in treatment.items():
            if type(value) not in types:
                raise ValueError("this optional CLI bridge supports scalar treatment fields; use another agent wrapper")
            controls[name] = {"type": types[type(value)], "enum": [value]}
        properties = {
            "baseline_manifest_sha256": {"type": "string"},
            "treatment": {
                "type": "object",
                "properties": controls,
                "required": list(controls),
                "additionalProperties": False,
            },
            "hypothesis": {"type": "string"},
            "falsifier": {"type": "string"},
            "expected_metric_changes": {"type": "array", "items": {"type": "string"}},
        }
    elif packet["role"] == "review":
        properties = {
            "evidence_manifest_sha256": {"type": "string"},
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "findings": {"type": "array", "items": {"type": "string"}},
        }
    else:
        raise ValueError("unknown agent role")
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-bin", default="codex", help="Explicit compatible CLI binary")
    args = parser.parse_args()
    packet = json.load(sys.stdin)
    schema = response_schema(packet)
    prompt = (
        "Perform only the read-only evidence role in the following packet. "
        "Inspect its local sealed artifacts if needed. Do not edit any files, use SSH, "
        "submit GPU jobs, contact other chats/apps, or change acceptance criteria. "
        "Return the schema-conforming JSON response. Distinguish missing measurements "
        "from measured results.\n" + json.dumps(packet)
    )
    with tempfile.TemporaryDirectory(prefix="nsperf-agent-role-") as directory:
        root = Path(directory)
        (root / "schema.json").write_text(json.dumps(schema))
        result = subprocess.run(
            [
                args.codex_bin,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(root / "schema.json"),
                "--output-last-message",
                str(root / "response.json"),
                "-",
            ],
            cwd=Path(packet["evidence_root"]).resolve(),
            input=prompt,
            text=True,
            capture_output=True,
            timeout=840,
            check=False,
        )
        sys.stderr.write(result.stderr)
        if result.returncode:
            sys.stderr.write(result.stdout[-4000:])
            raise SystemExit(result.returncode)
        response = json.loads((root / "response.json").read_text())
        print(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
