"""Evidence-bound packets for coding agents; agents never select their own reward."""

import json
import subprocess
from pathlib import Path

from .evidence import file_hash, read, verify_bundle, verify_plan, write


def packet(root, role):
    root = Path(root)
    verify_bundle(root)
    plan = verify_plan(root, execution=False)
    summaries = {}
    for name in ("benchmark.json", "comparison.json", "outcome.json"):
        if (root / name).exists():
            summaries[name] = read(root / name)
    if role not in ("optimize", "review"):
        raise ValueError("unknown role")
    base = {
        "protocol": 1,
        "role": role,
        "evidence_root": str(root.resolve()),
        "evidence_manifest_sha256": file_hash(root / "evidence.manifest.json"),
        "plan_sha256": file_hash(root / "plan.lock.json"),
        "contract": plan,
        "summary": summaries,
        "constraints": [
            "Do not modify sealed baseline, evaluator or acceptance policy.",
            "Report missing measurements as unavailable, never zero.",
            "Kernel gains require complete-model confirmation.",
            "Do not deploy production or stop foreign processes.",
        ],
        "artifacts": [r["path"] for r in read(root / "evidence.manifest.json")["files"]],
    }
    if role == "optimize":
        base["instructions"] = (
            "Inspect benchmark and layered profile. Assess the single pre-registered treatment; "
            "do not select another treatment. "
            "Write a JSON proposal with baseline_manifest_sha256, treatment, hypothesis, falsifier, "
            "and expected_metric_changes. Request additional diagnostics when evidence is incomplete. "
            "The runner owns actual measurements and acceptance. Rejection is a valid outcome."
        )
        base["allowed_treatment"] = plan["arms"]["candidate"]
    else:
        base["instructions"] = (
            "Read-only independent review. Verify executed code/data identity, complete outputs, "
            "matched work, frozen statistics, telemetry qualification and profile coverage. "
            "Write evidence_manifest_sha256, verdict pass/fail, and findings. Your prose cannot "
            "override a deterministic rejected or unconfirmed result."
        )
    return base


def invoke(root, role, command_path, output, timeout_s=900):
    """Connect any coding-agent CLI implementing JSON stdin / JSON stdout."""
    plan = verify_plan(root)
    if str(Path(command_path).resolve()) not in {str(Path(row["path"]).resolve()) for row in plan["sealed_files"]}:
        raise ValueError("agent command file must be sealed in the plan")
    ledger_root = Path(plan["attempt_ledger"]) / plan["campaign_id"]
    ledger_before = {str(p): file_hash(p) for p in ledger_root.glob("*.json")}
    trusted_manifest = file_hash(Path(root) / "evidence.manifest.json")
    request = packet(root, role)
    command = read(command_path)
    if not isinstance(command, list) or not command or any(not isinstance(s, str) for s in command):
        raise ValueError("agent command must be an explicit argv array")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "input.json", request)
    write(output / "command.json", {"argv": command, "timeout_s": timeout_s})
    result = subprocess.run(
        command, input=json.dumps(request), capture_output=True, text=True, timeout=timeout_s, check=False
    )
    (output / "stderr.txt").write_text(result.stderr)
    verify_bundle(root)
    verify_plan(root)
    if file_hash(Path(root) / "evidence.manifest.json") != trusted_manifest:
        raise ValueError("agent changed the sealed evidence manifest")
    if {str(p): file_hash(p) for p in ledger_root.glob("*.json")} != ledger_before:
        raise ValueError("agent changed operator-owned attempt ledger")
    if result.returncode:
        raise RuntimeError("agent failed; no proposal or approval applied")
    response = json.loads(result.stdout)
    if role == "optimize":
        if (
            response.get("baseline_manifest_sha256") != request["evidence_manifest_sha256"]
            or response.get("treatment") != request["allowed_treatment"]
            or not response.get("hypothesis")
            or not response.get("falsifier")
        ):
            raise ValueError("agent proposal violates the evidence/treatment contract")
    elif response.get("evidence_manifest_sha256") != request["evidence_manifest_sha256"] or response.get(
        "verdict"
    ) not in ("pass", "fail"):
        raise ValueError("reviewer response is not evidence-bound")
    write(output / "response.json", response)
    return response
