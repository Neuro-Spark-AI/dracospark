import json
import subprocess
import sys
from pathlib import Path

from inference_opt.agents import invoke
from inference_opt.evidence import digest, file_hash, lock_plan, seal, source_closure, write


def test_agent_bridge_binds_proposal_to_sealed_evidence(tmp_path, monkeypatch):
    adapter = tmp_path / "code/adapter.py"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_text("fixture")
    command = tmp_path / "agent.json"
    command.write_text('["fixture-agent"]')
    plan = {
        "protocol": 1,
        "adapter_entrypoint": str(adapter),
        "adapter_source_root": str(adapter.parent),
        "campaign_id": "cpu-contract",
        "attempt_ledger": str(tmp_path / "ledger"),
        "concurrency": 1,
        "world_size": 1,
        "adapter": [sys.executable, str(adapter)],
        "sealed_files": source_closure(sys.executable, adapter.parent, command),
        "runtime_identity_sha256": digest({}),
        "arms": {"baseline": {"cache": False}, "candidate": {"cache": True}},
        "treatment_fields": ["cache"],
        "warmups": 1,
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
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    root = tmp_path / "baseline"
    lock_plan(source, root)
    write(root / "outcome.json", {"status": "awaiting_candidate"})
    seal(root)
    response = {
        "baseline_manifest_sha256": file_hash(root / "evidence.manifest.json"),
        "treatment": {"cache": True},
        "hypothesis": "skip repeat encoding",
        "falsifier": "different output or no encoder saving",
    }

    def agent(argv, input, **kwargs):
        packet = json.loads(input)
        assert packet["role"] == "optimize"
        assert packet["allowed_treatment"] == {"cache": True}
        assert packet["evidence_root"] == str(root)
        return subprocess.CompletedProcess(argv, 0, json.dumps(response), "")

    monkeypatch.setattr(subprocess, "run", agent)
    assert invoke(root, "optimize", command, tmp_path / "proposal") == response


def test_run_agent_command_path_does_not_override_subcommand(monkeypatch, capsys):
    import sys

    from inference_opt import __main__

    calls = []
    monkeypatch.setattr(__main__, "invoke", lambda *args: calls.append(args) or {"ok": True})
    monkeypatch.setattr(
        sys,
        "argv",
        ["nsperf", "run-agent", "baseline", "--role", "optimize", "--command", "agent.json", "--output", "proposal"],
    )
    __main__.main()
    assert calls == [(Path("baseline"), "optimize", Path("agent.json"), Path("proposal"))]
    assert json.loads(capsys.readouterr().out) == {"ok": True}
