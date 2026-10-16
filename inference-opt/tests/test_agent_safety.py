import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_runner import make_plan

from inference_opt import agents, codex_role
from inference_opt.evidence import lock_plan, read, seal, source_closure


def test_agent_cannot_change_adapter_between_measurements(tmp_path, monkeypatch):
    source = make_plan(tmp_path)
    plan = read(source)
    command = tmp_path / "agent.json"
    command.write_text('["fixture-agent"]')
    plan["sealed_files"] = source_closure(sys.executable, plan["adapter_source_root"], command)
    source.write_text(json.dumps(plan))
    root = tmp_path / "baseline"
    lock_plan(source, root)
    seal(root)

    def malicious(*args, **kwargs):
        Path(plan["adapter_entrypoint"]).write_text('print("fake accepted timings")')
        return subprocess.CompletedProcess(args[0], 0, "{}", "")

    monkeypatch.setattr(agents.subprocess, "run", malicious)
    with pytest.raises(ValueError, match="changed"):
        agents.invoke(root, "optimize", command, tmp_path / "response")


def test_non_git_codex_bridge_retains_read_only_policy(tmp_path, monkeypatch, capsys):
    assert not (tmp_path / ".git").exists()
    packet = {"role": "optimize", "allowed_treatment": {"attention_plain": True}, "evidence_root": str(tmp_path)}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(packet)))
    monkeypatch.setattr(sys, "argv", ["bridge", "--codex-bin", "fixture-codex"])

    def cli(argv, **kwargs):
        assert "--skip-git-repo-check" in argv
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert kwargs["cwd"] == tmp_path
        assert "--model" not in argv
        Path(argv[argv.index("--output-last-message") + 1]).write_text('{"fixture":true}')
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(codex_role.subprocess, "run", cli)
    codex_role.main()
    assert json.loads(capsys.readouterr().out)["fixture"] is True
