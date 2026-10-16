import json
import os
import sys
import time

import pytest
from test_runner import fake_call, make_plan

from inference_opt import runner
from inference_opt.evidence import file_hash, read, source_closure, verify_bundle
from inference_opt.processes import ProcessGroupTimeout, run_adapter

pytestmark = pytest.mark.skipif(os.name != "posix", reason="adapter execution uses POSIX process groups")


def test_successful_adapter_returns_its_json_and_exit_code():
    result = run_adapter(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"], input='{"protocol":1}', timeout=5
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"protocol": 1}


@pytest.mark.parametrize("timeout", [None, True, 0, -1, float("nan")])
def test_invalid_timeout_is_rejected_before_start(tmp_path, timeout):
    marker = tmp_path / "must-not-start"
    code = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    with pytest.raises(ValueError, match="finite positive"):
        run_adapter([sys.executable, "-c", code], input="", timeout=timeout)
    assert not marker.exists()


def test_timeout_kills_a_real_grandchild_with_inherited_pipes(tmp_path):
    marker = tmp_path / "late-work"
    ready = tmp_path / "child-ready"
    child = (
        f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); "
        f"time.sleep(3); Path({str(marker)!r}).touch()"
    )
    parent = (
        f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "print('started',flush=True); time.sleep(10)"
    )
    start = time.monotonic()
    with pytest.raises(ProcessGroupTimeout) as raised:
        run_adapter([sys.executable, "-c", parent], input="", timeout=1)
    assert raised.value.cleanup["signal_sent"] is True
    assert raised.value.cleanup["leader_reaped"] is True
    assert ready.exists()
    # The old subprocess.run timeout kills only the parent; its child writes
    # this marker after the caller has already begun baseline restoration.
    time.sleep(max(0, 3.5 - (time.monotonic() - start)))
    assert not marker.exists()


def test_campaign_timeout_never_dispatches_baseline_restore(tmp_path, monkeypatch):
    source = make_plan(tmp_path)
    log = tmp_path / "actions"
    ready = tmp_path / "pending-job"
    late = tmp_path / "late-work"
    child = (
        f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); "
        f"time.sleep(3); Path({str(late)!r}).touch()"
    )
    code = tmp_path / "code/adapter.py"
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text(
        "import json,sys,time,subprocess\nfrom pathlib import Path\n"
        "request=json.load(sys.stdin)\naction=request['action']\n"
        f"with Path({str(log)!r}).open('a') as out: out.write(action+'\\n')\n"
        "response={'protocol':1,'status':'ok'}\n"
        "if action=='inspect': response['identity']={'kind':'fixture','world_size':1,'concurrency':1}\n"
        "elif action=='configure': response['quiescent']=True\n"
        f"elif action=='request': subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(10)\n"
        "else: raise RuntimeError('unexpected operation')\n"
        "print(json.dumps(response))\n"
    )
    plan = read(source)
    plan.update(
        adapter=[sys.executable, str(code)],
        adapter_timeout_s=1,
        sealed_files=source_closure(sys.executable, code.parent),
    )
    source.write_text(json.dumps(plan))
    baseline = tmp_path / "baseline"
    with monkeypatch.context() as patched:
        patched.setattr(runner.Adapter, "call", fake_call)
        runner.baseline(source, baseline)
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps(
            {
                "baseline_manifest_sha256": file_hash(baseline / "evidence.manifest.json"),
                "treatment": {"cache": True},
                "hypothesis": "fixture",
                "falsifier": "fixture",
            }
        )
    )
    target = tmp_path / "campaign"
    with pytest.raises(RuntimeError, match="submission_unknown"):
        runner.campaign(baseline, proposal, target)
    assert ready.exists()
    assert log.read_text().splitlines() == ["inspect", "configure", "request"]
    rollback = read(target / "rollback.json")
    assert rollback["attempted"] is False and rollback["requires_reconciliation"] is True
    assert rollback["baseline_restored"] is False
    unknown = read(target / "operations/0003-request.result.json")
    assert unknown["status"] == "submission_unknown"
    assert unknown["process_cleanup"]["leader_reaped"] is True
    verify_bundle(target)


def test_unknown_campaign_skips_restore_even_if_remote_work_outlives_local_cleanup(tmp_path, monkeypatch):
    source = make_plan(tmp_path)
    baseline = tmp_path / "baseline"
    monkeypatch.setattr(runner.Adapter, "call", fake_call)
    runner.baseline(source, baseline)
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps(
            {
                "baseline_manifest_sha256": file_hash(baseline / "evidence.manifest.json"),
                "treatment": {"cache": True},
                "hypothesis": "fixture",
                "falsifier": "fixture",
            }
        )
    )
    operations = []

    def unresolved(self, action, **data):
        operations.append(action)
        if action == "request":
            self.submission_unknown = True
            raise RuntimeError("submission_unknown: remote worker may still run")
        return fake_call(self, action, **data)

    monkeypatch.setattr(runner.Adapter, "call", unresolved)
    target = tmp_path / "unresolved"
    with pytest.raises(RuntimeError, match="submission_unknown"):
        runner.campaign(baseline, proposal, target)
    assert operations == ["inspect", "configure", "request"]
    assert read(target / "rollback.json")["status"] == "skipped_submission_unknown"
    verify_bundle(target)
