"""Exercise the installed harness with synthetic requests and deterministic agents."""

import argparse
import json
import sys
from pathlib import Path

from inference_opt.evidence import digest, read, source_closure, verify_bundle, write
from inference_opt.loop import run
from inference_opt.report import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    adapter = Path(__file__).parent / "cpu_fixture/adapter.py"
    agent = Path(__file__).parent / "cpu_fixture/agent.py"
    command = root / "agent-command.json"
    write(command, [sys.executable, str(agent)])
    identity = {"kind": "synthetic-cpu-fixture", "world_size": 1, "concurrency": 1}
    plan = {
        "protocol": 1,
        "adapter_entrypoint": str(adapter),
        "adapter_source_root": str(adapter.parent),
        "campaign_id": "synthetic-cycle",
        "attempt_ledger": str(root / "ledger"),
        "evidence_kind": "synthetic_fixture",
        "concurrency": 1,
        "world_size": 1,
        "profile_coverage_key": "all_steps_profiled",
        "adapter": [sys.executable, str(adapter)],
        "runtime_identity_sha256": digest(identity),
        "sealed_files": source_closure(sys.executable, adapter.parent, command),
        "arms": {"baseline": {"fixture_treatment": False}, "candidate": {"fixture_treatment": True}},
        "treatment_fields": ["fixture_treatment"],
        "warmups": 1,
        "baseline_repetitions": 2,
        "request_seed": 42,
        "block_seeds": list(range(10, 18)),
        "schedule": [["baseline", "candidate"], ["candidate", "baseline"]] * 4,
        "confirmation_schedule": [["candidate", "baseline"], ["baseline", "candidate"]] * 4,
        "confirmation_block_seeds": list(range(20, 28)),
        "statistics": {
            "rule": "paired_exact_sign_practical_gain",
            "minimum_blocks": 8,
            "practical_gain_fraction": 0.02,
            "confidence": 0.95,
        },
    }
    source = root / "plan.json"
    write(source, plan)
    run(source, command, command, root / "cycle")
    for name in ("baseline", "search", "confirmation"):
        verify_bundle(root / "cycle" / name)
    render(root / "cycle/confirmation", root / "report")
    print(
        json.dumps(
            {
                "evidence_kind": "synthetic_fixture",
                "gpu_actions": False,
                "outcome": read(root / "cycle/outcome.json"),
                "root": str(root),
            }
        )
    )


if __name__ == "__main__":
    main()
