"""Synthetic protocol fixture. Every duration/device receipt is invented, not measured."""

import argparse
import json
import sys


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    request = json.load(sys.stdin)
    action = request["action"]
    response = {"protocol": 1, "status": "ok"}
    if action == "inspect":
        response["identity"] = {"kind": "synthetic-cpu-fixture", "world_size": 1, "concurrency": 1}
    elif action == "configure":
        response["quiescent"] = True
    elif action == "request":
        duration = 10 if request["arm"] == "baseline" else 8
        start = int(request["operation_id"].split("-")[0]) * 20
        response["record"] = {
            **request,
            "status": "completed",
            "elapsed_s": duration,
            "start_monotonic": start,
            "end_monotonic": start + duration,
            "device_gate": True,
            "device_gate_scope": "synthetic fixture only",
        }
    elif action == "validate":
        response["validations"] = [
            {"client_request_key": r["client_request_key"], "passed": True} for r in request["records"]
        ]
    elif action == "compare_outputs":
        response["numerical"] = {"passed": True, "independent": True, "scope": "synthetic fixture only"}
    elif action == "analyze":
        response["analysis"] = {
            "coverage": {"all_steps_profiled": True},
            "ranks": [],
            "scope": "synthetic profile; no GPU trace",
        }
    else:
        raise ValueError("unsupported fixture action")
    print(json.dumps(response))


if __name__ == "__main__":
    main()
