"""python -m inference_opt: benchmark, candidate campaign and evidence verification."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from .agents import invoke, packet
from .evidence import file_hash, verify_bundle
from .loop import run
from .report import render
from .runner import baseline, campaign, reconcile, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest", help="Print the installed core source closure; no GPU actions")
    p = sub.add_parser("baseline")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("loop")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--optimizer-command", type=Path, required=True)
    p.add_argument("--reviewer-command", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run-agent")
    p.add_argument("root", type=Path)
    p.add_argument("--role", choices=("optimize", "review"), required=True)
    p.add_argument("--command", dest="agent_command", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("campaign")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--proposal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--confirm-search", type=Path)
    p = sub.add_parser("verify")
    p.add_argument("root", type=Path)
    p = sub.add_parser("review")
    p.add_argument("root", type=Path)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--search", type=Path)
    p = sub.add_parser("agent-packet")
    p.add_argument("root", type=Path)
    p.add_argument("--role", choices=("optimize", "review"), required=True)
    p = sub.add_parser("reconcile")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--request-key", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        package = Path(__file__).resolve().parent
        files = sorted(package.glob("*.py")) + [package / "analyzers.json"]
        print(
            json.dumps(
                {
                    "protocol": 1,
                    "package_version": version("dracospark-inference-opt"),
                    "sealed_files": [{"path": str(path), "sha256": file_hash(path)} for path in files],
                }
            )
        )
    elif args.command == "baseline":
        baseline(args.plan, args.output)
    elif args.command == "campaign":
        campaign(args.baseline, args.proposal, args.output, confirmation_root=args.confirm_search)
    elif args.command == "verify":
        print(json.dumps(verify_bundle(args.root)))
    elif args.command == "agent-packet":
        print(json.dumps(packet(args.root, args.role)))
    elif args.command == "reconcile":
        reconcile(args.plan, args.request_key, args.output)
    elif args.command == "run-agent":
        print(json.dumps(invoke(args.root, args.role, args.agent_command, args.output)))
    elif args.command == "report":
        print(render(args.root, args.output))
    elif args.command == "loop":
        run(args.plan, args.optimizer_command, args.reviewer_command, args.output)
    else:
        print(json.dumps(review(args.root, args.review, baseline_root=args.baseline, search_root=args.search)))


if __name__ == "__main__":
    main()
