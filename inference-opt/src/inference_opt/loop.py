"""One bounded autonomous candidate cycle; subsequent cycles get new plans/IDs."""

from pathlib import Path

from .agents import invoke
from .evidence import read, write
from .runner import baseline, campaign, review


def run(plan_path, optimizer_command, reviewer_command, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write(root / "session.json", {"protocol": 1, "maximum_candidates": 1, "production_adoption": False})
    baseline_root = root / "baseline"
    baseline(plan_path, baseline_root)
    if read(baseline_root / "outcome.json")["status"] != "awaiting_candidate":
        write(root / "outcome.json", {"status": "baseline_incomplete"})
        return
    invoke(baseline_root, "optimize", optimizer_command, root / "proposal")
    proposal = root / "proposal/response.json"
    search = root / "search"
    campaign(baseline_root, proposal, search)
    target = search
    if read(search / "outcome.json")["status"] == "requires_confirmation":
        target = root / "confirmation"
        campaign(baseline_root, proposal, target, confirmation_root=search)
    invoke(target, "review", reviewer_command, root / "review")
    result = review(
        target,
        root / "review/response.json",
        baseline_root=baseline_root,
        search_root=search if target != search else None,
    )
    write(
        root / "outcome.json",
        {
            "status": "candidate_qualified" if result["accepted"] else "candidate_not_adopted",
            "result": result,
            "next_cycle": "new plan/attempt and reprofile",
        },
    )
