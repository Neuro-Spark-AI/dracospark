"""One baseline, search and confirmation per predeclared campaign; append only."""

from pathlib import Path

from .evidence import digest, file_hash, read, write


def directory(plan):
    root = Path(plan["attempt_ledger"]) / plan["campaign_id"]
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("attempt ledger must not use symlinks")
    root.mkdir(parents=True, exist_ok=True)
    return root


def claim(plan, phase, root, parent=None):
    value = {"plan_sha256": digest(plan), "phase": phase, "root": str(Path(root).resolve()), "parent": parent}
    try:
        write(directory(plan) / (phase + ".started.json"), value)
    except FileExistsError as exc:
        raise ValueError("campaign phase already attempted; no retries or optional stopping") from exc


def finish(plan, phase, root):
    write(
        directory(plan) / (phase + ".finished.json"),
        {
            "plan_sha256": digest(plan),
            "root": str(Path(root).resolve()),
            "manifest_sha256": file_hash(Path(root) / "evidence.manifest.json"),
        },
    )


def verify(plan, phase, root):
    folder = directory(plan)
    start = read(folder / (phase + ".started.json"))
    end = read(folder / (phase + ".finished.json"))
    if (
        start["plan_sha256"] != digest(plan)
        or end["plan_sha256"] != digest(plan)
        or start["root"] != str(Path(root).resolve())
        or end["root"] != start["root"]
        or end["manifest_sha256"] != file_hash(Path(root) / "evidence.manifest.json")
    ):
        raise ValueError("attempt ledger evidence binding differs")
    return start
