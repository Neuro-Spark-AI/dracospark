"""Content-addressed plans and evidence; existing attempts are never overwritten."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from pathlib import Path

from .statistics_policy import validate_policy

# The adapter's profile analysis reports this boolean coverage field unless the plan names another.
DEFAULT_PROFILE_COVERAGE_KEY = "complete_model"


def coverage_key(plan):
    return plan.get("profile_coverage_key", DEFAULT_PROFILE_COVERAGE_KEY)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(canonical(value).decode() + "\n")


def finite(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("expected finite number")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("invalid numeric measurement")
    return value


def source_closure(executable, adapter_source_root, *extra):
    """Collect the installed evaluator, executable and complete adapter Python tree."""
    package = Path(__file__).resolve().parent
    paths = {p.resolve() for p in package.glob("*.py")}
    paths.add((package / "analyzers.json").resolve())
    paths.add(Path(shutil.which(str(executable)) or executable).resolve())
    paths.update(p.resolve() for p in Path(adapter_source_root).rglob("*.py") if "__pycache__" not in p.parts)
    paths.update(Path(p).resolve() for p in extra)
    return [{"path": str(p), "sha256": file_hash(p)} for p in sorted(paths)]


def verify_adapter_sources(plan):
    entry = Path(plan.get("adapter_entrypoint", ""))
    source = Path(plan.get("adapter_source_root", ""))
    if not entry.is_absolute() or not source.is_absolute() or not entry.resolve().is_relative_to(source.resolve()):
        raise ValueError("absolute adapter entrypoint and source root required")
    argv = plan["adapter"]
    direct = Path(argv[0]).resolve() == entry.resolve()
    python = Path(argv[0]).name.startswith("python") and len(argv) >= 2 and Path(argv[1]).resolve() == entry.resolve()
    if not (direct or python):
        raise ValueError("adapter argv must execute its declared entrypoint directly or through Python")
    sources = {str(p.resolve()) for p in source.rglob("*.py") if "__pycache__" not in p.parts}
    sources.add(str(entry.resolve()))
    package = Path(__file__).resolve().parent
    sources.update(str(p.resolve()) for p in package.glob("*.py"))
    sources.add(str((package / "analyzers.json").resolve()))
    executable = Path(shutil.which(argv[0]) or argv[0]).resolve()
    sources.add(str(executable))
    for argument in argv[1:]:
        if Path(argument).is_file():
            sources.add(str(Path(argument).resolve()))
    sealed = {str(Path(row["path"]).resolve()): row["sha256"] for row in plan["sealed_files"]}
    if not sources <= set(sealed):
        raise ValueError("adapter entrypoint/import sources must all be sealed")
    for file in sources:
        if file_hash(file) != sealed[file]:
            raise ValueError("locked adapter executable/import source changed")


def lock_plan(plan_path, root):
    plan = read(plan_path)
    canonical(plan)
    if plan.get("protocol") != 1 or type(plan.get("concurrency")) is not int or plan.get("concurrency") != 1:
        raise ValueError("first protocol requires concurrency one")
    if type(plan.get("world_size")) is not int or plan.get("world_size") not in (1, 2, 4, 8):
        raise ValueError("explicit world_size topology of 1, 2, 4 or 8 required")
    if not isinstance(plan.get("adapter"), list) or not plan["adapter"]:
        raise ValueError("adapter must be an argv array")
    if any(not isinstance(s, str) or not s for s in plan["adapter"]):
        raise ValueError("invalid adapter argv")
    if not plan.get("sealed_files"):
        raise ValueError("explicit code and input closure required")
    for item in plan["sealed_files"]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("sealed input/code drift")
    arms = plan.get("arms", {})
    if set(arms) != {"baseline", "candidate"}:
        raise ValueError("exactly two named arms required")
    changed = {
        k for k in set(arms["baseline"]) | set(arms["candidate"]) if arms["baseline"].get(k) != arms["candidate"].get(k)
    }
    if not changed or changed != set(plan.get("treatment_fields", [])):
        raise ValueError("treatment fields do not match arm differences")
    schedule = plan.get("schedule", [])
    if len(schedule) < 2 or any(sorted(b) != ["baseline", "candidate"] for b in schedule):
        raise ValueError("paired blocks must contain each arm once")
    policy = plan["statistics"]
    validate_policy(policy)
    for stage in ("schedule", "confirmation_schedule"):
        pairs = plan.get(stage, [])
        if len(pairs) < policy["minimum_blocks"] or any(sorted(order) != ["baseline", "candidate"] for order in pairs):
            raise ValueError("each stage requires the predeclared minimum paired blocks")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", plan.get("campaign_id", "")):
        raise ValueError("explicit campaign_id required")
    ledger = Path(plan.get("attempt_ledger", ""))
    if not ledger.is_absolute():
        raise ValueError("absolute external attempt_ledger required")
    for seeds in ("block_seeds", "confirmation_block_seeds"):
        if not plan.get(seeds):
            raise ValueError("predeclared search and held-out seeds required")
    if set(plan["block_seeds"]) & set(plan["confirmation_block_seeds"]):
        raise ValueError("confirmation seeds must be held out from search")
    if type(plan.get("warmups")) is not int or plan["warmups"] < 1:
        raise ValueError("explicit per-arm warmup required")
    repetitions = plan.get("baseline_repetitions", 3)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("baseline repetitions must be a positive integer")
    coverage = coverage_key(plan)
    if not isinstance(coverage, str) or not coverage:
        raise ValueError("explicit profile coverage key must be a nonempty string")
    if type(plan.get("request_seed")) is not int:
        raise ValueError("explicit frozen request seed required")
    if plan.get("block_seeds") is not None and (
        len(plan["block_seeds"]) != len(schedule) or any(type(seed) is not int for seed in plan["block_seeds"])
    ):
        raise ValueError("paired seed schedule must match block count")
    if plan.get("confirmation_block_seeds") is not None and (
        len(plan["confirmation_block_seeds"]) != len(plan.get("confirmation_schedule", []))
        or any(type(seed) is not int for seed in plan["confirmation_block_seeds"])
    ):
        raise ValueError("confirmation seed schedule must match block count")
    verify_adapter_sources(plan)
    root = Path(root)
    source_root = Path(plan["adapter_source_root"]).resolve()
    if root.resolve().is_relative_to(source_root):
        raise ValueError("artifact root must be outside adapter sources")
    if ledger.resolve().is_relative_to(root.resolve()):
        raise ValueError("attempt ledger must be outside immutable artifact root")
    root.mkdir(parents=True, exist_ok=False)
    write(root / "plan.lock.json", plan)
    write(root / "plan.sha256.json", {"sha256": digest(plan)})
    write(root / "attempt.json", {"id": uuid.uuid4().hex, "root": str(root.resolve())})
    closure = root / "closure"
    closure.mkdir()
    for item in plan["sealed_files"]:
        destination = closure / item["sha256"]
        if not destination.exists():
            shutil.copyfile(item["path"], destination)
        if file_hash(destination) != item["sha256"]:
            raise ValueError("code/input changed during closure collection")
    return plan


def verify_plan(root, *, execution=True):
    root = Path(root)
    plan = read(root / "plan.lock.json")
    if digest(plan) != read(root / "plan.sha256.json")["sha256"]:
        raise ValueError("locked plan changed")
    if execution:
        verify_adapter_sources(plan)
    for item in plan["sealed_files"]:
        path = Path(item["path"]) if execution else root / "closure" / item["sha256"]
        if file_hash(path) != item["sha256"]:
            raise ValueError("locked executable/input changed")
    return plan


def seal(root):
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("evidence must not contain symlinks")
        if path.is_file() and path != root / "evidence.manifest.json":
            files.append({"path": str(path.relative_to(root)), "sha256": file_hash(path)})
    write(root / "evidence.manifest.json", {"protocol": 1, "files": files})


def verify_bundle(root):
    root = Path(root)
    manifest = read(root / "evidence.manifest.json")
    for row in manifest["files"]:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe evidence path")
        path = root / rel
        ancestors = [root.joinpath(*rel.parts[:i]) for i in range(len(rel.parts) + 1)]
        if any(p.is_symlink() for p in ancestors) or file_hash(path) != row["sha256"]:
            raise ValueError("evidence drift")
    expected = {r["path"] for r in manifest["files"]}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "evidence.manifest.json"}
    if actual != expected:
        raise ValueError("evidence file set changed")
    return manifest
