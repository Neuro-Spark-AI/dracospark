"""Subprocess adapter orchestration with sealed evidence and bounded attempts."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from . import ledger
from .compare import compare, summarize
from .evidence import coverage_key, digest, file_hash, lock_plan, read, seal, verify_bundle, verify_plan, write
from .processes import run_adapter


class Adapter:
    def __init__(self, plan, root):
        self.plan, self.root = plan, Path(root)
        self.attempt_id = read(self.root / "attempt.json")["id"]
        self.serial = 0
        self.submission_unknown = False

    def call(self, action, **data):
        if self.submission_unknown and action != "reconcile":
            raise RuntimeError("submission_unknown: reconcile before another adapter operation")
        verify_plan(self.root)
        self.serial += 1
        operation = f"{self.serial:04d}-{action}"
        request = {"protocol": 1, "action": action, "operation_id": operation, "run_id": self.attempt_id, **data}
        write(self.root / "operations" / f"{operation}.request.json", request)
        try:
            result = run_adapter(
                self.plan["adapter"],
                input=json.dumps(request),
                timeout=self.plan.get("adapter_timeout_s", 1800),
            )
        except BaseException as exc:
            if not isinstance(exc, (subprocess.TimeoutExpired, KeyboardInterrupt)) and not hasattr(exc, "cleanup"):
                raise
            self.submission_unknown = True
            write(
                self.root / "operations" / f"{operation}.result.json",
                {
                    "status": "submission_unknown",
                    "reason": "adapter transport interrupted",
                    "process_cleanup": getattr(exc, "cleanup", {"requires_reconciliation": True}),
                },
            )
            for label in ("output", "stderr"):
                content = getattr(exc, label, None) or ""
                if isinstance(content, bytes):
                    content = content.decode(errors="replace")
                (self.root / "operations" / f"{operation}.partial-{label}.txt").write_text(content)
            # No retry; resume must reconcile the durable adapter journal first.
            raise RuntimeError("submission_unknown: reconcile before continuing") from exc
        (self.root / "operations" / f"{operation}.stderr.txt").write_text(result.stderr)
        try:
            response = json.loads(
                result.stdout, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f"nonfinite response {s}"))
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"adapter response invalid; exit={result.returncode}") from exc
        write(self.root / "operations" / f"{operation}.result.json", response)
        if result.returncode or response.get("protocol") != 1 or response.get("status") != "ok":
            raise RuntimeError(f"adapter {action} failed: {response}")
        verify_plan(self.root)
        for artifact in response.get("artifacts", []):
            relative = Path(artifact["name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe raw artifact name")
            source = Path(artifact["path"])
            if source.is_symlink() or file_hash(source) != artifact["sha256"]:
                raise ValueError("raw artifact changed before collection")
            target = self.root / "artifacts" / relative
            if target.exists():
                if file_hash(target) != artifact["sha256"]:
                    raise ValueError("conflicting immutable artifact")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if file_hash(target) != artifact["sha256"]:
                    raise ValueError("raw artifact changed during collection")
        return response

    def configure(self, arm):
        response = self.call("configure", arm=arm, treatment=self.plan["arms"][arm])
        if response.get("quiescent") is not True:
            raise RuntimeError("arm switch without quiescence proof")

    def request(self, arm, mode, index, *, block=-1, confirmation=False):
        key = f"{self.attempt_id}-{arm}-{mode}-{block}-{index}"
        seeds = self.plan.get("confirmation_block_seeds" if confirmation else "block_seeds", [])
        seed = seeds[block] if mode == "benchmark" and block >= 0 and seeds else self.plan["request_seed"]
        response = self.call("request", client_request_key=key, arm=arm, mode=mode, seed=seed, block=block)
        record = response["record"]
        if response.get("reconciled"):
            raise RuntimeError("journaled result cannot be counted as fresh measurement")
        if (
            record.get("client_request_key") != key
            or record.get("mode") != mode
            or record.get("arm") != arm
            or record.get("block") != block
            or type(record.get("seed")) is not int
            or record["seed"] != seed
        ):
            raise RuntimeError("adapter request identity mismatch")
        write(self.root / "requests" / f"{key}.json", record)
        if record["status"] != "completed":
            raise RuntimeError("request incomplete; reconcile, do not retry")
        return record


def validate_identity(plan, identity):
    if digest(identity) != plan["runtime_identity_sha256"]:
        raise RuntimeError("runtime identity differs from locked plan")
    if identity.get("world_size") != plan["world_size"] or identity.get("concurrency") != plan["concurrency"]:
        raise RuntimeError("actual topology/concurrency differs from declared plan")


def validate_records(adapter, records):
    result = adapter.call("validate", records=records)
    validations = result["validations"]
    if len(validations) != len(records):
        raise ValueError("validation cardinality mismatch")
    for record in records:
        matches = [v for v in validations if v["client_request_key"] == record["client_request_key"]]
        if len(matches) != 1:
            raise ValueError("validation request correlation mismatch")
        record["validation"] = matches[0]
    return records


def import_profile(adapter, arm, root):
    record = adapter.request(arm, "timeline", 0)
    response = adapter.call("analyze", record=record)
    write(root / f"profile-{arm}.json", response["analysis"])
    return record, response["analysis"]


def baseline(plan_path, root):
    plan = lock_plan(plan_path, root)
    root = Path(root)
    ledger.claim(plan, "baseline", root)
    adapter = Adapter(plan, root)
    try:
        identity = adapter.call("inspect")["identity"]
        validate_identity(plan, identity)
        write(root / "runtime.identity.json", identity)
        adapter.configure("baseline")
        warm = [adapter.request("baseline", "warmup", i) for i in range(plan["warmups"])]
        records = [adapter.request("baseline", "benchmark", i) for i in range(plan.get("baseline_repetitions", 3))]
        validate_records(adapter, warm + records)
        write(root / "benchmark.json", summarize(records, plan["world_size"]))
        trace_record, analysis = import_profile(adapter, "baseline", root)
        validate_records(adapter, [trace_record])
        write(root / "validated.records.json", warm + records + [trace_record])
        passed = (
            all(r["validation"]["passed"] is True for r in warm + records + [trace_record])
            and analysis["coverage"].get(coverage_key(plan)) is True
        )
        write(
            root / "outcome.json",
            {
                "status": "awaiting_candidate" if passed else "failed",
                "profile_coverage": analysis["coverage"],
                "quality": "numerical baseline only; semantic acceptance separate",
            },
        )
    except Exception as exc:
        write(root / "failure.json", {"reason": str(exc), "at_monotonic": time.monotonic()})
        raise
    finally:
        seal(root)
        ledger.finish(plan, "baseline", root)


def campaign(baseline_root, proposal_path, root, *, confirmation_root=None):
    baseline_root, root = Path(baseline_root), Path(root)
    verify_bundle(baseline_root)
    plan = verify_plan(baseline_root)
    ledger.verify(plan, "baseline", baseline_root)
    if not baseline_qualified(baseline_root, plan):
        raise ValueError("baseline validation/profile is incomplete")
    confirmation = confirmation_root is not None
    if confirmation:
        confirmation_root = Path(confirmation_root)
        verify_bundle(confirmation_root)
        search_plan = verify_plan(confirmation_root)
        if digest(search_plan) != digest(plan):
            raise ValueError("search and baseline plans differ")
        ledger.verify(plan, "search", confirmation_root)
        search = recompute_campaign(confirmation_root, plan, confirmation=False)
        if search["status"] != "requires_confirmation":
            raise ValueError("search qualification cannot be reproduced")
        search_proposal = read(confirmation_root / "candidate.json")
        if search_proposal != read(proposal_path):
            raise ValueError("confirmation candidate differs from search candidate")
        # The independent confirmation schedule must already be in the baseline lock.
        if not plan.get("confirmation_schedule") or plan["confirmation_schedule"] == plan["schedule"]:
            raise ValueError("fresh pre-registered confirmation schedule required")
        if plan.get("block_seeds") and not plan.get("confirmation_block_seeds"):
            raise ValueError("pre-registered confirmation seeds required for multi-seed search")
    proposal = read(proposal_path)
    if (
        proposal.get("baseline_manifest_sha256") != file_hash(baseline_root / "evidence.manifest.json")
        or proposal.get("treatment") != plan["arms"]["candidate"]
        or not proposal.get("hypothesis")
        or not proposal.get("falsifier")
    ):
        raise ValueError("candidate must bind baseline evidence and allowed treatment")
    phase = "confirmation" if confirmation else "search"
    parent_binding = {
        "baseline_manifest": file_hash(baseline_root / "evidence.manifest.json"),
        "search_manifest": file_hash(confirmation_root / "evidence.manifest.json") if confirmation else None,
    }
    ledger.claim(plan, phase, root, parent_binding)
    lock_plan(baseline_root / "plan.lock.json", root)
    write(
        root / "parent.json",
        {
            "manifest_sha256": proposal["baseline_manifest_sha256"],
            "search_manifest_sha256": file_hash(confirmation_root / "evidence.manifest.json") if confirmation else None,
            "kind": "confirmation" if confirmation else "search",
        },
    )
    write(root / "candidate.json", proposal)
    adapter = Adapter(plan, root)
    records = []
    final_outcome = None
    try:
        validate_identity(plan, adapter.call("inspect")["identity"])
        all_warmups = []
        for arm in ("baseline", "candidate"):
            adapter.configure(arm)
            warms = [adapter.request(arm, "warmup", i) for i in range(plan["warmups"])]
            validate_records(adapter, warms)
            all_warmups.extend(warms)
            if not all(r["validation"]["passed"] is True for r in warms):
                raise RuntimeError("candidate warmup validation failed")
        write(root / "warmup.records.json", all_warmups)
        schedule = plan["confirmation_schedule"] if confirmation else plan["schedule"]
        for block, order in enumerate(schedule):
            for arm in order:
                adapter.configure(arm)
                records.append(adapter.request(arm, "benchmark", 0, block=block, confirmation=confirmation))
        validate_records(adapter, records)
        numerical = adapter.call("compare_outputs", records=records)["numerical"]
        write(root / "validated.records.json", records)
        write(root / "numerical.json", numerical)
        analysis_plan = {**plan, "schedule": schedule}
        verdict = compare(analysis_plan, records, numerical, confirmation=confirmation)
        adapter.configure("candidate")
        trace, analysis = import_profile(adapter, "candidate", root)
        validate_records(adapter, [trace])
        write(root / "profile-validation.json", trace["validation"])
        if trace["validation"]["passed"] is not True or analysis["coverage"].get(coverage_key(plan)) is not True:
            verdict["status"] = "rejected"
            verdict["reasons"].append("candidate reprofile validation or coverage failed")
        final_outcome = {
            "status": verdict["status"],
            "reasons": verdict["reasons"],
            "profile_coverage": analysis["coverage"],
            "production_adoption": False,
        }
    except Exception as exc:
        write(root / "failure.json", {"reason": str(exc), "at_monotonic": time.monotonic()})
        raise
    finally:
        if adapter.submission_unknown:
            restored = False
            write(
                root / "rollback.json",
                {
                    "baseline_restored": False,
                    "attempted": False,
                    "status": "skipped_submission_unknown",
                    "requires_reconciliation": True,
                },
            )
        else:
            try:
                adapter.configure("baseline")
                restored = True
                write(root / "rollback.json", {"baseline_restored": True})
            except Exception as exc:
                restored = False
                write(root / "rollback.json", {"baseline_restored": False, "reason": str(exc)})
        if final_outcome is not None:
            if not restored:
                verdict["status"] = "rejected"
                verdict["reasons"].append("baseline restoration failed; runtime requires reconciliation")
                final_outcome.update(status="rejected", reasons=verdict["reasons"])
            write(root / "comparison.json", verdict)
            write(root / "outcome.json", final_outcome)
        seal(root)
        ledger.finish(plan, phase, root)


def baseline_qualified(root, plan):
    validate_identity(plan, read(root / "runtime.identity.json"))
    records = read(root / "validated.records.json")
    modes = {mode: [r for r in records if r["mode"] == mode] for mode in ("warmup", "benchmark", "timeline")}
    return (
        len(records) == plan["warmups"] + plan.get("baseline_repetitions", 3) + 1
        and len(modes["warmup"]) == plan["warmups"]
        and len(modes["benchmark"]) == plan.get("baseline_repetitions", 3)
        and len(modes["timeline"]) == 1
        and all(r.get("status") == "completed" and r.get("validation", {}).get("passed") is True for r in records)
        and read(root / "profile-baseline.json")["coverage"].get(coverage_key(plan)) is True
    )


def recompute_campaign(root, plan, *, confirmation):
    schedule = plan["confirmation_schedule"] if confirmation else plan["schedule"]
    verdict = compare(
        {**plan, "schedule": schedule},
        read(root / "validated.records.json"),
        read(root / "numerical.json"),
        confirmation=confirmation,
    )
    warmups = read(root / "warmup.records.json")
    if (
        len(warmups) != 2 * plan["warmups"]
        or any(sum(r["arm"] == arm for r in warmups) != plan["warmups"] for arm in ("baseline", "candidate"))
        or not all(
            r.get("mode") == "warmup"
            and r.get("status") == "completed"
            and r.get("validation", {}).get("passed") is True
            for r in warmups
        )
    ):
        verdict["reasons"].append("warmup validation failed")
    if (
        read(root / "profile-validation.json").get("passed") is not True
        or read(root / "profile-candidate.json")["coverage"].get(coverage_key(plan)) is not True
    ):
        verdict["reasons"].append("candidate reprofile validation or coverage failed")
    if read(root / "rollback.json").get("baseline_restored") is not True:
        verdict["reasons"].append("baseline restoration failed; runtime requires reconciliation")
    # Search's ordinary held-out requirement is not an additional rejection.
    hard = [r for r in verdict["reasons"] if r != "held-out confirmation required before adoption"]
    if hard:
        verdict["status"] = "rejected"
    return verdict


def review(root, review_path, *, baseline_root, search_root=None):
    root, baseline_root = Path(root), Path(baseline_root)
    verify_bundle(root)
    verify_bundle(baseline_root)
    plan = verify_plan(baseline_root, execution=False)
    ledger.verify(plan, "baseline", baseline_root)
    if not baseline_qualified(baseline_root, plan):
        raise ValueError("baseline qualification cannot be reproduced")
    if digest(verify_plan(root, execution=False)) != digest(plan):
        raise ValueError("review plan differs from trusted baseline")
    parent = read(root / "parent.json")
    if parent["manifest_sha256"] != file_hash(baseline_root / "evidence.manifest.json"):
        raise ValueError("review baseline ancestry mismatch")
    confirmation = parent["kind"] == "confirmation"
    phase = "confirmation" if confirmation else "search"
    ledger.verify(plan, phase, root)
    if confirmation:
        if search_root is None:
            raise ValueError("confirmation review requires explicit trusted search root")
        search_root = Path(search_root)
        verify_bundle(search_root)
        ledger.verify(plan, "search", search_root)
        if digest(verify_plan(search_root, execution=False)) != digest(plan):
            raise ValueError("search plan differs from trusted baseline")
        if parent["search_manifest_sha256"] != file_hash(search_root / "evidence.manifest.json"):
            raise ValueError("confirmation search ancestry mismatch")
        if recompute_campaign(search_root, plan, confirmation=False)["status"] != "requires_confirmation":
            raise ValueError("search qualification cannot be reproduced")
        if read(root / "candidate.json") != read(search_root / "candidate.json"):
            raise ValueError("confirmation candidate differs from search")
    candidate = read(root / "candidate.json")
    if (
        candidate["treatment"] != plan["arms"]["candidate"]
        or candidate["baseline_manifest_sha256"] != parent["manifest_sha256"]
    ):
        raise ValueError("candidate contract differs from trusted baseline")
    value = read(review_path)
    if value.get("evidence_manifest_sha256") != file_hash(root / "evidence.manifest.json"):
        raise ValueError("review does not bind sealed evidence")
    verdict = recompute_campaign(root, plan, confirmation=confirmation)
    return {
        "accepted": value.get("verdict") == "pass" and verdict["status"] == "eligible_for_review",
        "baseline_restored": read(root / "rollback.json").get("baseline_restored") is True,
        "deterministic_status": verdict["status"],
        "recomputed": verdict,
        "review": value,
        "trust_boundary": (
            "operator-owned baseline/attempt ledger and registered producer receipts; not producer authenticity"
        ),
        "production_adoption": False,
    }


def reconcile(plan_path, key, root):
    plan = lock_plan(plan_path, root)
    try:
        result = Adapter(plan, root).call("reconcile", client_request_key=key)
        write(Path(root) / "reconciliation.json", result)
    finally:
        seal(root)
