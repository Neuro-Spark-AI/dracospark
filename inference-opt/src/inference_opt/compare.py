"""Deterministic candidate gates; reviewer prose cannot override these results."""

from __future__ import annotations

import statistics

from .evidence import finite
from .registry import provenance
from .statistics_policy import assessment, validate_policy


def summarize(records, world_size):
    formal = [r for r in records if r["mode"] == "benchmark"]
    completed = [
        r
        for r in formal
        if r["status"] == "completed"
        and r.get("validation", {}).get("passed") is True
        and r.get("performance_comparable") is not False
    ]
    times = [finite(r["elapsed_s"], positive=True) for r in completed]
    result = {
        "attempted": len(formal),
        "validated_completed": len(completed),
        "failures": len(formal) - len(completed),
        "world_size": world_size,
        "scope": "finite concurrency-one window; not saturated capacity",
        "samples_s": times,
        "mean_s": statistics.mean(times) if times else None,
        "median_s": statistics.median(times) if times else None,
    }
    if formal and all(r.get("block", -1) == -1 for r in formal):
        span = finite(formal[-1]["end_monotonic"] - formal[0]["start_monotonic"], positive=True)
        result.update(window_s=span, completed_per_minute=60 * len(completed) / span)
    return result


def compare(plan, records, numerical, *, confirmation=False):
    registered = provenance("dracospark.inference-opt.paired-comparison", __file__, "compare")
    validate_policy(plan["statistics"])
    reasons = []
    blocks = []
    if len(records) != 2 * len(plan["schedule"]):
        reasons.append("unexpected sample cardinality")
    for i, order in enumerate(plan["schedule"]):
        pair = {}
        for arm in order:
            matches = [r for r in records if r["mode"] == "benchmark" and r["block"] == i and r["arm"] == arm]
            if len(matches) != 1:
                reasons.append(f"block {i}/{arm}: missing or duplicate sample")
                continue
            record = matches[0]
            seeds = plan.get("confirmation_block_seeds" if confirmation else "block_seeds", [])
            expected_seed = seeds[i] if seeds else plan["request_seed"]
            if type(record.get("seed")) is not int or record["seed"] != expected_seed:
                reasons.append(f"block {i}/{arm}: observed seed differs from frozen schedule")
                continue
            if (
                record["status"] != "completed"
                or record.get("validation", {}).get("passed") is not True
                or record.get("performance_comparable") is False
            ):
                reasons.append(f"block {i}/{arm}: request/validation failed")
                continue
            if record.get("device_gate") is not True:
                reasons.append(f"block {i}/{arm}: device comparability unestablished")
            pair[arm] = finite(record["elapsed_s"], positive=True)
        if len(pair) == 2:
            blocks.append((pair["candidate"] - pair["baseline"]) / pair["baseline"])
    if numerical.get("passed") is not True or numerical.get("independent") is not True:
        reasons.append("independent matched-output numerical gate missing or failed")
    policy = plan["statistics"]
    if len(blocks) < policy["minimum_blocks"]:
        reasons.append("insufficient paired blocks")
    test = assessment(blocks, policy)
    if not test["passed"]:
        reasons.append("exact sign test or mean guard does not establish practical improvement")
    search_passed = not reasons
    if not confirmation:
        reasons.append("held-out confirmation required before adoption")
    status = (
        "eligible_for_review"
        if not reasons
        else ("requires_confirmation" if search_passed and not confirmation else "rejected")
    )
    return {
        "status": status,
        "analyzer": registered,
        "reasons": reasons,
        "paired_fractional_changes": blocks,
        "mean_fractional_change": statistics.mean(blocks) if blocks else None,
        "exact_sign_test": test,
        "median_fractional_change": statistics.median(blocks) if blocks else None,
        "confidence": policy["confidence"],
        "rule": policy["rule"],
        "limitations": [
            "exact sign test assumes independent representative paired blocks",
            "confidence concerns typical paired gain, not a confidence interval for the mean",
            "search results are not a production promotion",
        ],
    }
