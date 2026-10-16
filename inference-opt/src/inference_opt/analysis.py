"""Scoped Kineto accounting and explicit, counter-qualified roofline analysis."""

from __future__ import annotations

import bisect
import collections
import gzip
import json

from .evidence import DEFAULT_PROFILE_COVERAGE_KEY, file_hash, finite
from .registry import provenance

ANALYZER = "dracospark.inference-opt.kineto-roofline"
VERSION = 1
# Annotation-name prefixes for repeated model steps, per-step blocks and named stages. Adapters may
# override them to match the scopes their instrumented runtime emits.
DEFAULT_SCOPE_PREFIXES = {"step": "model.step.", "block": "model.block.", "stage": "model.stage."}


COVERAGE_FIELDS = frozenset(
    {
        "device_events",
        "runtime_correlated",
        "step_count",
        "step_block_pairs",
        "observed_stages",
        "step_block_ids",
        "expected_steps",
        "expected_blocks_per_step",
        "memory_counters",
        "full_model_flops",
    }
)


def union_us(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        finite(start)
        finite(stop)
        if stop < start:
            raise ValueError("reversed device interval")
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def roofline(flops, duration_s, precision, peaks, traffic=None):
    finite(flops, positive=True)
    finite(duration_s, positive=True)
    peak = peaks.get(precision)
    result = {
        "precision": precision,
        "algorithmic_flops": flops,
        "duration_s": duration_s,
        "achieved_flop_s": flops / duration_s,
        "compute_peak_flop_s": peak,
        "memory": {},
        "roofline_efficiency": None,
        "bound": "unestablished",
        "flops_definition": "algorithmic useful work",
    }
    if peak is None:
        return result
    finite(peak, positive=True)
    result["compute_peak_fraction"] = flops / duration_s / peak
    ceilings = [(peak, "compute")]
    for level, row in (traffic or {}).items():
        if row.get("source") != "measured_counter" or row.get("launch_signature_match") is not True:
            result["memory"][level] = {"status": "unavailable", "reason": "unqualified traffic"}
            continue
        count = finite(row["bytes"], positive=True)
        bandwidth = finite(row["peak_bytes_s"], positive=True)
        intensity = flops / count
        ceilings.append((intensity * bandwidth, level))
        result["memory"][level] = {
            "status": "measured",
            "bytes": count,
            "arithmetic_intensity": intensity,
            "achieved_bytes_s": count / duration_s,
            "peak_bytes_s": bandwidth,
        }
    if len(ceilings) > 1:
        roof, limiter = min(ceilings)
        result.update(
            roof_flop_s=roof,
            roofline_efficiency=flops / duration_s / roof,
            bound=f"roofline predicts {limiter}; causal validation required",
        )
    return result


def gemm_flops(op):
    dims = op.get("args", {}).get("Input Dims", [])
    types = op.get("args", {}).get("Input type", [])
    offset = 1 if op["name"] == "aten::addmm" else 0
    if op["name"] not in ("aten::mm", "aten::addmm") or len(dims) <= offset + 1:
        return None
    a, b = dims[offset : offset + 2]
    if len(a) != 2 or len(b) != 2 or a[1] != b[0] or not all(v > 0 for v in a + b):
        return None
    operands = types[offset : offset + 2]
    precision = (
        "bf16"
        if len(operands) == 2 and all("BFloat16" in str(t) for t in operands)
        else "fp16"
        if len(operands) == 2 and all("Half" in str(t) for t in operands)
        else None
    )
    if precision is None:
        return None
    return {
        "flops": 2 * a[0] * a[1] * b[1],
        "precision": precision,
        "shape": dims[offset : offset + 2],
        "definition": "2MNK recorded post-sharding shapes",
    }


def scope_prefixes(overrides=None):
    prefixes = {**DEFAULT_SCOPE_PREFIXES, **(overrides or {})}
    if set(prefixes) != set(DEFAULT_SCOPE_PREFIXES):
        raise ValueError("scope prefixes must be exactly step, block and stage")
    if any(not isinstance(v, str) or not v for v in prefixes.values()) or len(set(prefixes.values())) != 3:
        raise ValueError("scope prefixes must be distinct nonempty strings")
    return prefixes


def analyze_trace(
    path,
    *,
    rank,
    device,
    expected_steps,
    expected_blocks,
    peaks,
    flop_resolver=gemm_flops,
    prefixes=None,
    coverage_key=DEFAULT_PROFILE_COVERAGE_KEY,
):
    registered = provenance(ANALYZER, __file__, "analyze_trace")
    prefixes = scope_prefixes(prefixes)
    step, block, stage = prefixes["step"], prefixes["block"], prefixes["stage"]
    if not isinstance(coverage_key, str) or not coverage_key or coverage_key in COVERAGE_FIELDS:
        raise ValueError("coverage key must be a nonempty string distinct from built-in coverage fields")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as source:
        data = json.load(source)
    recorded_rank = data.get("distributedInfo", {}).get("rank")
    if recorded_rank is not None and recorded_rank != rank:
        raise ValueError("trace rank differs from capture manifest")
    events = data["traceEvents"]
    indices = {}
    annotations = collections.defaultdict(list)
    runtimes, ops = {}, {}
    for event in events:
        args = event.get("args", {})
        if event.get("cat") in ("cuda_runtime", "hip_runtime") and "correlation" in args:
            runtimes[(event.get("pid"), args["correlation"])] = event
        if event.get("cat") == "cpu_op" and "External id" in args:
            ops[(event.get("pid"), args["External id"])] = event
        if event.get("cat") == "user_annotation" and event.get("dur", 0) > 0:
            annotations[(event.get("pid"), event.get("tid"))].append(event)
    for thread, ranges in annotations.items():
        ranges.sort(key=lambda e: e["ts"])
        indices[thread] = ([e["ts"] for e in ranges], ranges)

    def resolve_scope(launch):
        if not launch:
            return []
        starts, ranges = indices.get((launch.get("pid"), launch.get("tid")), ([], []))
        end = bisect.bisect_right(starts, launch["ts"])
        return [e["name"] for e in ranges[:end] if e["ts"] + e["dur"] > launch["ts"]]

    kernels = []
    scope_rows = collections.defaultdict(list)
    operator_rows = collections.defaultdict(list)
    runtime_by_correlation = collections.defaultdict(list)
    op_by_external = collections.defaultdict(list)
    for (_, key), value in runtimes.items():
        runtime_by_correlation[key].append(value)
    for (_, key), value in ops.items():
        op_by_external[key].append(value)
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        args = event.get("args", {})
        finite(event["dur"])
        if event["dur"] < 0:
            raise ValueError("negative kernel duration")
        launches = runtime_by_correlation.get(args.get("correlation"), [])
        launch = launches[0] if len(launches) == 1 else None
        candidates = op_by_external.get(args.get("External id"), [])
        op = candidates[0] if len(candidates) == 1 else None
        scopes = resolve_scope(launch)
        row = {
            "rank": rank,
            "device": device,
            "stream": args.get("stream"),
            "name": event["name"],
            "category": event["cat"],
            "start_us": event["ts"],
            "duration_us": event["dur"],
            "mapped_runtime": launch is not None,
            "scopes": scopes,
            "operator": op["name"] if op else None,
            "external_id": args.get("External id"),
            "launch_signature": {k: args.get(k) for k in ("grid", "block", "registers per thread", "shared memory")},
        }
        kernels.append(row)
        for scope in scopes:
            # Prefix with step/block ancestry to keep repeated module invocations separate.
            ancestry = [] if scope.startswith(stage) else [s for s in scopes if s.startswith((step, block))]
            scope_rows[(tuple(ancestry), scope)].append(row)
        if op is not None:
            operator_rows[(op.get("pid"), args.get("External id"))].append((op, row))
    if not kernels:
        raise ValueError("trace contains no device work")
    start, end = min(k["start_us"] for k in kernels), max(k["start_us"] + k["duration_us"] for k in kernels)
    scope_stats = []
    for (ancestry, name), rows in scope_rows.items():
        scope_stats.append(
            {
                "rank": rank,
                "ancestry": list(ancestry),
                "name": name,
                "kernel_count": len(rows),
                "device_busy_us": union_us([(r["start_us"], r["start_us"] + r["duration_us"]) for r in rows]),
                "scope": "device activity attributed through launch correlation; inclusive",
            }
        )
    roofs = []
    for (_, external), rows in operator_rows.items():
        info = flop_resolver(rows[0][0])
        if info is None:
            continue
        duration = union_us([(r["start_us"], r["start_us"] + r["duration_us"]) for _, r in rows]) / 1e6
        if duration <= 0:
            continue
        roofs.append(
            {
                "external_id": external,
                "operator": rows[0][0]["name"],
                "definition": info["definition"],
                "shape": info["shape"],
                "metrics": roofline(info["flops"], duration, info["precision"], peaks),
            }
        )
    steps = {s for k in kernels for s in k["scopes"] if s.startswith(step)}
    pairs = {
        (tuple(s for s in k["scopes"] if s.startswith(step)), s)
        for k in kernels
        for s in k["scopes"]
        if s.startswith(block)
    }
    mapped = sum(k["mapped_runtime"] for k in kernels)
    step_names = {f"{step}{i}" for i in range(expected_steps)}
    block_names = {f"{block}{i}" for i in range(expected_blocks)}
    per_step = {name: {b for ancestry, b in pairs if name in ancestry} for name in step_names}
    busy = union_us([(r["start_us"], r["start_us"] + r["duration_us"]) for r in kernels])
    return {
        "analyzer": registered,
        "trace_sha256": file_hash(path),
        "rank": rank,
        "device": device,
        "scope_prefixes": prefixes,
        "coverage": {
            "device_events": len(kernels),
            "runtime_correlated": mapped,
            "step_count": len(steps),
            "step_block_pairs": len(pairs),
            "observed_stages": sorted(
                {e["name"] for rows in annotations.values() for e in rows if e["name"].startswith(stage)}
            ),
            "step_block_ids": {s: sorted(v) for s, v in per_step.items()},
            "expected_steps": expected_steps,
            "expected_blocks_per_step": expected_blocks,
            coverage_key: steps == step_names
            and all(v == block_names for v in per_step.values())
            and len(pairs) == expected_steps * expected_blocks
            and mapped == len(kernels),
            "memory_counters": "unavailable",
            "full_model_flops": "partial",
        },
        "observed_device_span_us": end - start,
        "device_busy_union_us": busy,
        "observed_activity_gap_fraction": 1 - busy / (end - start) if end > start else None,
        "scopes": scope_stats,
        "operators": roofs,
        "top_kernels": sorted(kernels, key=lambda r: r["duration_us"], reverse=True)[:30],
        "limitations": [
            "scope times are inclusive and cannot be summed across hierarchy",
            "no dependency-complete critical path or full-model lower bound",
            "missing HBM/L2 counters do not establish compute/memory bound",
            "no cross-rank time alignment inferred",
        ],
    }
