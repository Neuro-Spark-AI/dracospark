"""Evidence-qualified local headroom estimates, never an inferred E2E critical path."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from .evidence import file_hash, finite
from .registry import provenance

KEY = "dracospark.inference-opt.opportunity"
CONTEXT_KEYS = {"gpu", "partition", "backend", "layout", "clock_policy", "power_policy"}


def assess(metrics, envelope=None, context=None, root=None):
    """An envelope is an advisory target, not proof of a causal hardware bound."""
    seconds = finite(metrics["duration_s"], positive=True)
    rate = finite(metrics["algorithmic_flops"], positive=True) / seconds
    peak = metrics.get("compute_peak_flop_s")
    if peak is not None:
        finite(peak, positive=True)
    result = {
        "duration_s": seconds,
        "achieved_flop_s": rate,
        "compute_peak_fraction": rate / peak if peak else None,
        "roofline_efficiency": metrics.get("roofline_efficiency"),
        "roofline_bound": metrics.get("bound", "unestablished"),
        "memory": metrics.get("memory", {}),
        "reference_status": "unavailable",
        "target_flop_s": None,
        "attainable_fraction": None,
        "local_saving_s": None,
        "e2e_saving_s": None,
        "guidance": "Measure matched compute/bandwidth ceilings and HBM/L2 traffic before selecting a kernel change.",
    }
    if envelope is None:
        return result
    result["reference"] = {"envelope": envelope, "observed_context": context}
    target = finite(envelope["target_flop_s"], positive=True)
    result.update(reference_status="scenario; uncalibrated", target_flop_s=target)
    calibration = envelope.get("calibration")
    if calibration is not None:
        if root is None:
            raise ValueError("calibration requires a sealed evidence root")
        base = Path(root).resolve()
        path = (base / calibration["artifact"]).resolve()
        if not path.is_relative_to(base) or file_hash(path) != calibration["sha256"]:
            raise ValueError("calibration path or hash mismatch")
        # Membership, not only a caller-supplied hash, binds the reference to the bundle.
        manifest = json.loads((base / "evidence.manifest.json").read_text())
        if not _manifest_contains(manifest, path.relative_to(base).as_posix(), calibration["sha256"]):
            raise ValueError("calibration is outside the sealed manifest")
        data = json.loads(path.read_text())
        expected = dict(context or {}, precision=metrics["precision"])
        if not CONTEXT_KEYS.issubset(context or {}) or any(not expected[k] for k in CONTEXT_KEYS):
            result["reference_status"] = "scenario; observed hardware context incomplete"
        elif data.get("context") != expected:
            result["reference_status"] = "scenario; calibration context mismatch"
        elif data.get("flops_definition") != "algorithmic useful work":
            result["reference_status"] = "scenario; FLOPs convention mismatch"
        else:
            samples = [finite(v, positive=True) for v in data.get("samples_flop_s", [])]
            if len(samples) < 3 or target != statistics.median(samples):
                raise ValueError("target must equal median of at least three calibrated samples")
            result["reference_status"] = "matched empirical reference; estimate, not a proven ceiling"
    result["attainable_fraction"] = rate / target
    if (peak and target > peak) or rate > target:
        result.update(
            reference_status="inconsistent reference; recalibrate",
            guidance="Check FLOPs, precision, timing, clocks and partition; do not rank this target as an opportunity.",
        )
        return result
    result["local_saving_s"] = seconds * (1 - rate / target)
    if rate / target >= 0.9:
        result["guidance"] = (
            "Near reference: lower expected return from tile/LDS/register tuning. Test less padding/work, "
            "fusion or overlap; preserve quality. A dominant kernel can still be compute-bound."
        )
    else:
        result["guidance"] = (
            "Below reference: use matched counters to test LDS/register spills, occupancy, tiling and memory stalls; "
            "accept a cause only when a controlled change improves ordinary benchmark latency."
        )
    return result


def _manifest_contains(value, relative, digest):
    # Only a manifest file entry can qualify the calibration artifact.
    return any(row.get("path") == relative and row.get("sha256") == digest for row in value.get("files", []))


def opportunities(rank, envelopes=(), root=None):
    """Rank modeled groups by local serialized saving, explicitly excluding E2E inference."""
    analyzer = provenance(KEY, __file__, "opportunities")
    groups = defaultdict(list)
    for row in rank.get("operators", []):
        key = (row["operator"], json.dumps(row["shape"], sort_keys=True), row["metrics"]["precision"])
        finite(row["metrics"]["duration_s"], positive=True)
        finite(row["metrics"]["algorithmic_flops"], positive=True)
        groups[key].append(row)
    result = []
    for (operator, shape, precision), rows in groups.items():
        matches = [
            e
            for e in envelopes
            if e.get("operator") == operator
            and json.dumps(e.get("shape"), sort_keys=True) == shape
            and e.get("precision") == precision
        ]
        if len(matches) > 1:
            raise ValueError("ambiguous performance envelopes")
        # No averaging of distinct peak or memory-roof assumptions into a fictitious roof.
        metrics = dict(rows[0]["metrics"])
        metrics["algorithmic_flops"] = sum(r["metrics"]["algorithmic_flops"] for r in rows)
        metrics["duration_s"] = sum(r["metrics"]["duration_s"] for r in rows)
        same_peak = len({r["metrics"].get("compute_peak_flop_s") for r in rows}) == 1
        if not same_peak:
            metrics["compute_peak_flop_s"] = None
        metrics["roofline_efficiency"] = None
        metrics["memory"] = {}
        metrics["bound"] = "unestablished"
        # Retain per-call counter roofs; aggregating bytes or nested calls would be misleading.
        context = dict(rank.get("performance_context", {}), operator=operator, shape=json.loads(shape))
        advice = assess(metrics, matches[0] if matches else None, context, root)
        advice.update(operator=operator, shape=json.loads(shape), precision=precision, calls=len(rows))
        advice["counter_roofs_per_call"] = [
            {k: r["metrics"].get(k) for k in ("roofline_efficiency", "roof_flop_s", "bound", "memory")} for r in rows
        ]
        result.append(advice)
    result.sort(
        key=lambda r: (
            r["reference_status"].startswith("matched"),
            r["local_saving_s"] is not None,
            r["local_saving_s"] if r["local_saving_s"] is not None else r["duration_s"],
        ),
        reverse=True,
    )
    return {
        "analyzer": analyzer,
        "rank": rank["rank"],
        "opportunities": result,
        "limitations": [
            "Local savings are fixed-work advisory estimates, not measured speedups or acceptance evidence.",
            "Operator calls can overlap or nest; do not sum opportunities, ranks, or scope times into E2E.",
            "E2E headroom unavailable without dependency-complete critical path and an ordinary benchmark replay.",
            "Useful operator compute-peak fraction is not full-model MFU or MFMA busy percentage.",
            "Empirical median is a reachable reference, not a universal maximum or built-in device ceiling.",
        ],
    }
