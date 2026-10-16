"""Multi-rank causal fields; do not infer transfer bandwidth from waiting kernels."""

from collections import defaultdict

from .evidence import finite
from .registry import provenance


def collective_diagnostics(records, world_size, calibration=None):
    registered = provenance("dracospark.inference-opt.collective-causality", __file__, "collective_diagnostics")
    groups = defaultdict(list)
    for row in records:
        for field in ("sequence", "group", "rank", "bytes", "clock_domain", "ready_us"):
            if field not in row:
                raise ValueError(f"collective field missing: {field}")
        groups[(tuple(row["group"]), row["sequence"])].append(row)
    output = []
    for (membership, sequence), rows in groups.items():
        expected = set(membership)
        ranks = [r["rank"] for r in rows]
        complete = expected <= set(range(world_size)) and set(ranks) == expected and len(ranks) == len(expected)
        result = {
            "sequence": sequence,
            "group": list(membership),
            "complete": complete,
            "analyzer": registered,
            "arrival_spread_us": None,
            "calibration_uncertainty_us": None,
            "bandwidth_bytes_s": None,
            "reason": "arrival skew is not wire-transfer time",
        }
        if complete and calibration:
            times, errors = [], []
            for row in rows:
                key = row["clock_domain"]
                if key not in calibration:
                    break
                mapping = calibration[key]
                times.append(finite(row["ready_us"]) + finite(mapping["offset_us"]))
                errors.append(finite(mapping["uncertainty_us"]))
            else:
                result.update(arrival_spread_us=max(times) - min(times), calibration_uncertainty_us=2 * max(errors))
        output.append(result)
    return output
