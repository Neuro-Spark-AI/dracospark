"""Exact paired sign-test gate; ties are conservative non-wins, not discarded."""

import math
import statistics

RULE = "paired_exact_sign_practical_gain"


def validate_policy(policy):
    if policy.get("rule") != RULE:
        raise ValueError("unregistered statistical rule; legacy bootstrap is exploratory only")
    if type(policy.get("minimum_blocks")) is not int or policy["minimum_blocks"] < 8:
        raise ValueError("at least eight paired blocks required")
    gain = policy.get("practical_gain_fraction")
    confidence = policy.get("confidence")
    if type(gain) not in (int, float) or not math.isfinite(gain) or not 0.02 <= gain < 1:
        raise ValueError("practical gain must be at least two percent")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0.95 <= confidence < 1:
        raise ValueError("confidence must be at least 95 percent")


def exact_tail(wins, total):
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total


def assessment(changes, policy):
    validate_policy(policy)
    threshold = -policy["practical_gain_fraction"]
    wins = sum(value < threshold for value in changes)
    alpha = (1 - policy["confidence"]) / 2
    pvalue = exact_tail(wins, len(changes)) if changes else None
    mean = statistics.mean(changes) if changes else None
    return {
        "wins": wins,
        "pairs": len(changes),
        "one_sided_pvalue": pvalue,
        "alpha": alpha,
        "practical_gain_fraction": policy["practical_gain_fraction"],
        "passed": len(changes) >= policy["minimum_blocks"] and pvalue <= alpha and mean < threshold,
        "estimand": "median/typical paired fractional change; mean is an additional descriptive guard",
    }
