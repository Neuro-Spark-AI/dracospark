from inference_opt.compare import compare, summarize


def protocol():
    return {
        "schedule": [["baseline", "candidate"], ["candidate", "baseline"]] * 4,
        "request_seed": 42,
        "statistics": {
            "rule": "paired_exact_sign_practical_gain",
            "minimum_blocks": 8,
            "practical_gain_fraction": 0.02,
            "confidence": 0.95,
        },
    }


def records():
    return [
        {
            "mode": "benchmark",
            "block": block,
            "arm": arm,
            "status": "completed",
            "elapsed_s": seconds,
            "device_gate": True,
            "seed": 42,
            "validation": {"passed": True},
        }
        for block in range(8)
        for arm, seconds in (("baseline", 10), ("candidate", 8))
    ]


def test_numerical_failure_cannot_be_overridden_by_speed():
    verdict = compare(protocol(), records(), {"passed": False, "independent": True})
    assert verdict["status"] == "rejected"
    assert any("numerical" in reason for reason in verdict["reasons"])


def test_missing_telemetry_is_not_a_pass():
    samples = records()
    samples[0]["device_gate"] = None
    result = compare(protocol(), samples, {"passed": True, "independent": True})
    assert result["status"] == "rejected"
    assert any("comparability" in reason for reason in result["reasons"])


def test_search_requires_independent_confirmation():
    result = compare(protocol(), records(), {"passed": True, "independent": True})
    assert result["status"] == "requires_confirmation"
    assert result["mean_fractional_change"] == -0.2


def test_failure_not_in_success_latency_but_blocks_verdict():
    samples = records()
    samples[0]["status"] = "failed"
    result = compare(protocol(), samples, {"passed": True, "independent": True})
    assert result["status"] == "rejected"
    for i, row in enumerate(samples):
        row.update(start_monotonic=i * 10, end_monotonic=i * 10 + row["elapsed_s"])
    summary = summarize(samples, 1)
    assert summary["failures"] == 1
    assert summary["validated_completed"] == 15
    assert "completed_per_minute" not in summary


def test_confirmation_rejects_ignored_or_missing_observed_seeds():
    plan = protocol()
    plan["confirmation_block_seeds"] = list(range(50, 58))
    samples = records()
    for row in samples:
        row["seed"] = 50 + row["block"]
    numerical = {"passed": True, "independent": True}
    assert compare(plan, samples, numerical, confirmation=True)["status"] == "eligible_for_review"
    samples[0]["seed"] = 42
    del samples[1]["seed"]
    verdict = compare(plan, samples, numerical, confirmation=True)
    assert verdict["status"] == "rejected"
    assert sum("observed seed" in reason for reason in verdict["reasons"]) == 2
