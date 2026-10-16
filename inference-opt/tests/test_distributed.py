import pytest

from inference_opt.distributed import collective_diagnostics


def rows():
    return [
        {
            "sequence": 7,
            "group": [0, 1, 2, 3],
            "rank": i,
            "bytes": 1024,
            "clock_domain": f"rank{i}",
            "ready_us": 100 + (30 if i == 3 else 0),
        }
        for i in range(4)
    ]


def test_u4_missing_rank_and_missing_calibration_remain_unknown():
    result = collective_diagnostics(rows()[:3], 4)
    assert not result[0]["complete"]
    assert result[0]["arrival_spread_us"] is None
    result = collective_diagnostics(rows(), 4)
    assert result[0]["complete"]
    assert result[0]["arrival_spread_us"] is None


def test_known_rank_delay_detected_without_inventing_bandwidth():
    calibration = {f"rank{i}": {"offset_us": 0, "uncertainty_us": 2} for i in range(4)}
    result = collective_diagnostics(rows(), 4, calibration)[0]
    assert result["arrival_spread_us"] == 30
    assert result["calibration_uncertainty_us"] == 4
    assert result["bandwidth_bytes_s"] is None


def test_ambiguous_collective_records_fail():
    with pytest.raises(ValueError, match="clock_domain"):
        collective_diagnostics([{"sequence": 1, "group": [0], "rank": 0, "bytes": 100}], 1)
