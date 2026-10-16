import json

import pytest

from inference_opt.analysis import analyze_trace, roofline, union_us


def test_interval_union_does_not_count_overlap_twice():
    assert union_us([(0, 10), (5, 20), (30, 35)]) == 25
    with pytest.raises(ValueError):
        union_us([(2, 1)])


def test_counter_traffic_and_precision_required():
    peaks = {"bf16": 1000}
    assert roofline(100, 1, "fp32", peaks)["compute_peak_flop_s"] is None
    estimated = roofline(
        100,
        1,
        "bf16",
        peaks,
        {"HBM": {"source": "tensor_size", "bytes": 100, "peak_bytes_s": 10, "launch_signature_match": True}},
    )
    assert estimated["roofline_efficiency"] is None
    measured = roofline(
        100,
        1,
        "bf16",
        peaks,
        {"HBM": {"source": "measured_counter", "bytes": 100, "peak_bytes_s": 200, "launch_signature_match": True}},
    )
    assert measured["roof_flop_s"] == 200
    assert measured["roofline_efficiency"] == 0.5
    with pytest.raises(ValueError):
        roofline(float("nan"), 1, "bf16", peaks)


def scoped_trace(tmp_path, step="model.step.0", block="model.block.0"):
    events = [
        {"cat": "user_annotation", "name": step, "ts": 0, "dur": 50, "pid": 1, "tid": 1},
        {"cat": "user_annotation", "name": block, "ts": 1, "dur": 40, "pid": 1, "tid": 1},
        {
            "cat": "cpu_op",
            "name": "aten::mm",
            "ts": 2,
            "dur": 1,
            "pid": 1,
            "tid": 1,
            "args": {"External id": 7, "Input Dims": [[2, 3], [3, 4]], "Input type": ["BFloat16", "BFloat16"]},
        },
    ]
    for correlation, ts in ((8, 3), (9, 4)):
        events.append(
            {
                "cat": "cuda_runtime",
                "name": "hipLaunchKernel",
                "ts": ts,
                "dur": 1,
                "pid": 1,
                "tid": 1,
                "args": {"correlation": correlation},
            }
        )
        # Device executes after the CPU block ended. Map through the launch, not GPU ts.
        events.append(
            {
                "cat": "kernel",
                "name": "gemm",
                "ts": 100,
                "dur": 10,
                "args": {"correlation": correlation, "External id": 7, "stream": correlation},
            }
        )
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events, "distributedInfo": {"rank": 0}}))
    return path


def test_asynchronous_scope_correlation_and_one_flop_count(tmp_path):
    path = scoped_trace(tmp_path)
    result = analyze_trace(path, rank=0, device=0, expected_steps=1, expected_blocks=1, peaks={"bf16": 1000})
    assert result["coverage"]["complete_model"]
    assert result["coverage"]["step_block_ids"] == {"model.step.0": ["model.block.0"]}
    assert result["device_busy_union_us"] == 10
    assert len(result["operators"]) == 1
    assert result["operators"][0]["metrics"]["algorithmic_flops"] == 48
    assert result["operators"][0]["metrics"]["duration_s"] == 10 / 1e6
    assert result["operators"][0]["metrics"]["bound"] == "unestablished"
    with pytest.raises(ValueError, match="rank"):
        analyze_trace(path, rank=3, device=3, expected_steps=1, expected_blocks=1, peaks={})


def test_adapter_declares_scope_prefixes_and_coverage_key(tmp_path):
    path = scoped_trace(tmp_path, "decode.step.0", "decode.layer.0")
    prefixes = {"step": "decode.step.", "block": "decode.layer.", "stage": "decode.stage."}
    default = analyze_trace(path, rank=0, device=0, expected_steps=1, expected_blocks=1, peaks={})
    assert default["coverage"]["complete_model"] is False
    result = analyze_trace(
        path,
        rank=0,
        device=0,
        expected_steps=1,
        expected_blocks=1,
        peaks={},
        prefixes=prefixes,
        coverage_key="all_layers_profiled",
    )
    assert result["coverage"]["all_layers_profiled"] is True
    assert "complete_model" not in result["coverage"]
    assert result["scope_prefixes"] == prefixes
    with pytest.raises(ValueError, match="distinct"):
        analyze_trace(
            path, rank=0, device=0, expected_steps=1, expected_blocks=1, peaks={}, prefixes={"block": "model.step."}
        )
    with pytest.raises(ValueError, match="coverage key"):
        analyze_trace(path, rank=0, device=0, expected_steps=1, expected_blocks=1, peaks={}, coverage_key="step_count")
