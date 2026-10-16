# Validate regression diagnostics

Validate the offline evidence checks on a CPU first. Optional TPU fixtures test
the capture-to-analysis path; they do not establish production performance or
the cause of a model regression.

## Host checks

From a checkout with the development dependencies installed:

```bash
cd kernel-evolve
PYTHONPATH=src python -m pytest tests/test_transfer.py tests/test_regression.py \
  tests/test_validation_capture.py -q
```

These tests cover capability guards, incompatible contexts, stale bindings,
missing or incomplete evidence, trace coverage, and per-scope readiness gates.
They do not run TPU lowering or establish hardware numerical correctness.

## Optional TPU capture

Use an authorized idle TPU slice with compatible JAX/JAXlib, libtpu, NumPy and
XProf installed and distributed initialization configured. From the checkout
root on every worker, launch the following concurrently with identical arguments
and a fresh output directory:

```bash
JAX_PLATFORMS=tpu OPENBLAS_NUM_THREADS=4 \
  python kernel-evolve/scripts/validate_regression_tpu.py \
  --out attention-run1 --workload attention --heads 5 --dim 128 --size 1024
```

Only process zero benchmarks one local chip; other workers join synchronization.
Use a suitable job timeout and record every worker's exit status. The script
does not install packages, reserve hardware, or launch remote workers. Matmul is
also supported with `--workload matmul`.

The fixtures compare row/query tiles of 128 and 16, then restore the original
tile. A tile change is not guaranteed to regress on every device or shape.
Both implementations must pass the fixture's reference checks before timing.
Compilation and warmup are separate from synchronized, unprofiled A/A, A/B and
B/A samples; profiler captures are collected afterward. Attention is dense,
noncausal MHA without padding, bias, dropout or fused RoPE.

Analyze the generated sidecars on any machine with Dracospark; no TPU is required
for this command, run from the checkout root:

```bash
PYTHONPATH=kernel-evolve/src python -m kernel_evolve.regression \
  attention-run1/ab-0-with-trace.json attention-run1/ab-1-with-trace.json \
  --noise-pct 5 --out attention-run1/report.json
```

Repeat with `aa` and `ba` sidecars and fresh processes/output directories. Choose
the practical noise threshold before inspecting candidate results. Keep shapes,
runtime fingerprints and measurement scopes separate rather than pooling runs.

## Interpretation and limits

The capture script checks fixture-specific device tracks and complete operation
windows. Other callers need their own inspected selectors and coverage contract;
host annotation timestamps alone may clip device events. Missing trace coverage
must not be treated as evidence of absent work. Event counts do not establish
launch counts, and overlapping category spans are not an additive latency budget.

Use traces to localize differences and controlled interventions to test a cause.
Use repeated unprofiled target measurements for acceptance. These synthetic,
single-chip fixtures do not validate distributed attention, an application output,
or a production optimization. Transfer readiness still requires the target-bound
checks described in the [transfer workflow](optimization-transfer.md).
