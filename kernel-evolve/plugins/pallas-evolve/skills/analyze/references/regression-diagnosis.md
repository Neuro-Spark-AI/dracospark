# Diagnose a regression before proposing a fix

Use this when a candidate loses to the current winner, especially when it wins
standalone but loses in its real caller. The process is model-independent.
Do not turn a regression into a permanent rejection of an optimization method.

## Evidence ladder

1. **Observed:** comparable latency samples show a regression in a named scope.
2. **Localized:** paired traces show increased work/waits in a named region.
3. **Hypothesized:** source and traces suggest a mechanism and a discriminating test.
4. **Supported by intervention:** changing that mechanism, with other relevant
   conditions held fixed, moves the predicted trace feature and unprofiled
   target metric while preserving correctness. Cite the intervention artifacts
   and remaining confounders; a threshold/counter change alone is not this step.

Collect control and candidate code revisions, executed routes, exact local
shapes, masks/segments, dtypes and layouts, identical input identity, backend and
compiler versions, rank mapping, concurrency, timing scope and synchronization.
Compare standalone, real-caller and complete-request timings separately. For a
shape sweep, make one paired comparison per shape/context; aggregate only with
explicit workload weights and per-region regression limits.

If the historical experiment supplies only aggregate numbers, retain them as
historical evidence. Do not manufacture repeated samples, traces or current
runtime compatibility. Search its source history for later fixes before
reimplementing the failed candidate.

## Offline paired-evidence comparator

From a Dracospark checkout, with the package installed or `PYTHONPATH=src`:

```bash
cd kernel-evolve
PYTHONPATH=src python3 -m kernel_evolve.regression control.json candidate.json \
  --noise-pct 1 --out regression.json
```

Choose the noise threshold from repeatability of the intended experiment, not
from the example value. It is a practical median-delta threshold, not a
statistical confidence test. Exit `0` means the latency comparison was possible
(it can still report regression or missing traces); exit `2` means input/context
is invalid or insufficient. Neither means production approval.

Each sidecar has this format (values below are **synthetic**, not measurements):

```json
{
  "revision": "exact-kernel-and-integration-revision",
  "context": {
    "workload": {
      "shapes": {"q": [5, 8192, 128], "k": [5, 8192, 128], "v": [5, 8192, 128]},
      "dtypes": {"input": "bf16", "accumulator": "fp32"},
      "layout": "HSD-contiguous",
      "semantics": {"mask": "dense", "dropout": 0, "rope": "already-applied"},
      "input_id": "input-artifact-hash",
      "concurrency": 1
    },
    "hardware": {"device": "target-model", "topology": "rank-map-id"},
    "runtime": {"framework": "version", "compiler": "version", "backend": "revision"},
    "timing": {"scope": "attention-caller", "clock": "host-monotonic",
               "synchronized": true, "steady_state": true, "profiled": false}
  },
  "samples_ms": [10.0, 10.1, 9.9],
  "correctness_passed": true,
  "trace": {
    "path": "trace_events.json",
    "window_us": [100000, 130000],
    "repetitions": 3,
    "selectors": {
      "custom_call": {"pid": 1, "tid": 2, "name_regex": "^call_custom_kernel$"},
      "wait": {"pid": 3, "name_regex": "^SyncWait"},
      "kernel": {"pid": 3, "name_regex": "^pallas"}
    }
  }
}
```

Derive selectors from the **actual** trace labels and host/device/thread metadata;
the example names are not a guaranteed profiler schema for any framework. Pin comparable
caller scopes, choose complete warmed iteration windows, and record the actual
repetitions. Do not use largest-gap guesses to invent iteration boundaries.
Paths are relative to each sidecar. Accepts the evaluator's event-list (`trace_events.json`)
artifact or Chrome JSON with `traceEvents`; `ts`/`dur` are microseconds and only
complete `ph="X"` events are included. For multi-rank work, inspect each relevant
rank; one trace is not a slice-wide critical-path measurement.

The common context must match exactly; implementation revisions may differ.
At least three finite positive unprofiled samples are needed per side. Metadata
is caller-declared: the tool cannot verify that a producer synchronized or that
the supplied trace belongs to the claimed revision. Preserve raw artifacts and
review these declarations. Profiled samples belong in the trace, not samples_ms.

Missing traces do not erase a valid latency observation, but leave localization
unresolved. Category intervals are clipped and unioned to avoid double-counting
nested/overlapping spans **within** a category, then normalized by repetitions.
Categories themselves can overlap; do not add them together or subtract them
from wall time to invent an "unexplained overhead". Zero matched events can mean
missing coverage or a bad selector. Counts describe matched events, not inferred
device launches. No event label alone proves a queue flush or HBM stall.

## Choose a discriminating experiment

| Increased evidence category | Hypothesis to test, not an automatic conclusion |
| --- | --- |
| `custom_call` / `wait` | Repeated custom-op boundaries may drain or serialize a queue. Hold masks/shapes fixed and compare grouped versus per-range calls, including surrounding ATen work. Inspect actual queue behavior. |
| `layout_copy` | A caller layout, contiguous conversion, padding or intermediate buffer may erase the kernel win. Compare caller HLO and isolate one conversion. A bitcast need not move data. |
| `kernel` | Local shape/tile choices may change utilization or spilling. Compare HLO/LLO and replay the real input contract before retuning one parameter. |
| `collective` | Rank mapping, payload or overlap may have changed. Hold the distributed plan fixed and test communication separately. |
| `compile` | New shape specializations or cache misses may pollute steady samples. Investigate cold start and warmed execution separately. |

For each experiment, record prediction, one changed variable, fixed conditions,
correctness/coverage result, trace change, unprofiled caller/request outcome,
and conclusion (`supported`, `contradicted`, or `unresolved`). The comparator
deliberately leaves `root_cause=unproven` and `promotion=not_evaluated`; reviewed
intervention evidence, not an automatic heuristic, advances the diagnosis.

Keep research selection separate from runtime promotion. Maintain the incumbent
fallback while investigating; only deploy a repaired variant inside its tested
contract after operator, application, memory, compile and target-metric gates
pass. Do not loosen tolerance, drop keys/frames or silently disable semantics to
make a candidate win. Hardware runs and live changes still require the user's
applicable authorization; the offline comparator launches neither.
