# Inference optimization harness

An installable, standard-library core for a bounded agent-driven
benchmark → profile/analyze → proposal → paired evaluation → independent
confirmation → reprofile → review cycle. An adapter supplies the model/runtime
semantics, profile coverage keys and any model-specific analyzers; see the
synthetic CPU fixture in `examples/`. This package does not reserve nodes,
start services, rewrite arbitrary kernels or deploy production.

## Concepts

A cycle needs three things you provide:

- **Plan** (`plan.json`): the frozen experiment — adapter command, the two arms
  (`baseline`, `candidate`) and the fields they differ in, warmups, paired seed
  schedules, statistical rule and the sealed source closure.
  `examples/run_cpu_fixture.py` builds a complete plan; start from it.
- **Adapter**: a script that speaks JSON on stdin/stdout (`{"action": ...}` in,
  `{"protocol": 1, "status": "ok", ...}` out) and owns everything model- or
  runtime-specific. The harness calls these actions:

  | Action | Adapter returns |
  |---|---|
  | `inspect` | runtime `identity` (hashed into the plan) |
  | `configure` | switches to an arm; reports `quiescent` |
  | `request` | one timed request `record` (elapsed, monotonic start/end, device gate) |
  | `validate` | per-request numerical/semantic `validations` |
  | `compare_outputs` | baseline-vs-candidate `numerical` agreement |
  | `analyze` | profile `analysis` with a `coverage` object holding the plan's coverage key |
  | `reconcile` | outcome of a request whose submission state was lost (after a timeout) |

  `examples/cpu_fixture/adapter.py` is a minimal (synthetic) implementation.
- **Agents**: optimizer and reviewer commands, each a JSON argv array file. They
  read a JSON packet on stdin and print a JSON response; see
  `examples/cpu_fixture/agent.py`, or use the Codex bridge below.

Each stage writes a sealed, content-addressed bundle (`baseline`, `search`,
`confirmation`); `verify` re-checks it and `report` renders Markdown/HTML from it.

## Install and run

```bash
python -m pip install ./inference-opt
inference-opt --help
inference-opt manifest > core-closure.json

# Synthetic CPU protocol exercise: no GPU, model API or measured speedup.
python inference-opt/examples/run_cpu_fixture.py --output runs/new-fixture-attempt
```

The fixture tests the installed adapter/agent subprocess protocol, search,
held-out seed confirmation, candidate profiling, independent review, sealed
evidence and baseline restoration. Its durations, numerical pass and device
qualification are invented; `candidate_qualified` in a fixture is not hardware
or application acceptance. A fixture output path must be new.

For an authorized prepared engine with a frozen plan:

```bash
inference-opt loop --plan plan.json \
  --optimizer-command optimizer-argv.json --reviewer-command reviewer-argv.json \
  --output runs/new-cycle

# The same stages are separately resumable:
inference-opt baseline --plan plan.json --output runs/baseline
inference-opt agent-packet runs/baseline --role optimize
inference-opt run-agent runs/baseline --role optimize \
  --command optimizer-argv.json --output runs/proposal
inference-opt campaign --baseline runs/baseline \
  --proposal runs/proposal/response.json --output runs/search
inference-opt campaign --baseline runs/baseline \
  --proposal runs/proposal/response.json --confirm-search runs/search \
  --output runs/confirmation
inference-opt run-agent runs/confirmation --role review \
  --command reviewer-argv.json --output runs/review
inference-opt review runs/confirmation --review runs/review/response.json \
  --baseline runs/baseline --search runs/search
inference-opt verify runs/confirmation
inference-opt report runs/confirmation --output reports/confirmation
```

Agent commands are explicit JSON argv arrays, with JSON stdin/stdout. The
optimizer receives the immutable evidence binding and registered treatment;
the reviewer receives the actual measured outcome. No agent can override
deterministic rejection. An optional bridge is
`["python", "-m", "inference_opt.codex_role"]`; it retains the operator's
configured model and uses read-only ephemeral CLI sessions. Other compatible
agents can use the same protocol. The optional CLI bridge supports scalar
treatment fields (flags, sizes, modes); nested configurations can use a
different JSON agent wrapper. Ordinary CLI use does not require a Skill.

## Plan contract

The runnable CPU example constructs a complete protocol-1 plan. Replace all
fixture values before actual use. A real plan freezes:

- Explicit `adapter_entrypoint`/`adapter_source_root`, script argv, interpreter/
  runtime identity, world size (1/2/4/8) and concurrency 1. The complete adapter Python
  tree and installed evaluator/registry must be sealed. Module `-m`, shell
  `-c` and undeclared entrypoints are refused by this execution contract.
- Core module/registry closure from `manifest`, adapter/evaluator/agent sources
  and command files, request/input files, plus their SHA256 values. Include the
  chosen Python executable/environment identity. Model assets are verified by
  the lifecycle/adapter and recorded in runtime identity; do not duplicate
  hundreds of GiB of weights into every code closure.
- Exactly two arms, their declared differing `treatment_fields`, workload
  inputs, per-arm warmups, baseline repetitions, request seed, paired schedule
  and independent confirmation schedule/seeds.
- At least 2% practical gain, confidence >=95%, at least eight paired blocks per
  stage, and
  `profile_coverage_key` (default `complete_model`; adapters declare the
  boolean complete-scope field their `analyze` response reports). Missing
  coverage does not pass.

`source_closure(executable, adapter_source_root, *extra)` in
`inference_opt.evidence` collects the required interpreter/evaluator/adapter
closure. Include agent command/config/input files as extras. Preregister an
explicit `campaign_id` and absolute operator-owned `attempt_ledger` outside
bundles. Each baseline/search/confirmation phase is claimed once using exclusive
creation; failed/aborted attempts cannot retry under another output directory.
New studies require explicit preregistration, not silent optional stopping.

Every pair has the same seed in both arms. `block_seeds` and
`confirmation_block_seeds` must match their schedules and use disjoint search/confirmation seed sets. Multi-seed confirmation
requires a separate predeclared seed schedule. Numerical/semantic acceptance
belongs to the adapter's independently versioned, sealed evaluator; do not
change its policy after observing results.

The current agent selects a single registered candidate. It does not synthesize
an arbitrary source patch and automatically compile/deploy it. A new candidate
type needs adapter support, source guards and validation, then a new locked
plan. Multiple optimization rounds use new attempt IDs and reprofiling, rather
than modifying a sealed successful baseline.

## Evidence and analysis

Completed request latency excludes profiling/decoder work. Recovered results
never count as fresh performance samples. POSIX adapter execution uses a private
process group. Timeout terminates that group, bounds pipe draining/reaping, records
partial output and cleanup receipts, then preserves `submission_unknown`.
Automatic baseline restoration and subsequent adapter operations are skipped;
use `reconcile` with the original request key and prove terminal/quiescent state
before restoring or starting another attempt. Killing a local group is not proof
that a detached worker or a remotely submitted GPU job has stopped. Arm
switches need quiescence. Failed baseline restoration blocks qualification.

Bundles contain locked plans, content-addressed code/input snapshots,
operation journals, raw copied artifacts, validated records, profile analyses,
statistics and outcomes. Verification rejects changed, missing, extra or
symlinked files. Report generation reads the sealed bundle and writes elsewhere.
Acceptance uses the exact one-sided binomial sign tail for paired improvements
beyond the practical threshold, at alpha=(1-confidence)/2. Ties count as
non-wins. It infers typical/median paired gain; the mean remains descriptive
and must also improve. This avoids the small-n percentile-bootstrap gate;
legacy intervals stay historical/exploratory. Independence/representative
sampling are still necessary. See [NIST sign test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm).

Final review requires explicit trusted baseline/search roots and external ledger
bindings. It recomputes statistical, warmup, profile, numerical and rollback
qualification; `outcome.json` and reviewer prose are not acceptance authority.
The operator-owned ledger/baseline and registered producer are trust boundaries;
this is not cryptographic proof of measurement honesty. Agents are checked before
and after execution for source, evidence and ledger drift. Generic CLI commands
still need caller-managed OS read-only isolation: prompt text is not a sandbox.
Codex's explicit read-only sandbox prevents writes but does not isolate reads.

Fresh independent confirmation is required before `eligible_for_review`; even
an accepted review records `production_adoption=false`.

The packaged analyzer registry records the built-in analyzers' source/symbol
identity. Adapter-owned analyzers are registered in a separate, user-supplied
registry file with the same protocol-1 schema and passed to
`inference_opt.registry.provenance(..., source_repo=..., source_path=...,
registry_path=...)`; seal that file in the plan closure. Module and registry
hashes travel with results. Kineto attribution uses launch correlation and a
union of device intervals per GPU. The packaged trace analyzer groups
annotations by adapter-supplied step/block/stage prefixes (defaults
`model.step.`, `model.block.`, `model.stage.`) and reports the plan's coverage
key; model families with other structure provide their own analyzer through
the adapter. Inclusive scope durations cannot be
summed into a critical path. Rated compute fractions do not prove a bound;
roofline traffic needs matching measured counters. Unknown metrics stay
unavailable. Collective issue spread is not wire bandwidth.

## Integration and validation

To integrate a model, write an adapter that speaks the JSON stdin/stdout
protocol shown in `examples/cpu_fixture/adapter.py`, seal its source tree and
evaluator in the plan, and declare its coverage key. Preserve historical
bundles and analyzer keys; a new package/source identity requires new plans and
fresh hardware replay. CPU fixture success and measured hardware speedups are
separate claims.

```bash
python -m pip install -e './inference-opt[dev]'
python -m pytest -q inference-opt/tests
ruff check inference-opt
ruff format --check inference-opt
python -m build --wheel --outdir dist inference-opt
```

### Roofline distance and useful optimization priority

`report` exports `opportunities.json` alongside Markdown/HTML, with a registered,
source-hashed advisor. It separates theoretical compute-peak fraction, measured
per-call hierarchical counter roofs, and an empirical operator reference. Operator
useful FLOPs/s is **not** full-model MFU or the hardware MFMA busy counter. Missing
traffic, unknown precision/partition peaks, incomplete FLOPs and missing critical
path remain unavailable; the report never substitutes a card name or a utilization
percentage for this evidence.

A profile may contain optional `performance_envelopes`, each with an exact
`operator`, recorded `shape`, `precision` and positive `target_flop_s`. These are
**scenarios by default**, not measured ceilings and not adoption criteria. An
adapter should copy these targets from its frozen configuration and must not invent a
calibrated hardware context. For example, an attention target of 60% of a supplied
dense peak, with achieved 55%, implies `1 - 55/60 = 8.33%` local time saving at
fixed work. If separately proven to occupy 40% of an otherwise unchanged serial
critical path, this would imply 3.33% E2E latency reduction (1.0345x speedup).
Neither a 40% timeline share nor eight rank-local durations establish that path.
There is no built-in vendor- or device-specific ceiling.

An optional envelope `calibration: {artifact, sha256}` can qualify an empirical
reference only when the JSON artifact is inside the sealed evidence manifest and
matches the observed rank `performance_context` exactly: `gpu`, `partition`,
`backend`, `layout`, `clock_policy`, `power_policy`, plus the operator, shape and
precision. The calibration contains this complete `context`,
`flops_definition: "algorithmic useful work"`, and at least three positive
`samples_flop_s`; the target must equal their median. This median is a demonstrated
reference, **not a proof of the maximum achievable rate**. Missing/mismatched
context remains a scenario; broken artifact hashes/manifests fail closed. A
reference below the observation or above the supplied theoretical peak is
inconsistent and supplies no rankable saving. References need an equivalent
mask/sequence distribution, FLOPs convention and timing method: include these in
`layout` or additional context fields. Capture artifacts through the adapter's
artifact contract; never add them to an already sealed bundle.

The advisor ranks calibrated local savings before hypothetical savings, separately
from duration. Calls can nest or overlap, so these are non-additive estimates, not
E2E savings. Near a matched reference (the explicit heuristic is 90% or more),
more tile/LDS/register search may offer little return even for the largest kernel.
A kernel can remain compute-bound while being efficient. Test work reduction,
padding elimination, fusion and exposed communication/overlap opportunities;
below a reference, collect matched memory/occupancy/spill evidence before choosing
a change. Each recommendation is a hypothesis: ordinary benchmark → separate
counters → controlled optimization → ordinary benchmark, correctness, and held-out
confirmation. An improvement must survive the existing deterministic acceptance
policy; the advisor cannot authorize promotion.

Methodology references: [Roofline model (Williams, Waterman, Patterson, CACM 2009)](https://doi.org/10.1145/1498765.1498785),
[AMD empirical hierarchical roofline](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.2.2/how-to/analyze/cli.html)
and [AMD HIP performance/roofline](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html).
