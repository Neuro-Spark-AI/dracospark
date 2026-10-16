# Record and execute an optimization transfer

Dracospark can reuse a method across models and call sites through the
`pallas-evolve:transfer` skill. The workflow inventories the source implementation,
adapts its guarded code, uses an authorized evaluator or caller harness, and
records target evidence with `python -m kernel_evolve.transfer`. It is **not** a
universal automatic kernel translator or a production dispatcher.

The offline command implements the record and acceptance checks; it does not
execute code from a manifest, reserve hardware, submit jobs, commit, or deploy.
The skill connects it to Dracospark's existing adaptation/evaluation workflow.

## What is reusable?

A method (tiling, fusion, caching, layout changes) can be reused immediately as
an experiment. An implementation/configuration can be selected only inside its
validated contract. Record one target workload region per plan, and use the same
method name/source provenance in records for any number of other models or
regions. Model names never replace shape, semantic or runtime predicates.

The source's faster result does not become the target's result. Kernel, compiled
caller and full-request scopes have separate contexts and measurements. In
particular, eager versus compiled timing is a runtime-context change, not merely
a different enclosing function. Never subtract their timings to infer overhead.

## Minimal runnable example

The checked-in example is **synthetic and unmeasured**. It demonstrates retuning
a head tile from 5 to 7 for 14 local heads, not a hardware-validated configuration.

```bash
cd kernel-evolve
PYTHONPATH=src python -m kernel_evolve.transfer \
  examples/transfers/head-tile-retune.json --out /tmp/head-tile-transfer.json
```

The result is `needs_validation`: the declared guards pass, but there is no
hardware or application evidence. The output must not already exist. Use a new
filename for each assessment, preserving failed and inconclusive trials.

For a real transfer, copy the plan to its authorized project/run directory and
replace every synthetic context and revision. Private kernel records belong in
the private kernel project, not an upstream public engine fork.

## Plan contract (schema version 1)

| Field | Meaning |
| --- | --- |
| `id` | Unique transfer attempt identifier. |
| `method` | Name, description, exact `source_revision`, and `source_context` provenance. Source success is not target acceptance. |
| `candidate` | Exact implementation `revision`, tuned `parameters`, `adaptation` description, and nonempty capability `guards`. |
| `target` | Provenance `name` and named `scopes`; each declares its incumbent revision and complete measurement context. |
| `policy` | `objective_scope`, `required_scopes`, `noise_pct`, and `min_fresh_runs` (at least two; choose before testing). |
| `checks` | Target-bound compilation, operator/structural/application correctness and memory receipts. Missing means unvalidated, not pass. |
| `experiments` | Paired target measurements, independent-process identity and lifecycle receipts. |
| `interventions` | Optional predictions, changed variable, fixed conditions, conclusion and evidence for tested hypotheses. |

Each scope's `context` uses the existing
[regression measurement format](../kernel-evolve/plugins/pallas-evolve/skills/analyze/references/regression-diagnosis.md):
`workload`, `hardware`, `runtime`, `timing`. Record local shapes after sharding,
mask/segment semantics, dtype/rounding, layout, input identity, concurrency,
backend/compiler revisions, topology, graph boundaries and timing protocol.
Context must match exactly in the plan, control and candidate sidecars.

Declare every production-relevant region/scope separately. Required scopes must
include the objective. For a production decision, the objective should be the
real caller or full request and required regions must cover its supported
distribution; a kernel-only objective can only justify kernel research review.
This version evaluates latency, not a multi-objective throughput/memory optimizer.
Memory and application checks remain explicit receipts.

### Capability guards

Each guard selects a key path in one scope's context:

```json
{
  "scope": "kernel",
  "path": ["workload", "shapes", "local_heads"],
  "op": "divisible_by",
  "value": 7,
  "reason": "This implementation has no partial head tile"
}
```

Operations are `equals`, `one_of` and positive-integer `divisible_by`. Boolean
`false` is not numeric zero. Missing/null facts are unknown and fail closed.
No Python expression, model blacklist, shell command or network request runs.
This deliberately small language is not a general compiler constraint solver.
Computed predicates (e.g. VMEM bounds) need reviewed target code and a receipt;
do not invent facts to make a manifest eligible. Also retain executable guards
in the actual kernel/adapter: an offline manifest is not a dispatch safety layer.

### Bind evidence before collecting it

```bash
PYTHONPATH=src python -m kernel_evolve.transfer plan.json --binding
```

The SHA256 binding covers method, candidate, all target contexts and policy.
Changing shapes, semantics, caller/runtime, implementation, parameters, guard or
policy invalidates old experiment/check bindings. Do not relabel old evidence
with the new binding; rerun the affected validation. Fresh-process identities
are producer declarations and must come from actual distinct process launches.

Every local artifact reference has `path` (relative to the plan, or absolute)
and lowercase `sha256`. For example, use `sha256sum caller-control.json` to
obtain the checksum. No URL is fetched. A missing file or checksum mismatch
cannot establish acceptance. Raw measurement JSON is snapshotted into the output
along with resolved paths and hashes; large traces and compiler artifacts remain
external and must be retained separately. The comparator still validates trace
selectors and reports missing/incomplete trace evidence independently of timing.

An experiment has this shape (replace illustrative strings with real identities
and hashes; these are not benchmark results):

```json
{
  "id": "caller-run-01",
  "scope": "caller",
  "fresh_process_id": "recorded-launch-identity",
  "binding": "SHA256 printed before this experiment",
  "control": {"path": "caller-control.json", "sha256": "SHA256 of file"},
  "candidate": {"path": "caller-candidate.json", "sha256": "SHA256 of file"},
  "lifecycle": {
    "passed": true,
    "evidence": {"path": "worker-exit-report.json", "sha256": "SHA256 of file"}
  }
}
```

The sidecar revisions must match the declared incumbent/candidate. At least
three finite positive synchronized, warm, unprofiled samples per arm are needed
for comparison; independent-process repetition is an additional gate. Multiple
A/B pairs from one process count only once. Identical measurement content copied
under new filenames/IDs is rejected as fresh evidence. These checks do not prove
producer honesty or statistical independence.

Lifecycle `passed` is boolean or null. A complete timing file does not prove the
job exited normally; include every relevant rank's exit status and shutdown
acceptance. Failed or unknown lifecycle retains valid latency observations but
blocks readiness. Do not override failed correctness with a favorable median.

Required check names are `compilation`, `operator_correctness`,
`structural_correctness`, `application_correctness`, and `memory`. Each has
`passed` (boolean/null), `binding`, and an `evidence` artifact. Receipts must
explain the actual checks/tolerances and their scope; a hash proves file identity,
not the truth or sufficiency of its contents. Reviewers must verify coverage.

### Diagnostic interventions

Record `hypothesis`, `changed_variable`, `held_fixed`, `prediction`,
`conclusion` (`supported`, `contradicted`, `unresolved`) and a nonempty `evidence`
list. Preserve negative experiments. Source/trace observations and a controlled
intervention are distinct evidence levels. The tool never infers a proven root
cause just because the author recorded a supported hypothesis.

## Decisions and next actions

| Decision | Action |
| --- | --- |
| `ineligible` | Adapt the failing capability or retain the incumbent. Revalidate, do not drop semantics. |
| `needs_contract` | Obtain missing target facts before running the incompatible/unknown candidate. |
| `validation_failed` | Repair compilation/correctness/application/memory failures; preserve failed evidence. |
| `regressed_in_target` | Diagnose with paired traces and one-variable interventions; retain the incumbent. |
| `needs_validation` | Collect missing, stale, invalid, unrepeated or lifecycle-incomplete evidence. |
| `no_demonstrated_gain` | Retain the incumbent; target objective has not improved beyond the practical threshold. |
| `ready_for_review` | Declared predicates/checks and repeated required scopes passed; a human still reviews the bounded region and objective. |

A regression in any required scope blocks readiness. Every required scope needs
enough valid fresh-process runs; the objective must improve beyond the threshold
in every accepted comparison. There is no averaging away a regressed region.
Optional investigative scopes are reported but do not replace required ones.
The noise threshold is practical, not a confidence interval. Every output keeps
`production_promotion=not_performed` and `root_cause=unproven`.

Exit zero means the assessment was recorded, **not** that the transfer passed.
Exit two means the manifest/output operation was invalid. Consumers must inspect
`decision`, `checks`, `scopes` and each experiment's `issues`.

## Scope and validation

This feature supplies executable guard/evidence assessment, immutable records,
and an agent workflow for adapting and evaluating methods across arbitrary
models. It builds on the paired regression comparator and existing TPU evaluator,
without adding autonomous fleet orchestration or changing production recipes.
The [validation guide](regression-validation.md) covers host checks and optional
synthetic TPU fixtures without requiring another project's source or artifacts.
