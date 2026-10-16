---
name: transfer
description: Use when migrating or reusing a kernel optimization method across models, shapes, runtimes or callers; adapt guarded implementations and record target validation. Not for data migration or automatic production deployment.
---

# Transfer an optimization with target evidence

Use the existing think/submit/analyze loop to adapt and evaluate an implementation,
and the offline `kernel_evolve.transfer` command to preserve the scoped decision.
Read the transfer contract, `docs/optimization-transfer.md` at the Dracospark
repository root ([in a checkout](../../../../../docs/optimization-transfer.md);
an installed plugin does not include it, so use a checkout or
[the published copy](https://github.com/Neuro-Spark-AI/dracospark/blob/main/docs/optimization-transfer.md)),
before constructing a plan. A synthetic example plan is
`kernel-evolve/examples/transfers/head-tile-retune.json`. The command assesses local artifacts; it does not launch
hardware, translate kernels or deploy a winner by itself.

## Establish source and target

Identify the reusable method separately from its model-specific implementation
and constants. Inspect source/callers, exact revisions and previous successful
and rejected experiments. Record why the method is worth trying; never reuse a
source-model latency as a target result.

Define the actual target call after sharding: shapes, masks/segments, dtype and
rounding, layout, positional transforms where fused, runtime/compiler, topology,
graph boundaries and workload objective. Make one plan per region, with separate
kernel, compiled caller and request scopes where applicable. Model names are
provenance, not admission rules. Eager and compiled paths are different contexts.

Derive capability guards from the implementation. Unknown facts fail closed.
When a guard fails, adapt the implementation/parameters while preserving semantics,
or keep its existing fallback. A guard failure does not reject the whole method.
Retain executable guards in the adapted code, outside mutable regions where
practical. The offline record cannot enforce runtime dispatch.

## Adapt and evaluate

Create the plan in the authorized kernel project's run directory; keep private
code and results out of public engine forks. Record the candidate revision,
parameters, changed variable and source provenance. Declare required scopes,
practical threshold and fresh-process repetitions before measuring.

Run the command from `kernel-evolve/` with the package installed or
`PYTHONPATH=src`. Use `python -m kernel_evolve.transfer plan.json --binding` to
bind receipts to this exact target. Implement the requested adaptation, then use the existing
authorized evaluator for kernel checks and a matched harness for caller/request
checks. Do not present the standalone evaluator's fitness as caller acceptance.
If hardware access or the application oracle is missing, save an unvalidated
record and identify the missing evidence. Do not invent samples or relax checks.

Only run hardware or external mutations within the user's authorization. Read
the actual evaluator's launch/cleanup instructions (for Pallas kernels,
`pallas-evolve:submit` covers the SSH TPU-VM path); do not assume a TPU
topology, queue, credential or container from this skill. A request to record historical
evidence alone does not authorize new jobs, commits, issue comments or deployment.

For each independent run, preserve synchronized warm unprofiled samples,
correctness/coverage results, per-rank lifecycle/exit receipts, and separate raw
traces/compiler artifacts. Generate checksum-pinned artifact references. If the
candidate regresses or its kernel win disappears in the caller, use
[regression diagnosis](../analyze/references/regression-diagnosis.md); test one
discriminating intervention and record contradicted hypotheses too.

## Record and reuse

Run `python -m kernel_evolve.transfer plan.json --out NEW_RECORD.json` after each
attempt. Never overwrite earlier records. Inspect the decision and missing/stale
checks, not only the command's exit code. A timing-complete job with failed or
unknown teardown must not become ready for review.

Link the record from the project's learnings when useful. State applicability,
source and target contexts, observations, uncertainty and next intervention;
do not fill a "root cause" field with a guess. Reuse the method in other target
plans, but regenerate evidence when its bound contract or runtime changes.

`ready_for_review` is not production approval. Preserve the incumbent until the
declared application objective and supported regions are reviewed and deployment
is authorized. The offline command never changes production selection.
