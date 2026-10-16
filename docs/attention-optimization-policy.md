# Model-independent attention optimization and regression diagnosis

Use one investigation and promotion process across models, not one universal
kernel or tile configuration. Model names identify the provenance of a workload;
they are not eligibility tests or explanations of a performance difference.
This policy applies to inference attention in diffusion, prefill, decode and
cross-attention. It does not claim the existing examples support all of them.

## Describe the operator, then select an implementation

Record the actual call after sharding/redistribution, not checkpoint defaults:

| Contract | Facts needed |
| --- | --- |
| Semantics | Dense/causal/window/block/packed visibility, padding and segment bounds, Q-to-KV head mapping (MHA/GQA/MQA), scaling, bias, soft caps and dropout. Record any special representation rather than pretending it is ordinary Q/K/V. |
| Geometry and representation | Batch, Q/KV lengths, local Q/KV heads, QK/value dimensions, dtype/accumulation/quantization, layout/strides, cache organization and relevant length buckets. |
| Positional transforms | Whether RoPE is already applied; if fused, pairing layout, rotated extent, positions, frequencies and rounding order. An attention kernel taking already-rotated Q/K need not implement RoPE itself. |
| Execution context | Backend/compiler/device revisions, hardware and memory limits, rank mapping, parallel degrees, caller graph boundaries, workload stage, concurrency and input identity. |

Each implementation declares its own capability predicates. A head-count
divisibility requirement belongs to a particular kernel configuration, not to
every attention kernel. A failed predicate means retune or adapt that implementation,
or use the correct fallback; it does not mean the optimization method cannot
apply to a model. Unknown requirements fail closed. Shape-compatible is not
synonymous with numerically correct, compilable, or fast.

Search legal tile sizes, fusion boundaries, layouts and scheduling choices
against this contract. Never improve speed by silently changing attention
visibility, positional semantics or numerical acceptance. Share successful
methods and search strategies across models; validate configurations on the new
call site. Keep different winners for different validated workload regions.

## Regression is an investigation input, not the end of the search

1. Reproduce the control/candidate comparison with identical inputs and context.
   Record the executed route and separate standalone, caller and complete-request
   timing. Do not subtract unrelated measurements and label the difference
   "dispatch overhead". Report cold compilation separately from warmed execution.
2. Capture matched caller traces and compiler artifacts. Localize whether extra
   time appears in kernel work, layout/padding copies, custom-op boundaries,
   synchronization, collectives or recompilation. Preserve event labels, selected
   devices/threads, windows and repetitions. Overlapping trace spans are not an
   additive latency budget, and profiler measurements are not acceptance timings.
   Verify host/device clock alignment and event coverage. When the caller has a
   known event-count contract, declare `expected_events_per_repetition` on the
   selector; the comparator rejects missing or clipped declared events as trace
   evidence while retaining valid unprofiled timings. Counts alone are not a
   synchronization-time regression; more short waits can take less total time.
3. State observations separately from hypotheses. Head counts and RoPE layouts
   explain compatibility boundaries, not a regression unless an experiment links
   them to the slowdown. Static instruction counts, average utilization and a
   `SyncWait` label are not causal proof.
4. Test a discriminating change: for example, group custom calls while preserving
   shapes and masks; remove one layout conversion; or retune a tile while holding
   the caller boundary fixed. Revalidate correctness, inspect the predicted trace
   change, then rerun unprofiled target-workload A/B. Record failed hypotheses too.
5. Retain the incumbent default until a candidate passes the promotion gates
   below. A failed candidate may seed the next experiment; do not blacklist its
   whole kernel family or model.

The [Dracospark regression workflow and input format](../kernel-evolve/plugins/pallas-evolve/skills/analyze/references/regression-diagnosis.md)
connect these steps to the offline `kernel_evolve.regression` comparator and the
analysis skill. The comparator localizes evidence and proposes experiments; it
does not run jobs, establish a root cause, or change runtime dispatch.

## Promotion and reuse

Promote a configuration only within its recorded workload/backend region after:

- Capability guards and target-hardware compilation pass, including supported
  tails/fallbacks and bounded memory/compilation behavior.
- Operator numerical **and structural** checks pass: include mask/segment
  coverage, excluded-key perturbations, partial tiles and fusion semantics where
  relevant. A loose tolerance must not hide a missing head, frame or query range.
- The target application's declared output checks pass. Token, video and audio
  acceptance are application-specific; do not generalize one model's tolerance.
- Repeated, synchronized, unprofiled A/B against the current winner improves the
  declared objective beyond a predeclared practical noise threshold. Include
  representative concurrency, tails and production shape frequencies; do not
  hide important regressions inside a weighted average. Report latency,
  throughput, memory and cold-start tradeoffs separately.

Store method, implementation revision, capability predicates, tuned parameters,
tested region, reference/tolerance, raw samples and artifact paths, hypotheses,
ablation results and decision. Distinguish `ineligible`, `untested`,
`regressed-in-this-context`, and `accepted-in-this-region`. Invalidate or
revalidate the relevant evidence when the compiler, backend, hardware, semantic
contract or caller changes. A new model may reuse the method and legal tuning
space immediately; its production promotion still needs target validation.

The [structured transfer workflow](optimization-transfer.md) implements guarded
target records, per-scope comparisons, stale-evidence invalidation and readiness
checks. Its highest automated decision is `ready_for_review`, never deployment
or an assertion that the method is universally accepted.
