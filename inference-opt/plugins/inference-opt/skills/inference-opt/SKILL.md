---
name: inference-opt
description: Run or assess an evidence-bound model or framework inference optimization cycle with Dracospark. Use for benchmark, layered profile analysis, a registered candidate, paired confirmation and review through a model/runtime adapter. Use pallas-evolve for existing standalone Pallas TPU kernel evolution.
---

# Inference optimization

Use the executable `inference-opt` harness for measurements and acceptance.
Read the [package contract and commands](references/commands.md) before composing
a plan. Skill discovery does not install the Python package.

An adapter supplies the model/runtime semantics: it inspects runtime identity,
configures arms, runs requests, validates outputs and reports profile analyses,
including the boolean coverage field the plan names in `profile_coverage_key`
and any adapter-owned analyzers registered in its own registry file. The
synthetic CPU fixture in the Dracospark repository
(`inference-opt/examples/cpu_fixture/`, driven by
`inference-opt/examples/run_cpu_fixture.py`) shows the protocol; it is not part
of the installed package. The user supplies the adapter, plan and agent command
files for a real workload.

Identify the requested workload, runtime and optimization scope. Reuse the
user's existing authorization and reservation; perform only the host actions
covered by that task. Inspect the actual adapter identity before freezing a
plan. Keep model-specific patches in the adapter's own repository.

Freeze sources/inputs/evaluator, A/B treatment fields, paired seeds, fresh
confirmation seeds, quality and statistical rules. Include the installed core
and registry from `inference-opt manifest` in the closure. Ordinary benchmarks
and diagnostic/profiling passes are distinct. Keep failed and outlier requests.

Use layered profiles to propose a hypothesis with a falsifier. Run the bounded
cycle or individual stages in the package guide. The agent may propose only a
registered candidate; a new implementation needs adapter integration and a new
plan. For each round report the changing bottleneck, measured gain or null
result, correctness, uncertainty and next discriminating experiment. Missing
traffic counters cannot establish a roofline bound; per-GPU inclusive scopes
cannot be summed into request latency.

On timeout, reconcile the original request key instead of resubmitting. Failed
baseline restoration blocks readiness. Review is independent and read-only;
favorable prose never overrides the executable gates. A qualified experiment
does not authorize production rollout. Subsequent rounds get new plans/IDs;
preserve sealed evidence and rejected candidates.
