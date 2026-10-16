---
name: reflect
description: Use when recording batch kernel optimization learnings to AGENT.md — iterates over N variant results, extracts failure patterns and successful optimizations, writes local round summaries with optional GitHub tracking
---

# Record Batch Optimization Learnings

After each batch evolution round, iterate over all variant results, extract learnings, and record them to the resolved local learnings file. Save a local round summary; mirror it to GitHub only when explicitly enabled.

## Context

Invoked by `pallas-evolve:start` after `pallas-evolve:analyze`, or standalone. Expects the batch round directory `iteration_{N}/variants/*/eval_result.json`, along with `iteration_{N}/batch_analysis.md` and `iteration_{N}/selection.md`.

## Procedure

### Step 1: Read iteration data

Read:
- `iteration_{N}/variants/*/eval_result.json` — evaluation results for ALL variants in this round
- `iteration_{N}/batch_analysis.md` — comparative analysis across variants
- `iteration_{N}/selection.md` — lineage selection decisions (promotions, prunings)
- `iteration_{N}/strategy.md` — the optimization directions attempted this round
- the validated session config and actual run directory — resolve the learnings file with `config.session.resolved_agent_md_path(run_dir)`, default `{run_dir}/AGENT.md`
- the resolved learnings file — existing learnings

When invoked standalone, validate the session config before GitHub actions. With no config, record locally in the run directory and treat GitHub tracking as disabled. Never infer authorization from an Issue number, repository URL or authenticated `gh`.

### Step 2: Determine learnings across variants

Iterate over each variant's result. Not every variant produces a learning. Only record when:

**Failure worth recording** (new pattern not already in AGENT.md):
- A compilation error with a non-obvious cause
- A correctness error that reveals a Pallas/TPU constraint
- A timeout or infrastructure failure with a reproducible cause

**Success worth recording** (meaningful improvement):
- Speedup improved >=5% over the previous best
- A specific technique produced a measurable improvement
- The improvement reveals a generalizable optimization principle

**Comparative learnings** (patterns across variants or rounds):
- Direction X consistently outperforms direction Y across rounds
- A mutation type reliably produces improvements vs. another
- Lineage characteristics that predict success

Skip recording if:
- The failure is a trivial syntax error (typo, missing import)
- The speedup change is negligible (<5% improvement)
- A similar pattern already exists in AGENT.md (check by reading existing entries)

**Deduplication within the round**: Multiple variants may hit the same failure or discover the same optimization. Only record each unique pattern once, noting which variants exhibited it.

### Step 3: Update AGENT.md (if new learnings)

Write new learnings to the resolved local path, default `{run_dir}/AGENT.md`. Treat the repo-root `AGENT.md` as read-only seed knowledge. An explicitly configured `session.agent_md_path` changes the local destination only. If it doesn't exist, create it with:

```markdown
# Pallas Kernel Optimization Agent Knowledge

## Failure Patterns

## Successful Optimizations
```

**For failures**, find the next available `[Fxxx]` number and append under `## Failure Patterns`:

```markdown
### [F{NNN}] {Short description}
- **Symptom**: {What the error looks like — include key error message text}
- **Root cause**: {Why it happens — the underlying Pallas/TPU constraint}
- **Fix**: {How to avoid it in future kernels}
- **First seen**: {YYYY-MM-DD}, {kernel_name} optimization
```

**For successful optimizations**, find the next `[Sxxx]` number and append under `## Successful Optimizations`:

```markdown
### [S{NNN}] {Short description of the technique}
- **Optimization**: {What was changed in the kernel code}
- **Impact**: {before_speedup}x -> {after_speedup}x on {shape}
- **Why it works**: {Root cause analysis — why this specific change improved performance}
- **Applicable when**: {Conditions where this technique should be tried again}
- **First seen**: {YYYY-MM-DD}, {kernel_name} optimization
```

**Deduplication**: Before adding, read all existing entries. If a similar pattern exists:
- For failures: update the existing entry with additional context rather than creating a duplicate
- For successes: update with the new performance data if the technique is the same

Batch all updates into a single AGENT.md write — do not make multiple separate edits.

### Step 4: Save the round summary locally

Write `iteration_{N}/round_summary.md` with the variant status/speedup/direction table, active and pruned lineages, and new learning IDs. Persist this file even when GitHub is unavailable or disabled.

Only when validated `session.github_tracking` is true and this run has a tracking Issue (created by `pallas-evolve:start` and saved in the run directory), mirror the file to the explicitly configured destination, with `TRACKING_REPO="{session.tracking_repo}"`:

```bash
gh issue comment {issue_number} --repo "$TRACKING_REPO" --body-file "{run_dir}/iteration_{N}/round_summary.md"
```

Never fall back to checkout origin. On a GitHub error, retain the local file and report it without changing destinations.

### Step 5: Preserve local changes

Leave the learnings file and summaries on disk. Do not automatically stage, commit, push or create a PR, including when GitHub tracking is enabled. Those operations require a separate user request identifying the owning repository and files.

### Step 6: Context note

All learnings have been persisted:
- The resolved local learnings file — updated failure patterns and successful optimizations
- `iteration_{N}/round_summary.md` — complete round summary
- Optional GitHub mirror — report its URL only if successfully posted; otherwise report disabled/failed

**Do NOT compact here** — this skill is typically invoked within the `pallas-evolve:start` loop (Phase 4), and the subsequent Phase 5 (COMPACT) handles context compression with proper state verification. Compacting mid-loop would lose orchestration context.

If invoked **standalone** (outside the start loop), invoke `/compact` after this skill completes.
