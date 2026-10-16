---
name: start
description: Use when starting a Pallas kernel optimization session on a TPU-VM (v6e) over SSH — reads config, verifies baseline, records locally with optional GitHub tracking, runs profile-informed think-submit-analyze-reflect-compact loop with lineage tracking
---

# Pallas Kernel Batch Optimization

Optimize a Pallas TPU kernel through batch evolutionary mutation with lineage tracking. Each round generates N variants exploring genuinely different TECHNICAL DIRECTIONS, evaluates them serially on a TPU-VM over SSH, selects top-K as surviving lineages, and iterates. You are the optimization brain.

**Evaluator path**: this skill drives the SSH evaluator only (`evaluator.type: ssh`, a Cloud TPU VM such as v6e). Configs that use the GKE evaluator (`evaluator.type: kube`, the default when `type` is omitted — e.g. `examples/matmul.yaml`, `examples/gmm_fp8_blockwise.yaml`) are not supported by this loop; evaluate those through `KubeEvaluator` / the Kernel Evaluation GitHub workflow instead.

## Arguments

Expects a config YAML path. A bare filename resolves under `kernel-evolve/examples/`; a path such as `examples/<name>.yaml` resolves relative to `kernel-evolve/`:

```
/pallas-evolve:start qkv_attention_pallas.yaml
```

SSH-ready examples: `qkv_attention_pallas.yaml` (ships with a baseline), `matmul_pallas.yaml` and `matmul_v6e_ssh.yaml` (generate a baseline first, see Step 6).

**Before running**, the user must:
- Replace every `REPLACE_WITH_*` value under `evaluator` (worker IPs in `ssh_hosts` or `ssh_host`, `ssh_user`, `ssh_key`) with their own TPU VM details.
- Prepare every worker as described in `kernel-evolve/SSH_V6E.md`: a checkout at `ssh_remote_repo` (default `~/dracospark`) and a Python env at `ssh_python` (default `~/tpu-venv/bin/python`) with JAX/libtpu and `kernel-evolve` installed.
- Optional: set `session.github_tracking: true` plus an explicit `session.tracking_repo: owner/name` to mirror summaries to a GitHub Issue. Without it, everything is recorded locally under the run directory.

## Startup

Execute these steps in order:

1. **Read config**: Validate the YAML with `kernel_evolve.config.load_config` before any remote or GitHub action. Read the YAML config file resolved from the argument (see Arguments). Extract: kernel name, template path, reference path, shapes, correctness method/thresholds, evaluator settings (`type`, `ssh_host`/`ssh_hosts`, `ssh_user`, `ssh_key`, `ssh_remote_repo`, `ssh_python`, `ssh_remote_tmp`, `ssh_artifacts_dir`, `ssh_env`, `ssh_extra_libtpu_init_args`, `timeout`), `roofline` peaks, session settings (`max_iterations`, `output_dir`, `github_tracking`, `tracking_repo`, `agent_md_path`).

   **Stop** if `evaluator.type` is not `ssh` (this loop does not drive GKE), if no host is configured, or if any `REPLACE_WITH_*` placeholder remains; tell the user which values to fill in.

2. **Read batch config**: From the same YAML, extract the `batch` section:
   - `variants_per_round` (N): how many variants to generate per active lineage per round
   - `top_k` (K): how many best lineages survive each round
   - `max_active_lineages`: cap on total active lineages to prevent exponential growth

   If `batch` section is missing, use the config defaults `variants_per_round=1, top_k=1, max_active_lineages=4` (single-variant behavior).

3. **Read kernel template**: Read the kernel file at the template path (relative to config dir). Identify the `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` region — this is what you will optimize.

4. **Read reference kernel**: Read the reference implementation. This is the correctness baseline.

5. **Read existing knowledge**: Read the repo-root `AGENT.md` as seed knowledge without editing it. Also read `session.agent_md_path` if explicitly configured. New learnings default to `{run_dir}/AGENT.md`, resolved after Step 9. Existing knowledge contains:
   - `## Failure Patterns` — known errors and how to avoid them. **You MUST avoid these.**
   - `## Successful Optimizations` — proven strategies. **Prioritize these.**

6. **Check SSH and baseline**: Verify connectivity to the TPU-VM(s). Forward every `evaluator.ssh_env` assignment to each probe with shell-quoted values, using the same environment as evaluation. Reduced-topology settings such as `TPU_VISIBLE_CHIPS` and `TPU_PROCESS_BOUNDS` must apply before JAX imports. Bound each remote probe with `timeout 120`; a probe without these overrides can wait on unlaunched peers. In the commands below, `{ssh_env_assignments}` means those shell-quoted `KEY=value` arguments, or an empty string when no overrides are configured.

   **Single-host slice** (`evaluator.ssh_host`, or `ssh_hosts` with one entry):
   ```bash
   ssh -i {ssh_key} {ssh_user}@{host} "env {ssh_env_assignments} timeout 120 {ssh_python} -c 'import jax; print(jax.devices())'"
   ```

   **Multi-host slice** (`evaluator.ssh_hosts` with >1 entry, e.g. v6e-16): a
   single-host `jax.devices()` HANGS on the ICI init barrier — it can only
   succeed when all hosts start together. Co-launch the check on every host and
   wait for all:
   ```bash
   SSH_HOSTS=( {space-separated evaluator.ssh_hosts} )
   pids=()
   for H in "${SSH_HOSTS[@]}"; do
     ssh -i {ssh_key} -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 \
       {ssh_user}@"$H" "env {ssh_env_assignments} timeout 120 {ssh_python} -c 'import jax; print(jax.process_index(), jax.devices()[0].device_kind, len(jax.devices()))'" </dev/null &
     pids+=($!)
   done
   for p in "${pids[@]}"; do wait "$p"; done
   ```
   Every host should print its process index, device kind (`TPU v6 lite`), and
   global device count. If it hangs, a worker IP is stale or unreachable —
   re-discover all workers with gcloud (see the submit skill's error-handling
   note). Then verify baseline profiling exists:

   ```bash
   ls kernel-evolve/examples/kernels/{kernel_name}_baseline/eval_result.json
   ```

   If the baseline directory or `eval_result.json` is missing, **stop**: "Baseline not found. Generate it with the baseline-profiling step (Step 9d) of `/pallas-evolve:init-kernel`, or run `/pallas-evolve:init-kernel` to import a new kernel." In this repository only `qkv_attention_pallas` ships a baseline.

7. **Ask mode**: Use AskUserQuestion with header "Mode" and two options:
   - "Step-by-step" — pause after each round for user review and direction
   - "Autonomous" — run continuously until a termination condition is met

8. **Resolve recording policy**: `session.github_tracking` defaults to `false`. An existing `tracking_repo`, authenticated `gh`, or autonomous mode does not enable it. With tracking disabled, skip every GitHub command and continue without an Issue or GitHub credentials. With tracking enabled, require an explicit `owner/name` destination and check its identity/visibility before sending workload data; private workload records require a private destination. Never fall back to checkout origin. The opt-in covers Issue creation, round comments and closure; commit, push and PR publication require a separate user request.

9. **Initialize run directory**: Create `kernel-evolve/{output_dir}_{YYYYMMDD_HHMMSS}/`:
   - Call `kernel_evolve.config.snapshot_run_inputs(config_path, run_dir)`. It preserves `source_config.yaml`, snapshots the template as `baseline_kernel.py` and the reference as `reference.py`, and writes `config.yaml` with run-relative kernel paths. Existing inputs are never overwritten.
   - Use the saved `config.yaml` for submission and state reload; do not depend on the original config directory after initialization.
   - Copy baseline artifacts: `cp -r kernel-evolve/examples/kernels/{kernel_name}_baseline/ {run_dir}/baseline/`
   - Initialize `lineages.json`:
     ```json
     {"lineages": [], "pruned": [], "round": 0}
     ```
   - Resolve the learnings path with `config.session.resolved_agent_md_path(run_dir)`. If absent, initialize it from the repo-root `AGENT.md` when available, otherwise create an empty knowledge file. Never overwrite existing learnings.
   - Write `session_summary.md` with the kernel, shapes, config, recording policy and destination. Update it at session end.
   - Set `iteration = 0`

   Create a tracking Issue only when `session.github_tracking` is true, using `TRACKING_REPO="{session.tracking_repo}"` from the validated config and a local body file:
   ```bash
   gh issue create --repo "$TRACKING_REPO" \
     --title "[pallas-evolve] Optimize {kernel_name}" \
     --body-file "{run_dir}/session_summary.md"
   ```
   Save the returned Issue URL in the run directory (e.g. `{run_dir}/tracking_issue.txt`) so reflect and End of Session can reuse it after compaction. If creation fails, keep local records and report the failure; do not silently switch destinations.

## Optimization Loop

In all phases below, `AGENT.md` means the resolved run-local or explicitly configured learnings file. Pass that resolved path and recording policy to child skills and sub-agents; retain them in the saved run state across compaction.

For each round from 1 to `max_iterations`:

### Phase 0: PROFILE (Generate Profile Brief)

Before generating variants, generate a profile brief from the previous round's best variant for sub-agents.

1. **Identify the profile source**:
   - Round 1: Use `{run_dir}/baseline/` (copied from `examples/kernels/{kernel_name}_baseline/` during Step 9). The baseline `profile_brief.md` already exists — read it directly and skip to step 3.
   - Round 2+: Read `lineages.json`, find the lineage with the highest `best_speedup`, use its `best_kernel` directory (the directory containing the kernel file, which should also contain `eval_result.json`, `llo_final.txt`, etc.)

2. **Generate profile brief (Round 2+ only)**: Invoke `pallas-evolve:profile-brief` via the Skill tool:

   ```
   /pallas-evolve:profile-brief {profile_source_dir} --baseline {run_dir}/baseline/ --round {N}
   ```

   This generates `profile_brief.md` in the profile source directory with a delta-vs-baseline table.

3. **Copy profile brief to iteration directory**: Copy `profile_brief.md` to `{run_dir}/iteration_{N}/profile_brief.md` for archival. For Round 1, copy from `{run_dir}/baseline/profile_brief.md`. For Round 2+, copy from the profile source directory (where `profile-brief` skill wrote it in step 2).

4. **Read the profile brief into a variable** for passing to sub-agents in Phase 1.

### Phase 1: THINK (Batch Variant Generation)

**CRITICAL RULE**: Each variant MUST explore a genuinely different TECHNICAL DIRECTION. Changing block sizes from 128 to 256 is NOT a different direction — that is a parameter variation. Different directions means fundamentally different optimization APPROACHES:
- Tiling strategy (K-tiling, 2D tiling, diamond tiling)
- HBM–compute overlap (double buffering, emit_pipeline, DMA prefetch to hide HBM latency)
- MXU–VPU overlap (schedule matrix ops and vector ops to run concurrently on separate units)
- Memory layout (scratch memory, accumulator placement, data reuse patterns)
- MXU utilization (dual MXU balancing, operand layout for matrix units)
- Vectorization (inner dimension alignment, loop restructuring for SIMD)

**Parallel generation**: Use the Agent tool to generate ALL variants concurrently. Launch one sub-agent per variant in a single message so they run in parallel. Each sub-agent independently writes its kernel file. After all sub-agents complete, the main agent collects their results and writes `strategy.md`.

#### Round 1 (No lineages yet)

Generate N variants from the baseline kernel, each exploring a DIFFERENT technical direction **derived from the profile brief's bottleneck diagnosis and optimization priorities**:

1. Re-read AGENT.md failure patterns and successful optimizations
2. Prepare shared context for sub-agents: read the template kernel content, AGENT.md content, and **profile brief content** (from Phase 0) into variables
3. **Derive N directions from the profile brief**: Read the profile brief's "Optimization Priorities" and "Bottleneck Diagnosis" sections. Based on the diagnosed bottleneck and supporting metrics, select N concrete optimization directions. Each direction must be a genuinely different technical approach motivated by a specific profile signal. For example, if the profile shows `compute_ratio=0.35` (memory-bound) with `dual_ratio=0.0` and `1.9M register spills`, you might derive:
   - `k_tiling` — motivated by low compute_ratio (memory-bound)
   - `register_pressure_reduction` — motivated by 1.9M spills
   - `mxu_dual_scheduling` — motivated by dual_ratio=0.0
   - `dma_prefetch` — motivated by no double buffering in LLO trace
   - `loop_restructuring` — motivated by low ILP (avg_ops_per_bundle)

   **Do NOT use a fixed list of directions.** Each session's directions must be uniquely tailored to what the profile data actually shows.
4. **Dispatch N sub-agents in parallel** (one Agent tool call per direction, all in a single message):
   Each sub-agent receives a prompt containing:
   - The full template kernel content
   - The AGENT.md failure patterns and successful optimizations
   - Its assigned direction name (derived from the profile brief in step 3 above)
   - The TPU v6e hard rules and optimization knowledge from this skill
   - **The profile brief content** (full text of `iteration_1/profile_brief.md` generated in Phase 0), including hardware utilization, bottleneck diagnosis, LLO key observations, and optimization priorities
   - **Direction-specific guidance**: Based on the profile brief's bottleneck diagnosis, explain what specific profile signal motivates this sub-agent's direction and what concrete optimization it should attempt. For example: "The profile shows compute_ratio=0.35 (memory-bound) with no double buffering. Your direction `dma_prefetch` should focus on adding DMA prefetch to hide the 65% memory transfer time visible in the LLO trace."
   - The output path: `iteration_1/variants/{direction_name}/kernel.py`
   - Instructions to:
     - Design an optimization approach specific to its assigned direction
     - Write the mutated kernel file (keep everything outside EVOLVE-BLOCK unchanged)
     - Replace only the code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`
     - Preserve function signatures
     - Validate: the file must be valid Python (no syntax errors)
     - Return a summary: approach taken, **which profile signal motivated this approach**, expected impact on the identified bottleneck, **which hw utilization metric is expected to improve**, key changes
5. Collect the returned summaries from all sub-agents and write `iteration_1/strategy.md`:
   ```markdown
   ## Round 1 Strategy

   Generating {N} variants from baseline, each exploring a different technical direction derived from profile analysis.
   Variants generated in parallel via sub-agents.

   ### Variant: {direction_1}
   **Technical direction**: {direction_name}
   **Profile motivation**: {which profile signal drove this approach}
   **Approach**: {specific optimization technique}
   **Expected impact**: {why this should improve performance, referencing profile metrics}
   **Target metric improvement**: {e.g., "compute_ratio 0.35 → 0.60+", "dual_ratio 0.3 → 0.8+"}
   **Key changes**: {summary of code changes}

   ### Variant: {direction_2}
   ...
   ```

#### Round 2+ (Has active lineages)

1. Read `lineages.json` for active lineages
2. Re-read AGENT.md failure patterns and successful optimizations
3. For each active lineage, read its `best_kernel` file
4. **Derive directions from the current round's profile brief**: Read `iteration_{N}/profile_brief.md` (generated in Phase 0). Based on the updated bottleneck diagnosis, delta vs baseline, and optimization priorities, select directions that address the **current** bottleneck — which may differ from previous rounds as the kernel evolves. Avoid repeating directions that the reflect skill (AGENT.md) has flagged as unsuccessful.
5. Plan variant assignments:
   - If only 1 active lineage: assign N different profile-derived directions to it
   - If multiple active lineages: distribute N variants across lineages (at least 1 per lineage, extras to best-performing lineages)
6. Variant naming convention: `{lineage_id}_{direction}` (e.g., `L1_tiling`, `L2_memory`)
7. **Dispatch all variant sub-agents in parallel** (one Agent tool call per variant, all in a single message):
   Each sub-agent receives a prompt containing:
   - The base kernel content (from its assigned lineage's `best_kernel`)
   - The AGENT.md failure patterns and successful optimizations
   - Its assigned direction and lineage context (lineage ID, previous best speedup, prior direction)
   - The TPU v6e hard rules and optimization knowledge from this skill
   - **The profile brief content** (full text of `iteration_{N}/profile_brief.md`), including the delta vs baseline table, hardware utilization, bottleneck diagnosis, and LLO key observations
   - **Direction-specific guidance**: Based on the profile brief's bottleneck diagnosis, explain what specific profile signal motivates this direction and what concrete optimization it should attempt, referencing what changed since the previous round
   - The output path: `iteration_{N}/variants/{variant_name}/kernel.py`
   - Same mutation instructions as Round 1
   - Return a summary: approach taken, **which profile signal motivated this approach**, expected impact on the identified bottleneck, **which hw utilization metric is expected to improve**, key changes
8. Collect the returned summaries from all sub-agents and write `iteration_{N}/strategy.md`:
   ```markdown
   ## Round {N} Strategy

   Active lineages: {count}
   Total variants this round: {total}
   Variants generated in parallel via sub-agents.

   ### Lineage {L_id} (best speedup: {X}x, direction: {dir})

   #### Variant: {L_id}_{direction_1}
   **Base kernel**: {lineage best_kernel path}
   **Technical direction**: {direction_name}
   **Profile motivation**: {which profile signal drove this approach}
   **Approach**: {specific optimization technique}
   **Expected impact**: {why this should improve performance, referencing profile metrics}
   **Target metric improvement**: {e.g., "compute_ratio 0.35 → 0.60+", "dual_ratio 0.3 → 0.8+"}

   #### Variant: {L_id}_{direction_2}
   ...
   ```

### Phase 2: SUBMIT

Invoke `pallas-evolve:submit` via the Skill tool.

The submit skill will:
- Collect all variant kernels into a batch payload
- Stage it on the TPU-VM and run evaluate.py over SSH, evaluating all variants serially via subprocess isolation
- Parse multiple `EVAL_RESULT:` lines from the SSH run output
- Save individual results to `iteration_{N}/variants/{variant_name}/eval_result.json`
- scp IR/trace artifacts back into each variant dir

### Phase 3: ANALYZE

Invoke `pallas-evolve:analyze` via the Skill tool.

The analyze skill will:
- Read all variant results for the round
- Rank by speedup, compare across directions
- Select top-K lineages (create new lineages or update existing ones)
- Enforce `max_active_lineages` cap
- Update `lineages.json` with new state
- Write `iteration_{N}/batch_analysis.md` and `iteration_{N}/selection.md`

### Phase 4: REFLECT

Invoke `pallas-evolve:reflect` via the Skill tool.

The reflect skill will:
- Extract learnings from ALL variant results (batch)
- Update the resolved local learnings file with failure patterns and successful optimizations
- Write `iteration_{N}/round_summary.md`; mirror it to the explicit tracking Issue only when `session.github_tracking` is true

### Phase 5: COMPACT

Before compacting, verify all critical state is persisted to files:
- `lineages.json` is up to date (written by analyze)
- All `eval_result.json` files exist for each variant
- `strategy.md`, `batch_analysis.md`, `selection.md` are written
- `profile_brief.md` exists for the current iteration (written by Phase 0)
- The resolved learnings file and local round summary are updated (written by reflect)
- The validated recording policy and resolved learnings path remain available from the run state

Then invoke `/compact` to compress conversation context. After compaction, the loop will reconstruct state from files.

### Phase 6: CONTINUE?

Read the updated `lineages.json` (written by the analyze skill).

**Step-by-step mode**: Present a lineage summary table and ask the user:

```
## Round {N} Summary

| Lineage | Best Speedup | Direction | Rounds Active | Last Improvement |
|---------|-------------|-----------|---------------|------------------|
| L1      | 1.62x       | tiling    | 3             | Round 3          |
| L2      | 1.45x       | memory    | 2             | Round 2          |

Total variants evaluated this round: {count}
Best overall speedup: {best}x (Lineage {id})
```

Options: "Continue", "Adjust strategy", "Stop"

If "Adjust strategy": ask what directions to prioritize, which lineages to prune, or whether to change variants_per_round.

**Autonomous mode**: Check termination conditions:
- `round >= max_iterations` -> stop
- ALL active lineages have not improved for 3 consecutive rounds -> stop (stagnation)
- Otherwise -> continue to next round

## End of Session

When the loop terminates:

1. **Report final results**: For each lineage, report its best speedup, direction, and evolution history. Highlight the overall best lineage.

2. **Save the best lineage locally**: Copy the selected kernel to `{run_dir}/best/{kernel_name}.py`, preserving its evaluation evidence. Update `{run_dir}/session_summary.md` with the final lineage table, measured speedup, shape, timing source and limitations. Report these paths even when no improvement was found. Do not replace the source template or automatically commit, push, or create a PR. If the user separately requests publication, verify the owning repository and publish only the requested files there.

3. **Finish optional tracking**: If `session.github_tracking` is true and this run has a tracking Issue, post the local final summary with `gh issue comment {issue_number} --repo "$TRACKING_REPO" --body-file "{run_dir}/session_summary.md"`, then close that Issue with the same explicit `--repo`. Otherwise finish with local records only.

## TPU v6e Pallas Optimization Knowledge

When writing kernel mutations, follow these TPU v6e (Trillium) constraints:

**Hard rules (violating these causes compilation errors):**
- Always use `jnp.bfloat16`, never `jnp.float16` — Mosaic compiler requires it
- Use `Ref` indexing: `x_ref[...]` or `x_ref[pl.ds(start, size)]` — never `pl.load()`
- Use `pl.ds(start, size)` — never `pl.dslice(start, size)`
- Kernel function signature must stay unchanged (same Ref arguments)
- `optimized_compute` signature must stay unchanged (same shape parameter names and defaults as the template, e.g. `M, N, K` for matmul)

**Optimization levers (from most to least impactful):**
1. **K-tiling**: Split the K (reduction) dimension into tiles. Accumulate partial results in scratch memory. Reduces HBM bandwidth pressure.
2. **Block size tuning**: Try 64, 128, 256. Larger blocks = more compute per tile but more VMEM. Match to matrix dimensions.
3. **Scratch memory**: Allocate VMEM scratch accumulators (`scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)]`) instead of writing back to HBM each tile.
4. **HBM–compute overlap**: Use `pltpu.emit_pipeline` or manual double buffering to overlap HBM DMA transfers with MXU/VPU compute. Prefetch the next tile while computing the current one.
5. **MXU–VPU overlap**: Schedule MXU matmul operations and VPU element-wise/reduction operations to execute concurrently. TPU can run both units in parallel when data dependencies allow.
6. **Vectorization**: Ensure inner dimensions are multiples of 128 for MXU efficiency.

**Common pitfalls:**
- Block size 512 may OOM on 2048x2048 matrices (VMEM limit)
- Accumulator dtype should be `jnp.float32` for numerical stability, cast to `bfloat16` on store
- Grid dimensions must evenly divide matrix dimensions
