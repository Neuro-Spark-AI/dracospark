# kernel-evolve

Python package and Claude Code plugin behind Dracospark's evolutionary TPU kernel
optimizer. The optimization loop is driven by the `pallas-evolve` skills inside
Claude Code; this package provides config validation, EVOLVE-BLOCK handling, TPU
evaluators, profiling analysis and offline evidence tools that those skills call.

## How It Works

```
        /pallas-evolve:start <config.yaml>
                      |
       think: generate N variants of the EVOLVE-BLOCK
                      |
       submit: evaluate the batch on a TPU VM over SSH
               (SSHEvaluator, evaluator.type: ssh)
                      |
       analyze: bottleneck classification, top-K lineages
                      |
       reflect: record failure patterns / successful optimizations
                      |
                 (loop back)
```

The package also provides `KubeEvaluator` (`evaluator.type: kube`) for running the
same evaluation as a GKE Job from Python or GitHub Actions; the skill loop itself
submits over SSH.

Each variant runs through `docker/evaluate.py`: compile, correctness against the
reference on every configured shape, performance on `shapes[0]`, XPlane trace
capture, and deep IR analysis (HLO/LLO/Mosaic dumps). Results are emitted as
`EVAL_RESULT:{json}` lines and parsed by the evaluator.

## Quick Start

```bash
pip install -e ".[dev]"

# Validate a config
python -c "from kernel_evolve.config import load_config; print(load_config('examples/matmul.yaml'))"

# Run one batch over SSH without Claude Code (TPU VM required)
python scripts/run_ssh_batch.py --config examples/matmul_v6e_ssh.yaml \
  --iteration-dir runs/matmul_v6e/iteration_1
```

In Claude Code, start the full loop with `/pallas-evolve:start qkv_attention_pallas.yaml` (the path is relative to `kernel-evolve/examples/`). The loop needs a profiled baseline in `examples/kernels/<kernel-name>_baseline/`; only `qkv_attention_pallas` ships one, and `/pallas-evolve:init-kernel` (Step 9d) generates it for other kernels.

## Configuration

See [`examples/matmul.yaml`](examples/matmul.yaml) (GKE) and
[`examples/matmul_v6e_ssh.yaml`](examples/matmul_v6e_ssh.yaml) (TPU VM over SSH)
for full examples. Replace the `REPLACE_WITH_*` placeholders with your own
cluster, hosts and keys. Key sections:

```yaml
kernel:
  name: "tiled_matmul"
  template: "kernels/matmul.py"       # Pallas kernel with EVOLVE-BLOCK markers
  reference: "kernels/matmul_ref.py"  # Reference implementation for correctness

shapes:                               # correctness checks all; performance uses shapes[0]
  - { M: 1024, N: 1024, K: 1024 }

correctness:
  method: "allclose"                  # or "abs"
  rtol: 1e-2
  atol: 1.0

evaluator:
  type: "kube"                        # or "ssh"; see SSH_V6E.md
  job_template: ".github/ci/kernel-eval-job.yaml"
  repo: "Neuro-Spark-AI/dracospark"   # repository the Job clones

session:
  output_dir: "runs/matmul"
  github_tracking: false              # opt-in; requires tracking_repo: "owner/name"

batch:
  variants_per_round: 5
  top_k: 2
```

For the GKE transport, set `GCS_BUCKET` in the environment that runs the
evaluator to upload profile artifacts to your bucket; when it is unset the
upload is skipped. The SSH transport copies artifacts back with `scp` and needs
no bucket. See [`SSH_V6E.md`](SSH_V6E.md) for TPU VM setup.

For moving an optimization method to another model, shape or caller, use
`pallas-evolve:transfer` and the [structured transfer workflow](../docs/optimization-transfer.md).
`python -m kernel_evolve.transfer` evaluates target guards and checksum-pinned
evidence, records per-scope outcomes, and blocks stale or lifecycle-incomplete
acceptance. It does not launch jobs or deploy candidates.

For attention tuning across models, see the
[general optimization and regression policy](../docs/attention-optimization-policy.md).
It separates capability guards, regression diagnosis and target-workload promotion.
The offline `python -m kernel_evolve.regression`
comparator produces paired timing/trace evidence and next experiments without
launching hardware or changing defaults; its input format is in the
[analysis reference](plugins/pallas-evolve/skills/analyze/references/regression-diagnosis.md).
The [validation guide](../docs/regression-validation.md) covers host tests and
optional synthetic tile experiments. `scripts/validate_regression_tpu.py`
captures these fixtures on an already-authorized idle TPU slice; it does not
reserve machines.

## GKE Evaluation

`KubeEvaluator` renders a Job template from `.github/ci/` (substituting
`$JOB_NAME`, `$BRANCH`, `$REPO`, `$VARIANT_ID`, `$ACTIVE_DEADLINE` and
`$GCS_BUCKET`), creates a ConfigMap with the payload, applies the Job with
`kubectl`, and parses `EVAL_RESULT:` from the pod logs. The pod clones
`evaluator.repo` at `evaluator.branch`, so variants are evaluated against the
`evaluate.py` in that branch. `kubectl` must point at a GKE cluster with TPU node
pools; the templates request `tpu7x` nodes with a `2x2x1` topology — adjust the
`nodeSelector` for your hardware.

The [`kernel-eval.yaml`](../.github/workflows/kernel-eval.yaml) workflow runs the
same Job from GitHub Actions; see the top-level README for the repository
variables it needs.

## Claude Code Plugin

The pallas-evolve skills are bundled as a Claude Code plugin at
`plugins/pallas-evolve/`, published through the marketplace at the repository
root (`.claude-plugin/marketplace.json`). Install it as described in the
top-level README:

```text
/plugin marketplace add Neuro-Spark-AI/dracospark   # or ./ from a local clone
/plugin install pallas-evolve@dracospark
```

When you edit skills in a local clone, run `/plugin marketplace update dracospark`
(or reinstall the plugin) and restart Claude Code so the new versions load. If the
skills don't appear, run `/doctor`.

### Available skills

```
/pallas-evolve:init-kernel <name> --repo <url>  # Initialize a kernel project from an upstream repo
/pallas-evolve:profile-brief                    # Generate a profile brief from TPU eval artifacts
/pallas-evolve:start <config.yaml>              # Start an optimization session
/pallas-evolve:submit                           # Submit variants for TPU evaluation
/pallas-evolve:analyze                          # Analyze evaluation results
/pallas-evolve:reflect                          # Record learnings to AGENT.md
/pallas-evolve:transfer                         # Adapt a method to a new workload
```

## Output

Each session writes to a timestamped run directory under `session.output_dir`:
the original config and kernel snapshots, a reloadable `config.yaml`, one
`iteration_N/` directory per round (variants, `eval_result.json` files and
artifacts), round summaries, and a run-local `AGENT.md` with new learnings.
Nothing is committed or pushed automatically.

## Project Structure

```
kernel-evolve/
├── src/kernel_evolve/
│   ├── config.py                  # Pydantic YAML config validation
│   ├── mutation.py                # EVOLVE-BLOCK extraction/injection
│   ├── evaluator.py               # EvalRequest/EvalResult and batch data classes
│   ├── kube_evaluator.py          # GKE Job submission via kubectl
│   ├── ssh_evaluator.py           # TPU VM evaluation over SSH
│   ├── docker_evaluate_helpers.py # Batch subprocess dispatch
│   ├── profiler.py                # XPlane trace & IR analysis
│   ├── regression.py              # Offline paired-evidence comparator
│   └── transfer.py                # Offline optimization-transfer records
├── docker/
│   ├── Dockerfile                 # TPU evaluation container
│   └── evaluate.py                # 5-stage evaluator
├── examples/                      # Configs, kernels, references, transfer plan
├── plugins/pallas-evolve/         # Claude Code skill plugin
├── scripts/                       # Utility and validation scripts
└── tests/                         # pytest suite
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

`tests/standalone_gmm_fp8_blockwise_test.py` runs the real FP8 kernel and needs a
TPU with `qwix` and `tokamax`; it is skipped automatically when those packages are
missing. The SSH cleanup escape tests run only on Linux, where the remote script runs.
