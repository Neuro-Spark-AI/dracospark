# Dracospark

LLM-driven evolutionary optimizer for JAX Pallas TPU kernels. Automatically discovers high-performance kernel variants through iterative mutation, evaluation on real TPU hardware (GKE TPU v7x or a Cloud TPU VM over SSH), multi-signal profiling, and structured reflection.

## Learn more

Our [launch blog](https://neuro-spark.ai/blog/dracospark-tpu-kernel-optimization) explains how Dracospark uses profiling, guided search, and validation to optimize TPU kernels. It also walks through a TPU kernel optimization example and the broader engineering work behind MiniMax H3 performance improvements, including the test conditions and measurement limitations. The kernel experiment and H3 service-level results are separate measurements; the H3 improvements are not attributed solely to the agent.

## Model and framework inference optimization

`inference-opt/` is an independent, installable harness for evidence-bound
whole-request inference optimization (frozen A/B plans, ordinary benchmarks,
layered profiles, registered agent proposals, paired confirmation, immutable
evidence and read-only review). Model- and runtime-specific semantics are
supplied by adapters; see [`inference-opt/README.md`](inference-opt/README.md)
and the CPU fixture.

The `inference-opt` Skill plugin is available from this marketplace alongside
`pallas-evolve`; installing a Skill does not install its Python package. The
Pallas flow below is independent of it, and its kernel-specific `AGENT.md`
learnings are not universal constraints on other models or hardware.

## How It Works

Dracospark operates as an interactive Claude Code skill loop:

```
                 pallas-evolve:start
                        |
          +-------------+-------------+
          |                           |
      THINK (LLM)              Read config &
    Generate N kernel          record session
    variant mutations            locally
          |                           |
          +-------------+-------------+
                        |
                pallas-evolve:submit
            Build payloads, evaluate on
            a TPU VM over SSH, collect results
                        |
                pallas-evolve:analyze
            Multi-signal bottleneck
            classification, top-K
            lineage selection
                        |
                pallas-evolve:reflect
            Extract failure patterns &
            successful optimizations
            into AGENT.md
                        |
                    compact
                 (loop back)
```

Each iteration:

1. **Think** -- Generate N kernel variants by mutating code between `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` markers, guided by learnings in `AGENT.md`
2. **Submit** -- Package variants into a batch, evaluate them on a Cloud TPU VM over SSH, and run a 5-stage evaluation pipeline (compile, correctness, performance, XPlane trace, deep IR analysis)
3. **Analyze** -- Classify bottlenecks using compute ratio, VLIW bundles, MXU utilization, arithmetic intensity, HBM bandwidth, register spills, and more. Select top-K lineages for the next round.
4. **Reflect** -- Record failure patterns (`[Fxxx]`) and successful optimizations (`[Sxxx]`) into the run-local `AGENT.md`. Save round summaries locally; GitHub mirroring is opt-in.

## Project Structure

```
dracospark/
├── kernel-evolve/                    # Core Python package
│   ├── src/kernel_evolve/
│   │   ├── config.py                 # Pydantic YAML config validation
│   │   ├── mutation.py               # EVOLVE-BLOCK extraction/injection
│   │   ├── evaluator.py              # EvalResult/EvalRequest data classes
│   │   ├── kube_evaluator.py         # K8s Job submission via kubectl
│   │   ├── ssh_evaluator.py          # TPU VM evaluation over SSH
│   │   ├── profiler.py               # XPlane trace & IR analysis
│   │   ├── regression.py             # Offline paired-evidence comparator
│   │   ├── transfer.py               # Offline optimization-transfer records
│   │   └── docker_evaluate_helpers.py # Batch subprocess dispatch
│   ├── docker/
│   │   ├── Dockerfile                # TPU evaluation container
│   │   └── evaluate.py               # 5-stage evaluator (K8s Pod or TPU VM)
│   ├── examples/
│   │   ├── matmul.yaml               # Simple tiled matmul config (GKE)
│   │   ├── matmul_v6e_ssh.yaml       # Tiled matmul on a TPU VM over SSH
│   │   ├── matmul_pallas.yaml        # K-tiled Pallas matmul (SSH)
│   │   ├── qkv_attention_pallas.yaml # qkv self-attention Pallas core (SSH)
│   │   ├── gmm_fp8_blockwise.yaml    # Grouped MatMul FP8 config (GKE)
│   │   ├── transfers/                # Synthetic optimization-transfer plan
│   │   └── kernels/                  # Template & reference implementations
│   ├── plugins/pallas-evolve/        # Claude Code skill plugin
│   │   └── skills/{init-kernel,profile-brief,start,submit,analyze,reflect,transfer}/
│   ├── scripts/                      # Utility scripts
│   ├── SSH_V6E.md                    # TPU VM (SSH) evaluator guide
│   └── tests/                        # pytest suite
├── inference-opt/                    # Whole-request inference optimization harness
├── .github/
│   ├── workflows/
│   │   ├── kernel-eval.yaml          # TPU evaluation workflow
│   │   ├── build-image.yaml          # Docker image build & push
│   │   └── inference-opt.yml         # inference-opt CPU contract tests
│   └── ci/
│       ├── kernel-eval-job.yaml      # K8s Job template
│       ├── kernel-eval-gmm-job.yaml  # GMM FP8 variant Job template
│       └── xplane-explore-job.yaml   # XPlane exploration Job
├── AGENT.md                          # Accumulated optimization learnings
├── docs/                             # Transfer, regression and attention policy guides
└── LICENSE                           # Apache 2.0
```

## Prerequisites

- Python >= 3.10
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed
- TPU access of your own:
  - for the skill loop, a Cloud TPU VM (e.g. v6e) reachable over SSH (see [`kernel-evolve/SSH_V6E.md`](kernel-evolve/SSH_V6E.md))
  - optionally, `kubectl` access to a GKE cluster with TPU node pools for `KubeEvaluator` and the `kernel-eval` workflow (the Job templates target TPU v7x), plus a GCS bucket for profile artifacts (set `GCS_BUCKET`; there is no default bucket)

See [Infrastructure](#infrastructure) for setting up your own GCP/TPU resources.

## Installation

Clone the repository:

```bash
git clone https://github.com/Neuro-Spark-AI/dracospark.git
cd dracospark
```

Example configs record sessions locally; GitHub tracking is off by default and requires an explicit `tracking_repo` you control (see [Session records](#session-records-and-optional-github-tracking)).

### 1. Install the Python package

```bash
cd kernel-evolve
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[charts]"    # matplotlib for visualization
pip install -e ".[profile]"   # xprof for trace analysis
```

### 2. Install the pallas-evolve skill plugin

The `pallas-evolve` plugin ships with 7 skills that drive the optimization loop inside Claude Code.

The repository root is a Claude Code plugin marketplace named `dracospark` (`.claude-plugin/marketplace.json`). In Claude Code, add it and install the plugin:

```text
/plugin marketplace add Neuro-Spark-AI/dracospark
/plugin install pallas-evolve@dracospark
```

If the `/plugin` command in your Claude Code client does not register the marketplace, add it to `~/.claude/settings.json` instead and start a new session:

```json
{
  "extraKnownMarketplaces": {
    "dracospark": { "source": { "source": "github", "repo": "Neuro-Spark-AI/dracospark" } }
  },
  "enabledPlugins": { "pallas-evolve@dracospark": true }
}
```

To develop against your local clone or a fork, add the marketplace by path (`/plugin marketplace add ./` from the repository root) or as `<your-org>/dracospark`. The same marketplace also provides the `inference-opt` plugin (see [below](#model-and-framework-inference-optimization)).

### 3. Verify skill availability

Launch Claude Code in the project directory. You should see these skills available:

| Skill | Purpose |
|-------|---------|
| `pallas-evolve:init-kernel` | Initialize a kernel project from an upstream Pallas kernel repository (`--repo` required) |
| `pallas-evolve:profile-brief` | Generate a profile brief from TPU eval artifacts |
| `pallas-evolve:start` | Start an optimization session |
| `pallas-evolve:submit` | Submit a batch of variants for TPU eval |
| `pallas-evolve:analyze` | Analyze batch evaluation results |
| `pallas-evolve:reflect` | Record learnings to AGENT.md |
| `pallas-evolve:transfer` | Adapt optimization methods to new workloads and record guarded target validation |

### 4. Check the install without a TPU

The test suites and the inference-opt CPU fixture run on any machine:

```bash
cd kernel-evolve && pytest            # TPU-only scripts are skipped without qwix/tokamax
cd ../inference-opt && pip install -e ".[dev]" && pytest
python examples/run_cpu_fixture.py --output /tmp/inference-opt-fixture
```

## Usage

### Start an optimization session

The interactive skill loop evaluates on a Cloud TPU VM over SSH (`evaluator.type: "ssh"`). Copy an SSH example such as `kernel-evolve/examples/qkv_attention_pallas.yaml`, replace every `REPLACE_WITH_*` value with your TPU VM's hosts, user and key, and set up the VM as described in [`kernel-evolve/SSH_V6E.md`](kernel-evolve/SSH_V6E.md). Then, in Claude Code, pass the config path relative to `kernel-evolve/examples/`:

```
/pallas-evolve:start qkv_attention_pallas.yaml
```

This will:
1. Validate the config (shapes, correctness settings, SSH evaluator fields)
2. Check SSH connectivity and the baseline on the TPU VM
3. Create local session records; create a tracking Issue only when explicitly enabled
4. Enter the think-submit-analyze-reflect loop

The loop starts from a profiled baseline in `kernel-evolve/examples/kernels/<kernel-name>_baseline/`. Only `qkv_attention_pallas` ships one; for other kernels, generate it first with the baseline-profiling step (Step 9d) of `/pallas-evolve:init-kernel`.

The GKE configs (`matmul.yaml`, `gmm_fp8_blockwise.yaml`, `evaluator.type: "kube"`) are evaluated through `KubeEvaluator` from Python or the `kernel-eval` GitHub workflow (see [Infrastructure](#infrastructure)); the skill loop's submit step uses SSH.

### Transfer an optimization to a new workload

Invoke `pallas-evolve:transfer` with the source method and target call site. It
connects adaptation and authorized evaluation to a structured, model-independent
record: capability guards, exact contexts, kernel/caller/request comparisons,
diagnostic interventions and scoped acceptance checks. A standalone win cannot
substitute for a caller win, and no production default is changed automatically.
See [optimization transfer](docs/optimization-transfer.md) for the executable
offline command, example plan and decision states.

### Session records and optional GitHub tracking

By default, the loop saves code, results, round summaries and new learnings under its timestamped run directory. The run preserves the original config, snapshots the template/reference, and saves a reloadable `config.yaml`. The repo-root `AGENT.md` supplies existing knowledge and is left unchanged. No GitHub Issue, comment, commit, push or PR is created by default. This setting governs experiment recording; SSH/GKE evaluation still uses your configured infrastructure.

```yaml
session:
  output_dir: "runs/my_kernel"
  github_tracking: false
```

To mirror summaries to GitHub, set `github_tracking: true` and `tracking_repo: "owner/private-repo"`. The destination must be explicit; checkout origin is never a fallback. Review its visibility before sending workload data. This opt-in covers Issues, comments and Issue closure. Publishing kernels or committing learnings requires a separate user request. An explicit `agent_md_path` changes where learnings are saved locally; the default resolves to `<actual-run-dir>/AGENT.md`.

### Optimization modes

- **Step-by-step** (default) -- Pauses after each phase for human review
- **Autonomous** -- Runs the full loop unattended until max iterations or target speedup is reached

### Writing a new kernel config

Create a YAML config under `kernel-evolve/examples/` pointing to your kernel template and reference (paths are relative to the config file). Start from `matmul_v6e_ssh.yaml` for the skill loop:

```yaml
kernel:
  name: "my_kernel"
  template: "kernels/my_kernel.py"
  reference: "kernels/my_kernel_ref.py"
  evolve_markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

shapes:                     # shapes[0] is timed; every shape is checked for correctness
  - { M: 1024, N: 1024, K: 1024 }

correctness:
  method: "allclose"        # or "abs"
  atol: 1e-2
  rtol: 1e-2

evaluator:
  type: "ssh"
  ssh_hosts: ["REPLACE_WITH_WORKER_0_IP"]   # every worker of a multi-host slice
  ssh_user: "REPLACE_WITH_SSH_USER"
  ssh_key: "REPLACE_WITH_SSH_KEY_PATH"
  ssh_remote_repo: "~/dracospark"           # checkout on the VM (default)
  ssh_python: "~/tpu-venv/bin/python"       # env with JAX/libtpu and kernel-evolve (default)

session:
  max_iterations: 20
  output_dir: "runs/my_kernel"

batch:
  variants_per_round: 4
  top_k: 2
  max_active_lineages: 6
```

Mark the mutable region in your kernel template:

```python
# EVOLVE-BLOCK-START
def my_kernel(x_ref, y_ref, o_ref):
    # ... kernel code to be optimized ...
# EVOLVE-BLOCK-END
```

### Included kernel targets

| Kernel | Description | Evaluator | Config |
|--------|-------------|-----------|--------|
| `matmul` | Tiled matrix multiplication | SSH (skill loop) | `examples/matmul_v6e_ssh.yaml` |
| `matmul_pallas` | K-tiled Pallas matmul with VMEM accumulator | SSH (skill loop) | `examples/matmul_pallas.yaml` |
| `qkv_attention_pallas` | qkv self-attention with a Pallas attention core | SSH (skill loop) | `examples/qkv_attention_pallas.yaml` |
| `matmul` | Tiled matrix multiplication | GKE (`KubeEvaluator`) | `examples/matmul.yaml` |
| `gmm_fp8_blockwise` | Grouped MatMul with FP8 block-wise quantization; needs `qwix` and `tokamax` on the TPU | GKE (`KubeEvaluator`) | `examples/gmm_fp8_blockwise.yaml` |

Example configs contain `REPLACE_WITH_*` placeholders for clusters, hosts, users and keys; fill them in before running.

## Evaluation Pipeline

The evaluator runs in a configured GKE job or on a TPU VM over SSH. Its stages are:

1. **Compile** -- `exec()` the kernel, `jax.jit` + lower to HLO
2. **Correctness** -- Apply the configured `allclose` or absolute-error tolerance to each configured shape, stopping at the first failure. Shapes after a failure are unvalidated.
3. **Performance** -- Compile separately, then use the `stage_benchmark` defaults of 3 warmup calls and 5 measured calls for `shapes[0]`. Report median latency and reference latency / candidate latency as speedup. The evaluation entrypoint uses these defaults; YAML has no warmup/sample-count override.
4. **XPlane Trace** -- JAX profiler capture, compute ratio, memory transfer ratio, per-unit utilization (MXU, Scalar ALU, Vector ALU, Vector Load/Store)
5. **Deep IR** -- HLO/LLO/Mosaic dump parsing: VLIW bundles, MXU distribution, VMEM allocations, bundle density, DMA analysis, HBM bandwidth, FLOP counts, arithmetic intensity

Performance and XPlane capture share the measurement stage. When trace extraction covers the measured invocations, `timing_source` is `xprof_clustered`; otherwise it is `wallclock`. Read `metadata.benchmark.evaluation_times_ms`, `metadata.benchmark.timing_source` and `metadata.reference_benchmark` in each result. Verify `metadata.reference_perf_ok` before interpreting speedup. Candidate and reference are measured separately. A reference benchmark failure cannot establish a valid speedup comparison.

If a kernel exposes `make_inputs` and `timed_compute`, inputs are prepared before timing. The legacy `optimized_compute(**shape)` path includes input-generation work. Record the protocol when comparing results. These screening samples are not independent, unprofiled confirmation of a release/blog performance claim. Use the paired validation protocol in [regression validation](docs/regression-validation.md) for confirmation, with the target shape, runtime and correctness contract recorded.

SSH retrieves artifacts with `scp`. On GKE, artifacts are uploaded only when `GCS_BUCKET` names a bucket the pod's credentials can write to; otherwise the upload is skipped. GitHub tracking does not control artifact transport.

## Infrastructure

Dracospark does not ship with hosted infrastructure: bring your own GCP project and TPUs. Two transports are supported.

**TPU VM over SSH** (no GKE): create a Cloud TPU VM (e.g. v6e), install JAX/libtpu and `kernel-evolve` on every worker, and fill in the `ssh_*` fields of an SSH example config. See [`kernel-evolve/SSH_V6E.md`](kernel-evolve/SSH_V6E.md).

**GKE**: the reference setup is

| Component | Detail |
|-----------|--------|
| Cloud | Google Cloud Platform (your project) |
| Cluster | GKE cluster with a TPU node pool (templates select `tpu7x`, `2x2x1` topology, 4 chips; edit the `nodeSelector` for other TPUs) |
| TPU v7x peaks | 2307 TFLOPS (BF16), 3690 GB/s HBM |
| Pod identity | Kubernetes service account `kernel-eval`, bound to a GCP service account via Workload Identity |
| Artifact Storage | Optional GCS bucket passed as `GCS_BUCKET` |
| Container Registry | Optional Artifact Registry Docker repository for `kernel-evolve/docker/Dockerfile` |
| CI Auth | Workload Identity Federation for GitHub Actions |

One-time setup of the bucket, service accounts and Workload Identity binding (requires `gcloud` and `kubectl` access):

```bash
PROJECT=my-project BUCKET=my-kernel-profiles REGION=us-central1 \
  kernel-evolve/scripts/setup_gcs_profile_bucket.sh
```

Then export `GCS_BUCKET=my-kernel-profiles` in the shell that runs Claude Code so `KubeEvaluator` passes it to each Job.

### GitHub Actions variables

The workflows in `.github/workflows/` read your deployment from repository variables (Settings → Secrets and variables → Actions → Variables). Nothing is hard-coded.

| Variable | Used by | Example / meaning |
|----------|---------|-------------------|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | both | `projects/<number>/locations/global/workloadIdentityPools/<pool>/providers/<provider>` |
| `GCP_SERVICE_ACCOUNT` | both | `<name>@<project>.iam.gserviceaccount.com` used by Actions |
| `GCP_PROJECT` | `build-image.yaml` | GCP project ID that owns the Artifact Registry repository |
| `GCP_REGION` | both | Artifact Registry region and GKE cluster location, e.g. `us-central1` |
| `ARTIFACT_REGISTRY_REPO` | `build-image.yaml` | Docker repository name; the image is pushed to `<region>-docker.pkg.dev/<project>/<repo>/kernel-eval` |
| `GKE_CLUSTER` | `kernel-eval.yaml` | GKE cluster with TPU nodes |
| `GCS_PROFILE_BUCKET` | `kernel-eval.yaml` | Optional profile artifact bucket; empty skips the upload |

`kernel-eval.yaml` passes its `eval_payload` input to the Job through an environment variable, never by interpolating it into a script.

## Development

```bash
cd kernel-evolve
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .

cd ../inference-opt
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

CI runs both suites on Python 3.10 and 3.12 (`.github/workflows/`). Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues as described in [SECURITY.md](SECURITY.md).

## License

Apache 2.0 -- see [LICENSE](LICENSE).
