# Running pallas-evolve on a Cloud TPU VM (v6e) over SSH

The SSH evaluator runs the `pallas-evolve` loop against a Cloud TPU VM (for
example a v6e-16 slice) without a GKE cluster. Roofline metrics default to TPU
v6e (Trillium). You bring the TPU VM, an SSH key that can log in to every
worker, and a Python environment with JAX/libtpu on each worker.

## How the SSH evaluator works

`src/kernel_evolve/ssh_evaluator.py` provides `SSHEvaluator`, a drop-in
alternative to the GKE `KubeEvaluator`. For each batch it:

1. stages the base64 payload on every host over SSH,
2. co-launches `docker/evaluate.py` on all hosts of the slice,
3. parses `EVAL_RESULT:` lines from the primary host's stdout, and
4. copies artifacts (`llo_final.txt`, `hlo_post_opt.txt`, `trace_events.json`)
   back from the primary with `scp`.

No kubectl, ConfigMap or GCS bucket is involved. `docker/evaluate.py` reads the
roofline peaks from `PEAK_FLOPS`, `PEAK_HBM_BW`, `HBM_CAPACITY_GB` and
`VMEM_CAPACITY_MIB`, and copies artifacts to `ARTIFACTS_LOCAL_DIR` so the
evaluator can retrieve them.

## Multi-host slices (v6e-16 etc.)

A v6e-16 is **4 hosts** (4 chips each) sharing one ICI domain. libtpu does a
slice-wide init barrier, so `import jax; jax.devices()` on a single host **hangs**
waiting for the other three. Every TPU program — including `evaluate.py` and
any connectivity check — must be **co-launched on all hosts at once**.

The evaluator co-launches `evaluate.py` on all hosts and reads results and
artifacts from the **first** host (the primary). The kernel template is
single-device, so each host runs the same kernel on its local chip and prints an
identical `EVAL_RESULT:`; the other hosts run only to satisfy the barrier. List
all worker IPs under `evaluator.ssh_hosts` (the first is the primary).
Single-host slices (v6e-1/-4/-8) use `evaluator.ssh_host`.

## VM prerequisites (once, on every host)

On a multi-host slice, prepare **all** workers: each runs `evaluate.py`
locally, so each needs its own checkout and install. Discover the worker IPs
with:

```bash
gcloud compute tpus tpu-vm describe <name> --zone <zone> \
  --format="value(networkEndpoints[].accessConfig.externalIp)"
```

Then, on each worker, as the SSH user you will configure:

```bash
git clone https://github.com/Neuro-Spark-AI/dracospark.git ~/dracospark
# Use Python 3.11+ so pip can install a current jax/libtpu (with the VM's system
# Python 3.10 it falls back to an old jax). One way, with uv:
python3 -m pip install --user uv
~/.local/bin/uv python install 3.12
~/.local/bin/uv venv -p 3.12 ~/tpu-venv
# Pin the same versions on every worker, e.g. jax[tpu]==0.11.0:
VIRTUAL_ENV=~/tpu-venv ~/.local/bin/uv pip install "jax[tpu]==0.11.0" xprof
# kernel-evolve itself only needs pyyaml and pydantic:
VIRTUAL_ENV=~/tpu-venv ~/.local/bin/uv pip install -e ~/dracospark/kernel-evolve
```

Every worker of a slice must run the same commit and the same jax/libtpu
versions. If you evaluate kernels from a fork or branch, check that out instead
so the workers run the same `evaluate.py` as your local checkout. `xprof` lets
`evaluate.py` time kernels from the profiler trace; without it timing falls back
to wall clock.

Sanity-check the TPU by **co-launching on all hosts at once** (a single host
hangs):

```bash
for H in <ip0> <ip1> <ip2> <ip3>; do
  ssh -i <key> -o BatchMode=yes <user>@"$H" \
    "~/tpu-venv/bin/python -c 'import jax; print(jax.process_index(), jax.devices()[0].device_kind, len(jax.devices()))'" &
done; wait
```

## Config

Use `evaluator.type: ssh` and a `roofline` block; see
[`examples/matmul_v6e_ssh.yaml`](examples/matmul_v6e_ssh.yaml). Replace every
`REPLACE_WITH_*` value:

| Field | Meaning |
|---|---|
| `ssh_hosts` | All worker IPs, primary first (IPs change on stop/start; re-discover with gcloud) |
| `ssh_host` | Single-host slices only, instead of `ssh_hosts` |
| `ssh_user` | Login user on the workers |
| `ssh_key` | Local path to the private key for that user |
| `ssh_remote_repo` | Checkout on each worker (default `~/dracospark`) |
| `ssh_python` | Python with jax/libtpu and kernel-evolve (default `~/tpu-venv/bin/python`) |

### Using one chip of a larger slice

To evaluate on a single chip without co-launching every host (for example when
only one worker is reachable, or to keep a shared slice's other chips free), use
`ssh_host` for that worker and isolate one chip with `ssh_env`:

```yaml
evaluator:
  type: "ssh"
  ssh_host: "REPLACE_WITH_WORKER_0_IP"
  ssh_env:
    TPU_VISIBLE_CHIPS: "0"
    TPU_PROCESS_BOUNDS: "1,1,1"
    TPU_CHIPS_PER_PROCESS_BOUNDS: "1,1,1"
    TPU_PROCESS_ADDRESSES: "localhost"
    CLOUD_TPU_TASK_ID: "0"
```

These variables apply before JAX imports, so `jax.devices()` returns one device
and libtpu does not wait for the other hosts. The SSH probe in
`pallas-evolve:start` forwards the same `ssh_env`.

`ssh_remote_repo` and `ssh_python` are inserted unquoted into the remote shell
command, so a leading `~/` resolves to the SSH user's home directory on each
worker. Use absolute paths without spaces if you prefer.

## Run

- `/pallas-evolve:init-kernel <kernel> --repo <upstream-url>` — generates a v6e
  SSH config and baseline from an upstream Pallas kernel repository.
- `/pallas-evolve:start <config.yaml>` — the optimization loop (SSH connectivity
  check → think → submit over SSH → analyze → reflect).
- Standalone: `/pallas-evolve:submit` (one round), then
  `/pallas-evolve:profile-brief <dir>`.
- Scripted: `python scripts/run_ssh_batch.py --config examples/<project>.yaml ...`.

## Roofline peaks (v6e defaults, override if needed)

| Env / config | v6e default | Meaning |
|---|---|---|
| `PEAK_FLOPS` / `roofline.peak_flops` | 918e12 | bf16 FLOP/s per chip |
| `PEAK_HBM_BW` / `roofline.peak_hbm_bw` | 1759e9 | HBM bytes/s per chip |
| `HBM_CAPACITY_GB` / `roofline.hbm_capacity_gb` | 32 | HBM per chip |
| `VMEM_CAPACITY_MIB` / `roofline.vmem_capacity_mib` | 128 | physical VMEM per chip |

## Observations on v6e

- **Physical VMEM = 128 MiB** per chip. Driving a Pallas matmul to overflow
  makes XLA report "Used 131.86M of 127.94M vmem", so `vmem_capacity_mib: 128`
  is correct.
- **The default scoped VMEM budget is only 32 MiB**, not the full 128. A kernel
  that needs more than 32 MiB of VMEM fails at compile
  (`CompileTimeScopedVmemOom: limit 32.00M exceeded`). Raise it per compile with
  `compiler_options={"xla_tpu_scoped_vmem_limit_kib": "131072"}`, or globally via
  `XLA_FLAGS=--xla_tpu_scoped_vmem_limit_kib=131072`. This matters when variants
  tile large blocks into VMEM.
- **The LLO parser works on v6e.** The v6e LLO dump is standard Mosaic MLIR
  (`stable_mosaic.version = 15`), the format the `stage_profile_deep` parser
  expects, so deep-profile metrics (VLIW bundle count, MXU utilization, VMEM
  allocation, DMA analysis) are populated without a v6e-specific branch.

## Known caveats

- **`trace_events.json` is large** (~80 MB for a single matmul) and is copied
  per variant. If bandwidth or disk matters, skip it: `analyze` and
  `profile-brief` work from `llo_final.txt` and `eval_result.json` alone.
- The GCS upload path is used only by the GKE transport and only when
  `GCS_BUCKET` is set; on the SSH path artifacts travel by `scp`.
