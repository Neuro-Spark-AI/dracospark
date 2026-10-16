---
name: init-kernel
description: Use when initializing a new kernel optimization project from an upstream Pallas kernel repository (passed with --repo) — clones upstream, generates ref/template/test/config files, validates baseline
---

# Initialize Kernel from an Upstream Pallas Kernel Repository

Bootstrap a kernel optimization project by importing a kernel from an upstream Pallas kernel repository that you name with `--repo`. Clones the upstream repo, consolidates multi-file kernel code into self-contained files, generates a template with EVOLVE-BLOCK markers, copies tests, generates a YAML config, and validates.

### Expected upstream layout

The upstream repository must follow this layout (paths are relative to its root):

```
<package>/               # the upstream Python package (pass --package, or auto-detected)
├── ops/
│   ├── <kernel_name>/   # one directory per kernel: Pallas kernels, launchers, entry points
│   └── common/          # optional shared kernel helpers
└── utils.py             # optional repo-wide utilities
tests/                   # optional tests, e.g. tests/ops/<kernel_name>/ or test_<kernel_name>*.py
```

If the repository uses a different layout, adapt the paths below accordingly and tell the user which mapping you used.

> **Git policy**: this skill never runs `git add`, `git commit` or `git push`. When it finishes, tell the user which generated files to review and commit.

> **Evaluator path**: the generated config targets the SSH evaluator (`evaluator.type: ssh`, a Cloud TPU VM such as v6e), and baseline profiling (Step 9d) runs through `pallas-evolve:submit` over SSH. Steps 1–9c need no TPU. Before Step 9d the user must replace the `REPLACE_WITH_*` values in the generated YAML with their own worker IP(s), SSH user and key path, and prepare each worker as described in `kernel-evolve/SSH_V6E.md`. If those values are not available, stop after Step 9c and report that baseline profiling is pending.

> **CRITICAL RULE — NO REWRITING**
>
> When generating ref and template files, you MUST copy upstream function source code **verbatim**. You are an assembler, not an author. The only code you are allowed to write from scratch is:
> - `_make_test_data()` (test data generation)
> - `simple_compute()` / `optimized_compute()` (thin wrappers that call copied upstream functions)
> - `reference_fn()` (one-line delegation)
>
> Everything else — kernel functions, helper functions, custom_vjp rules, launchers — MUST be copied character-for-character from the upstream source. If a function has a bug upstream, copy the bug. If the style is inconsistent, copy it as-is. **NEVER rewrite, simplify, reimplement, or "improve" any upstream function.**

## Arguments

Expects arguments in this format:

```
/pallas-evolve:init-kernel <kernel_name> --repo <repo_url> [--ref-fn <function>] [--template-fn <function>] [--branch <branch>] [--package <dir>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `kernel_name` | Yes | — | Upstream kernel name, corresponding to the `<package>/ops/<kernel_name>/` directory |
| `--package` | No | auto-detect | Top-level package directory in the upstream repo that contains `ops/` |
| `--ref-fn` | No | auto-detect | Upstream function name to use as reference entry point (e.g. `chunk_simple_gla`). When specified, this function and all its transitive dependencies are copied verbatim into the ref file. |
| `--template-fn` | No | same as `--ref-fn` | Upstream function name to use as template entry point (e.g. `fused_chunk_simple_gla`). When different from `--ref-fn`, the template gets a different code path than the ref. |
| `--repo` | Yes | — | Upstream Pallas kernel repository URL (any `git clone` URL), e.g. `https://github.com/example-org/pallas-kernels` |
| `--branch` | No | `main` | Branch in the upstream repository |

**Parse the arguments** from the invocation. Extract:
- `KERNEL_NAME`: the first positional argument (required)
- `REF_FN`: value after `--ref-fn` flag, default `None` (auto-detect)
- `TEMPLATE_FN`: value after `--template-fn` flag, default same as `REF_FN`
- `BRANCH`: value after `--branch` flag, default `main`
- `REPO_URL`: value after `--repo` flag (required, no default)
- `PKG`: value after `--package`; if omitted, after cloning set it to the single top-level directory that contains `ops/` (stop and ask if there are none or several)
- `REPO`: `REPO_URL` with any scheme/host prefix (e.g. `https://github.com/`) and `.git` suffix removed, e.g. `example-org/pallas-kernels`; used in source headers

If `KERNEL_NAME` or `--repo` is missing, print usage and stop. Never guess or substitute a default upstream repository.

## Step 1 — Clone upstream repository

Run:

```bash
TMPDIR="/tmp/pallas-kernel-$(date +%s)"
git clone --depth 1 --branch $BRANCH $REPO_URL "$TMPDIR"
```

If the clone fails (network error, branch not found), report the error to the user with suggestions:
- Check that the branch name is correct: `git ls-remote --heads $REPO_URL`
- Check network connectivity
- **Stop here** — do not proceed without a successful clone.

Record the HEAD commit SHA for traceability:

```bash
COMMIT_SHA=$(git -C "$TMPDIR" rev-parse HEAD)
```

Store these values for use in later steps:
- `TMPDIR`: path to the cloned repo
- `COMMIT_SHA`: the exact commit
- `BRANCH`: the branch name
- `REPO_URL`: the repository URL
- `KERNEL_NAME`: the kernel name

**Important**: From this point forward, if ANY step fails, you MUST still execute Step 10 (Cleanup) to delete `$TMPDIR`.

## Step 2 — Discover and validate source files

### 2a. Find kernel source

Check that the kernel directory exists in the cloned repo:

```bash
ls "$TMPDIR/$PKG/ops/$KERNEL_NAME/"
```

If the directory does not exist:
1. List available kernel directories: `ls "$TMPDIR/$PKG/ops/"`
2. Report to user: "Kernel '$KERNEL_NAME' not found. Available kernels: [list]. Please check the kernel name."
3. **Go to Step 10** (cleanup) and stop.

Read all `.py` files in the kernel directory. These are the upstream source files that will be consolidated.

### 2b. Find upstream tests

Search for test files:

```bash
find "$TMPDIR/tests" -name "test_${KERNEL_NAME}*" -o -name "test_*${KERNEL_NAME}*" 2>/dev/null
ls "$TMPDIR/tests/ops/$KERNEL_NAME/" 2>/dev/null
```

If no test files found, warn the user: "No upstream test files found for '$KERNEL_NAME'. Will generate comparison test from kernel code, but shapes must be provided manually."

### 2c. Check for existing files

Check if any output files already exist in this repository:

```bash
ls kernel-evolve/examples/kernels/${KERNEL_NAME}_ref.py 2>/dev/null
ls kernel-evolve/examples/kernels/${KERNEL_NAME}.py 2>/dev/null
ls kernel-evolve/examples/${KERNEL_NAME}.yaml 2>/dev/null
```

If any files exist, use AskUserQuestion to ask the user: "Files already exist for '$KERNEL_NAME'. Overwrite existing files or skip?"
- If skip: **Go to Step 10** (cleanup) and stop.
- If overwrite: proceed.

## Step 2.5 — Analyze kernel structure

Read ALL `.py` files in `$TMPDIR/$PKG/ops/$KERNEL_NAME/`. For each file, analyze:

### 2.5a. Identify the main compute function

**If `--ref-fn` and/or `--template-fn` were specified**: use those function names directly. Validate that each specified function exists in at least one of the upstream `.py` files. If not found, report the error and list all top-level function names found in the upstream files so the user can correct the name.

**If not specified (auto-detect)**: look for the primary entry-point function. Search patterns (in priority order):

1. A function called from the upstream test files (most reliable)
2. A function decorated with `@jax.custom_vjp` (indicates a differentiable kernel)
3. A function matching the kernel name (e.g., `def chunk_gla(...)`, `def gla(...)`)
4. A function named `forward`, `compute`, or `main`

Record the function name(s) and their full signatures (parameter names and defaults).

### 2.5b. Check for custom_vjp

Search for `@jax.custom_vjp` or `@custom_vjp` in the kernel code. If found, note that the kernel has explicit forward/backward implementations. The `optimized_compute` wrapper must call the `custom_vjp`-decorated function.

### 2.5c. Identify external dependencies

Scan all import statements. Classify each import as:

- **Standard library**: `os`, `sys`, `functools`, etc. — OK
- **JAX ecosystem**: `jax`, `jax.numpy`, `jax.experimental.pallas`, `flax`, etc. — OK
- **External**: anything else (e.g., `qwix`, `tokamax`, `torch`, etc.) — WARN

If external dependencies are found:
1. Warn the user: "External dependencies detected: [list]. These must be available in the evaluation environment."
2. Note them for the YAML config. On the SSH path they must be installed into the `ssh_python` environment on every TPU worker. (Only if the user evaluates on GKE instead would a custom job template be needed — see `.github/ci/kernel-eval-gmm-job.yaml` for a template that installs extra packages.)

### 2.5d. Map function signature to shape_dict pattern

evaluate.py calls compute functions as `fn(**shape_dict)` where `shape_dict` is a dict like `{"M": 1024, "N": 1024}`. The upstream function must accept these as keyword arguments.

If the upstream function signature is `def compute(M, N, K)` → direct match, use as-is.
If the upstream function signature is `def compute(q, k, v, g)` → the wrapper (`simple_compute`/`optimized_compute`) must create tensors from shape params and call the inner function.

Determine how to write the wrapper function. Reference `kernel-evolve/examples/kernels/matmul_ref.py` (shape params map directly) and `kernel-evolve/examples/kernels/qkv_attention_pallas_ref.py` (wrapper builds tensors via `_make_test_data`) for examples of both patterns.

### 2.5e. Map multi-file structure

Determine the dependency graph between files in `$PKG/ops/$KERNEL_NAME/`:
- Which file imports from which other file?
- What is the topological order for inlining?

Plan the consolidation order: inline leaf dependencies first, then files that depend on them, building up to the main entry-point file.

### 2.5f. Complexity gate

If the upstream code meets ANY of these conditions:
- More than 5 source files in `$PKG/ops/$KERNEL_NAME/`
- External dependencies that cannot be pip-installed
- Deeply nested dependency chains (>3 levels of internal imports)

Then: inform the user that automated consolidation may be incomplete and generate a scaffold with `TODO` markers for sections that need manual review.

## Step 3 — Generate reference kernel

### 3a. Preserve upstream source

Copy the upstream source files unmodified for traceability:

```bash
mkdir -p kernel-evolve/upstream/$KERNEL_NAME
cp -r "$TMPDIR/$PKG/ops/$KERNEL_NAME/"* kernel-evolve/upstream/$KERNEL_NAME/
```

These files are for reference only — they are NOT used by evaluate.py.

### 3b. Generate self-contained ref file

**Critical**: evaluate.py loads kernel files via `exec(kernel_code, exec_globals)`. The ref file MUST be fully self-contained — no relative imports, no imports from adjacent directories.

**Critical**: You MUST NOT rewrite any upstream code. The consolidation process is purely mechanical: copy function source text, resolve imports by inlining, deduplicate. See the "NO REWRITING" rule at the top of this document.

Create `kernel-evolve/examples/kernels/${KERNEL_NAME}_ref.py` by **copying** upstream functions into a single file via the following procedure:

#### 3b-i. Identify the ref entry-point function

Use the `REF_FN` identified in Step 2.5a (either from `--ref-fn` or auto-detected). This is the function that will be called by `simple_compute()`.

#### 3b-ii. Trace transitive dependencies

Starting from the entry-point function, recursively find ALL functions and classes it depends on:

1. Read the source file containing the entry-point function
2. Find all function/class calls within its body (e.g., `fused_chunk_fwd(...)`, `exp(...)`, `chunk_fwd_h(...)`)
3. For each called function/class, locate its definition in the upstream source files. Search scope:
   - `$TMPDIR/$PKG/ops/$KERNEL_NAME/*.py` (kernel-specific files)
   - `$TMPDIR/$PKG/ops/common/*.py` (shared utilities)
   - `$TMPDIR/$PKG/utils.py` (repo-wide utilities)
4. Recursively trace each dependency's dependencies
5. Build a topologically-sorted list of all functions/classes needed

Skip standard library and JAX functions (e.g., `jnp.dot`, `lax.scan`, `functools.partial`) — these come from imports.

#### 3b-iii. Copy functions verbatim

For each function/class in the topological order (leaves first):

1. **Copy the exact source text** of the function definition — from `def func_name(` (or `class ClassName:`) through the end of its body, including all decorators (e.g., `@functools.partial(jax.custom_vjp, ...)`)
2. **Do NOT modify the function body in any way** — no renaming, no simplification, no "equivalent" reimplementation
3. Add a comment above each copied block indicating its source file: `# From: $PKG/ops/{path}`

#### 3b-iv. Assemble the output file

```python
# Source: {REPO} @ branch: {BRANCH}
# Commit: {COMMIT_SHA}
# Initialized: {TODAY's date YYYY-MM-DD}
"""Reference {KERNEL_NAME} kernel from {REPO}.

Upstream code copied verbatim into a single self-contained file.
Used as the correctness baseline for evolutionary optimization.
"""

# --- All imports (from all upstream files, deduplicated) ---
import functools
import jax
import jax.numpy as jnp
# ... (collect ALL import statements from all upstream files that
#      contributed functions, deduplicate, remove relative imports
#      like `from .utils import exp` since the code is now inlined)

# --- Copied upstream code (in dependency order, leaves first) ---

# From: $PKG/ops/utils.py
def exp(x):
    ...  # EXACT copy from upstream

# From: $PKG/ops/common/chunk_h.py
def _chunk_fwd_h_kernel(...):
    ...  # EXACT copy from upstream

# ... (all other dependencies)

# From: $PKG/ops/{KERNEL_NAME}/chunk.py
def chunk_simple_gla(...):
    ...  # EXACT copy of the entry-point function

# --- Private helpers (NEW code — only these are written by you) ---

def _make_test_data({shape_params_with_defaults}):
    """Create deterministic test data for the kernel."""
    # Generate input tensors using fixed PRNG seeds
    # Match the upstream test data generation pattern
    ...


# --- Public API for evaluate.py (NEW code — thin wrappers only) ---

def simple_compute({shape_params_with_defaults}):
    """Run the reference kernel and return a scalar loss for correctness comparison."""
    # Create test data via _make_test_data
    # Call the COPIED upstream entry-point function
    # Return a scalar (e.g., jnp.sum of the output)
    ...


def reference_fn(**kwargs):
    """Generic entry point for evaluate.py function discovery."""
    return simple_compute(**kwargs)
```

#### 3b-v. Allowed vs. prohibited transformations

**ALLOWED** (mechanical, no logic changes):
- Remove relative import lines (`from .utils import exp`, `from ..common import chunk_h`) — the imported code is now inlined above
- Deduplicate stdlib/JAX import statements at the top
- Add `# From:` source comments above each copied block
- Write `_make_test_data()`, `simple_compute()`, `reference_fn()` from scratch — these are new glue code

**PROHIBITED** (these will cause correctness bugs):
- Rewriting a function body to use different operations (e.g., replacing `pallas_call` with `lax.scan`)
- Simplifying or "cleaning up" upstream code
- Renaming internal variables or functions
- Changing precision annotations, dtype casts, or accumulator types
- Omitting functions you consider "unnecessary" — if the entry point calls it, include it
- Writing "equivalent" implementations instead of copying

Reference existing files for the pattern:
- Simple case: Read `kernel-evolve/examples/kernels/matmul_ref.py`
- Complex case: Read `kernel-evolve/examples/kernels/gmm_fp8_blockwise_ref.py`

## Step 4 — Generate template kernel

Create `kernel-evolve/examples/kernels/${KERNEL_NAME}.py` using the same **copy-based** consolidation process as Step 3b, but with the `TEMPLATE_FN` entry point (from `--template-fn`, or the same as `REF_FN` if not specified).

**Critical**: The same "NO REWRITING" rule applies. Copy upstream code verbatim. Do not reimplement.

### When `TEMPLATE_FN` == `REF_FN` (default)

The template file contains the **same copied upstream code** as the ref file, plus EVOLVE-BLOCK markers. The only difference is:
- Ref has `simple_compute()` wrapper → Template has `optimized_compute()` wrapper
- Template has `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers

### When `TEMPLATE_FN` != `REF_FN`

The template file contains a **different set of copied upstream code** (traced from `TEMPLATE_FN`). For example:
- Ref uses `chunk_simple_gla` (non-fused, scan-based) as entry point
- Template uses `fused_chunk_simple_gla` (fused Pallas kernels) as entry point

Both are copied verbatim from upstream — just different functions.

### Consolidation procedure

Follow the exact same Steps 3b-i through 3b-v, substituting:
- `TEMPLATE_FN` instead of `REF_FN` as the entry point
- `optimized_compute` instead of `simple_compute` as the wrapper name
- Add `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers around the copied code

### EVOLVE-BLOCK placement rules

The EVOLVE-BLOCK boundary determines what the optimizer can change. **Incorrect placement will break the evaluation pipeline.**

**OUTSIDE the EVOLVE-BLOCK** (frozen — optimizer cannot touch):
- All `import` statements
- `_make_test_data()` function
- Constants and configuration values

**INSIDE the EVOLVE-BLOCK** (mutable — optimizer will modify):
- All copied upstream kernel functions (Pallas kernels, launchers, custom_vjp wrappers, helpers)
- `optimized_compute` function

### Template file structure

```python
"""{KERNEL_NAME} Pallas TPU kernel — template for evolutionary optimization.

{Description from upstream docstrings}

Optimization targets within the EVOLVE-BLOCK:
  - Kernel fusion strategies
  - Block sizes and tiling
  - Memory layout and transpose strategies
  - Loop structure and pipelining
  - Grid dimensions and BlockSpec configurations
  - Accumulator precision choices

Reference dimensions from upstream:
  {shape info, e.g.: q, k, v: [2, 4096, 16, 128]}
"""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
# ... (all imports from the copied upstream code, deduplicated)


def _make_test_data({shape_params_with_defaults}):
    """Create deterministic test data for the kernel."""
    # MUST be identical to the _make_test_data in the ref file
    ...


# EVOLVE-BLOCK-START

# --- Copied upstream code (verbatim, same process as Step 3b) ---

# From: $PKG/ops/utils.py
def exp(x):
    ...  # EXACT copy

# From: $PKG/ops/{KERNEL_NAME}/fused_chunk.py
def _fused_chunk_fwd_kernel(...):
    ...  # EXACT copy

# ... (all dependencies, then entry point)

def optimized_compute({shape_params_with_defaults}):
    """Run the optimized kernel and return a scalar loss."""
    # Create test data, call the copied upstream entry-point, return scalar
    ...
# EVOLVE-BLOCK-END
```

Key rules:
1. **`_make_test_data` MUST be identical** in both ref and template — same PRNG seeds, same shapes, same dtypes.
2. **`optimized_compute` signature MUST match `simple_compute`** — same parameter names and defaults.
3. **All upstream code inside EVOLVE-BLOCK is copied verbatim** — it will diverge during optimization, but starts as an exact copy.
4. **Include Pallas imports** (`pl`, `pltpu`) even if the copied code doesn't use them yet — the optimizer will need them.

Reference existing files for the pattern:
- Simple case: Read `kernel-evolve/examples/kernels/matmul.py`
- Complex case: Read `kernel-evolve/examples/kernels/gmm_fp8_blockwise.py`

## Step 5 — Copy upstream test cases

If upstream test files were found in Step 2b:

1. Read each upstream test file
2. Create `kernel-evolve/tests/test_${KERNEL_NAME}.py` with adapted content:
   - Replace upstream import paths (e.g., `from <package>.ops.<kernel> import chunk`) with the local exec-based pattern or direct imports
   - Add convention tests following this pattern:

```python
"""Tests for {KERNEL_NAME} kernel conventions and basic correctness."""

import ast
import pathlib

import pytest

from kernel_evolve.config import load_config
from kernel_evolve.mutation import extract_evolve_block

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def test_template_syntax():
  ast.parse((EXAMPLES / "kernels/{KERNEL_NAME}.py").read_text())


def test_reference_syntax():
  ast.parse((EXAMPLES / "kernels/{KERNEL_NAME}_ref.py").read_text())


def test_evolve_block_extractable():
  code = (EXAMPLES / "kernels/{KERNEL_NAME}.py").read_text()
  block = extract_evolve_block(code)
  assert len(block) > 100
  assert "optimized_compute" in block


def test_template_has_optimized_compute():
  ns = {}
  exec((EXAMPLES / "kernels/{KERNEL_NAME}.py").read_text(), ns)
  assert callable(ns.get("optimized_compute"))


def test_reference_has_simple_compute():
  ns = {}
  exec((EXAMPLES / "kernels/{KERNEL_NAME}_ref.py").read_text(), ns)
  assert callable(ns.get("simple_compute"))


def test_reference_has_reference_fn():
  ns = {}
  exec((EXAMPLES / "kernels/{KERNEL_NAME}_ref.py").read_text(), ns)
  assert callable(ns.get("reference_fn"))


def test_matching_test_data():
  """Both template and ref must generate identical test data."""
  tmpl_ns = {}
  ref_ns = {}
  exec((EXAMPLES / "kernels/{KERNEL_NAME}.py").read_text(), tmpl_ns)
  exec((EXAMPLES / "kernels/{KERNEL_NAME}_ref.py").read_text(), ref_ns)
  # Call _make_test_data with small dimensions and compare
  # Use reduced shapes to keep test fast (no TPU needed for data generation)
  {small_shape_args}  # e.g., M=64, N=64
  tmpl_data = tmpl_ns["_make_test_data"]({small_shape_call})
  ref_data = ref_ns["_make_test_data"]({small_shape_call})
  # Compare all returned tensors
  import jax.numpy as jnp
  # If _make_test_data returns a tuple, compare element-wise
  # If it returns a single tensor, compare directly


def test_config_loads():
  config = load_config(EXAMPLES / "{KERNEL_NAME}.yaml")
  assert config.kernel.name == "{KERNEL_NAME}"
  assert len(config.shapes) >= 1
```

3. Preserve any upstream-specific tests that verify kernel-specific behavior (e.g., shape assertions, gradient checks)
4. Extract input shapes from upstream tests for use in Step 7 (YAML config generation)

If no upstream test files were found, generate only the convention tests above using the kernel analysis from Step 2.5.

## Step 6 — Generate standalone accuracy comparison test

Create `kernel-evolve/tests/standalone_${KERNEL_NAME}_test.py` following this pattern:

```python
"""Standalone integration test for {KERNEL_NAME} kernel.

Run on TPU, from the kernel-evolve/ directory:
    python tests/standalone_{KERNEL_NAME}_test.py

Verifies:
  1. Template and reference outputs have the same shape
  2. NaN and unequal infinite values fail; equal infinities compare as equal
  3. Output differences satisfy the configured correctness method and tolerances
  4. Measures forward+backward latency
"""

import time
import sys
import numpy as np

import jax
import jax.numpy as jnp


METHOD = "allclose"
RTOL = 1e-2
ATOL = 1.0


def absolute_difference(candidate, reference):
  """Return a reporting difference and an exact comparison difference."""
  if candidate.dtype.kind in "iu" and reference.dtype.kind in "iu":
    if max(candidate.dtype.itemsize, reference.dtype.itemsize) <= 4:
      exact = np.abs(candidate.astype(np.int64) - reference.astype(np.int64))
    else:
      exact = np.abs(candidate.astype(object) - reference.astype(object))
    return np.asarray(exact, dtype=np.float64), exact
  diff = np.abs(np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
  return diff, diff


def main():
  print(f"JAX devices: {jax.devices()}")
  print(f"Platform: {jax.default_backend()}")

  # Load kernels via exec (same as evaluate.py)
  tmpl_ns = {}
  ref_ns = {}
  exec(open("examples/kernels/{KERNEL_NAME}.py").read(), tmpl_ns)
  exec(open("examples/kernels/{KERNEL_NAME}_ref.py").read(), ref_ns)

  shapes = {SHAPES_DICT}  # e.g., {"M": 1024, "N": 1024, "K": 1024}
  print(f"\nShapes: {shapes}")

  # Correctness
  print("\n--- Correctness ---")
  tmpl_out = tmpl_ns["optimized_compute"](**shapes)
  ref_out = ref_ns["simple_compute"](**shapes)
  jax.block_until_ready(tmpl_out)
  jax.block_until_ready(ref_out)

  tmpl_arr = np.asarray(tmpl_out)
  ref_arr = np.asarray(ref_out)
  if tmpl_arr.shape != ref_arr.shape:
    print(f"FAIL: output shapes differ: {tmpl_arr.shape} != {ref_arr.shape}")
    sys.exit(1)
  if (tmpl_arr.dtype.kind in "iu") != (ref_arr.dtype.kind in "iu"):
    print(f"FAIL: output dtypes are not comparable: {tmpl_arr.dtype} vs {ref_arr.dtype}")
    sys.exit(1)

  tmpl_f = np.asarray(tmpl_arr, dtype=np.float64)
  ref_f = np.asarray(ref_arr, dtype=np.float64)
  diff, comparison_diff = absolute_difference(tmpl_arr, ref_arr)
  finite = np.isfinite(diff)
  if METHOD == "abs":
    bound = np.float64(ATOL)
  elif METHOD == "allclose":
    bound = ATOL + RTOL * np.abs(ref_f)
  else:
    raise ValueError(f"unknown correctness method: {METHOD!r}")

  finite_diff = np.where(finite, diff, 0.0)
  max_diff = float(np.max(finite_diff)) if diff.size else 0.0
  nonfinite_mismatch = bool(np.any(~finite & (tmpl_f != ref_f)))
  cast_collision = bool(np.any((diff == 0) & (tmpl_arr != ref_arr)))
  outside_tolerance = not bool(np.all(np.where(finite, comparison_diff <= bound, True)))
  print(f"Template output: shape={tmpl_arr.shape}, dtype={tmpl_arr.dtype}")
  print(f"Reference output: shape={ref_arr.shape}, dtype={ref_arr.dtype}")
  print(f"Max diff: {max_diff:.6e}")

  if nonfinite_mismatch or cast_collision or outside_tolerance:
    print(f"FAIL: correctness check failed under method={METHOD}, rtol={RTOL}, atol={ATOL}")
    sys.exit(1)
  print("PASS: correctness within tolerance")

  # Performance
  print("\n--- Performance ---")
  warmup = 10
  iters = 50

  for name, fn in [("template", tmpl_ns["optimized_compute"]), ("reference", ref_ns["simple_compute"])]:
    for _ in range(warmup):
      out = fn(**shapes)
      jax.block_until_ready(out)

    times = []
    for _ in range(iters):
      t0 = time.perf_counter()
      out = fn(**shapes)
      jax.block_until_ready(out)
      times.append(time.perf_counter() - t0)

    median_ms = np.median(times) * 1000
    print(f"{name}: {median_ms:.2f} ms (median of {iters})")

  print("\nDone.")


if __name__ == "__main__":
  main()
```

Replace `{SHAPES_DICT}` with the primary shape dict extracted from upstream tests (Step 5).
Keep the `METHOD`, `RTOL`, and `ATOL` constants identical to the values written under `correctness` in Step 7. The defaults are `allclose`, `1e-2`, and `1.0`.

## Step 7 — Generate YAML config

Create `kernel-evolve/examples/${KERNEL_NAME}.yaml`:

```yaml
# Source: {REPO} @ branch: {BRANCH}
# Commit: {COMMIT_SHA}
# Initialized: {TODAY}

kernel:
  name: "{KERNEL_NAME}"
  template: "kernels/{KERNEL_NAME}.py"
  reference: "kernels/{KERNEL_NAME}_ref.py"
  evolve_markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

shapes:
{SHAPES_YAML}

correctness:
  method: "allclose"
  rtol: 1e-2
  atol: 1.0

evaluator:
  type: "ssh"
  # Multi-host slice (e.g. v6e-16 = 4 hosts): list EVERY worker's external IP.
  # evaluate.py is co-launched on all of them (they share one ICI domain and
  # block on a slice-wide libtpu barrier); results/artifacts come from the first.
  # Discover with:
  #   gcloud compute tpus tpu-vm describe <name> --zone <zone> \
  #     --format="value(networkEndpoints[].accessConfig.externalIp)"
  ssh_hosts:                                # IPs go stale on stop/start
    - "REPLACE_WITH_WORKER_0_IP"
    - "REPLACE_WITH_WORKER_1_IP"
    - "REPLACE_WITH_WORKER_2_IP"
    - "REPLACE_WITH_WORKER_3_IP"
  # Single-host slice (v6e-1/-4/-8): drop ssh_hosts and set ssh_host instead:
  # ssh_host: "REPLACE_WITH_WORKER_IP"
  ssh_user: "REPLACE_WITH_SSH_USER"
  ssh_key: "REPLACE_WITH_PATH_TO_SSH_PRIVATE_KEY"
  ssh_remote_repo: "~/dracospark"            # checkout on each VM ("~/" = remote home)
  ssh_python: "~/tpu-venv/bin/python"        # python env with jax/libtpu + kernel-evolve
  ssh_remote_tmp: "/tmp/kernel_eval"
  ssh_artifacts_dir: "/tmp/kernel_eval_artifacts"
  timeout: 1800

roofline:                     # TPU v6e (Trillium) peaks
  peak_flops: 918.0e12
  peak_hbm_bw: 1759.0e9
  hbm_capacity_gb: 32.0
  vmem_capacity_mib: 128.0

batch:
  variants_per_round: 5
  top_k: 2
  max_active_lineages: 4

session:
  max_iterations: 20
  output_dir: "runs/{KERNEL_NAME}"
  # GitHub tracking is opt-in and off by default. To mirror round summaries:
  # github_tracking: true
  # tracking_repo: "owner/private-repo"
```

Leave the `REPLACE_WITH_*` placeholders for the user to fill in; never guess hosts, users or key paths.

**Shapes**: Extract from upstream tests (Step 5). Format as YAML list of dicts:
```yaml
shapes:
  - { M: 1024, N: 1024, K: 1024 }
  - { M: 2048, N: 2048, K: 2048 }
```

If shapes could not be extracted from upstream tests, use AskUserQuestion to ask the user to provide them.

**External dependencies**: If Step 2.5c detected external dependencies:
- Add a comment in the YAML noting the dependencies:
  ```yaml
  # External dependencies: qwix, tokamax
  # Install into evaluator.ssh_python on every TPU worker before Step 9d.
  ```
- Tell the user to install them into the `ssh_python` environment on every worker (e.g. `~/tpu-venv/bin/pip install <packages>`). Do not install anything on remote hosts yourself unless the user asks.

## Step 7.5 — Compact context

All source files have been generated and the upstream code has been consolidated. The following files are on disk:
- `kernel-evolve/examples/kernels/${KERNEL_NAME}_ref.py` — self-contained reference
- `kernel-evolve/examples/kernels/${KERNEL_NAME}.py` — template with EVOLVE-BLOCK
- `kernel-evolve/examples/${KERNEL_NAME}.yaml` — YAML config
- `kernel-evolve/upstream/${KERNEL_NAME}/` — unmodified upstream source
- `kernel-evolve/tests/test_${KERNEL_NAME}.py` — pytest convention tests
- `kernel-evolve/tests/standalone_${KERNEL_NAME}_test.py` — TPU integration test

Invoke `/compact` to compress conversation context before proceeding to validation. The upstream source code read during Steps 2–3 and the intermediate analysis are no longer needed in context — validation steps (8–9) will read generated files from disk.

## Step 8 — Validate generated config

Validate the generated YAML config against the Pydantic schema:

```bash
cd kernel-evolve && python -c "
from kernel_evolve.config import load_config
config = load_config('examples/${KERNEL_NAME}.yaml')
print(f'Config validated: kernel={config.kernel.name}, shapes={config.shapes}')
print(f'Template: {config.kernel.template}')
print(f'Reference: {config.kernel.reference}')
"
```

If validation fails:
1. Read the error message
2. Fix the YAML file
3. Re-run validation
4. If still failing after 3 attempts, report the error to the user and pause

## Step 9 — Verify baseline

### 9a. Syntax and import verification

Verify both generated kernel files have valid Python syntax:

```bash
cd kernel-evolve && python -c "
import ast
# Check ref
ref_code = open('examples/kernels/${KERNEL_NAME}_ref.py').read()
ast.parse(ref_code)
print('ref: syntax OK')

# Check template
tmpl_code = open('examples/kernels/${KERNEL_NAME}.py').read()
ast.parse(tmpl_code)
print('template: syntax OK')

# Check function discovery (same as evaluate.py)
ref_ns = {}
exec(ref_code, ref_ns)
assert callable(ref_ns.get('simple_compute')), 'ref missing simple_compute'
assert callable(ref_ns.get('reference_fn')), 'ref missing reference_fn'
print('ref: functions OK')

tmpl_ns = {}
exec(tmpl_code, tmpl_ns)
assert callable(tmpl_ns.get('optimized_compute')), 'template missing optimized_compute'
print('template: functions OK')

# Check _make_test_data exists in both
assert callable(ref_ns.get('_make_test_data')), 'ref missing _make_test_data'
assert callable(tmpl_ns.get('_make_test_data')), 'template missing _make_test_data'
print('both: _make_test_data OK')

print('All checks passed.')
"
```

If any check fails, fix the generated file and re-run.

### 9b. Run pytest convention tests

```bash
cd kernel-evolve && python -m pytest tests/test_${KERNEL_NAME}.py -v
```

If tests fail, fix the generated files and re-run.

### 9c. Evolve-block extraction verification

```bash
cd kernel-evolve && python -c "
from kernel_evolve.mutation import extract_evolve_block
code = open('examples/kernels/${KERNEL_NAME}.py').read()
block = extract_evolve_block(code)
print(f'EVOLVE-BLOCK: {len(block)} chars extracted')
assert 'optimized_compute' in block, 'optimized_compute not in evolve block'
print('EVOLVE-BLOCK content OK')
"
```

### 9d. Baseline profiling (Round 0)

Verify the template kernel compiles and runs on TPU, collect baseline performance metrics and profiling artifacts.

This step can also be run on its own to create a missing baseline for an existing SSH config (e.g. `examples/matmul_pallas.yaml`): use the config's `kernel.name` as `KERNEL_NAME` (the baseline directory is `examples/kernels/{kernel.name}_baseline/`, which is what `pallas-evolve:start` looks for) and copy the file named by its `kernel.template` instead of `examples/kernels/${KERNEL_NAME}.py`.

1. **Verify SSH connectivity to the TPU-VM(s)**:

   If the YAML still contains `REPLACE_WITH_*` values, stop here (see the Evaluator path note at the top).

   Single-host slice:
   ```bash
   ssh -i {ssh_key} {ssh_user}@{host} "env {ssh_env_assignments} timeout 120 {ssh_python} -c 'import jax; print(jax.devices())'"
   ```
   (`{ssh_env_assignments}`: the shell-quoted `evaluator.ssh_env` entries, or empty.)

   Multi-host slice (`ssh_hosts` with >1 entry): a single-host `jax.devices()`
   hangs on the ICI barrier. Co-launch the check on every host at once and wait
   for all (see the `pallas-evolve:start` skill Step 6 for the loop). Each host
   should print `TPU v6 lite` and the global device count.

   If unreachable or it hangs, **stop with error**: "TPU connectivity required
   for baseline profiling. Check every IP in evaluator.ssh_hosts (worker IPs go
   stale — re-discover all workers with gcloud) and evaluator.ssh_key, then
   re-run."

2. **Create baseline directory**:

   ```bash
   mkdir -p kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/
   cp kernel-evolve/examples/kernels/${KERNEL_NAME}.py \
      kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/kernel.py
   ```

3. **Set up temporary iteration structure for submit**: The `pallas-evolve:submit` skill expects an iteration directory with `variants/*/kernel.py`. Create a temporary structure:

   ```bash
   BASELINE_TMPDIR=$(mktemp -d)
   mkdir -p "${BASELINE_TMPDIR}/iteration_0/variants/baseline/"
   cp kernel-evolve/examples/kernels/${KERNEL_NAME}.py \
      "${BASELINE_TMPDIR}/iteration_0/variants/baseline/kernel.py"
   ```

4. **Submit baseline for TPU evaluation**: Invoke `pallas-evolve:submit` via the Skill tool, pointing at the temporary iteration directory, with `kernel-evolve/examples/${KERNEL_NAME}.yaml` as the config. This will:
   - Submit the baseline kernel as a single-variant batch
   - Collect `eval_result.json` with performance metrics and deep profiling data
   - scp `llo_final.txt`, `hlo_post_opt.txt`, `trace_events.json` back from the VM

5. **Copy results to permanent baseline directory**:

   ```bash
   cp "${BASELINE_TMPDIR}/iteration_0/variants/baseline/"* \
      kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/
   rm -rf "${BASELINE_TMPDIR}"
   ```

6. **Check for compilation failure**: Read `kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/eval_result.json`. If `status` is `COMPILE_ERROR`, **stop and report the error** — the template kernel itself is broken and must be fixed before optimization can begin.

7. **Generate baseline profile brief**: Invoke `pallas-evolve:profile-brief` via the Skill tool:

   ```
   /pallas-evolve:profile-brief kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/ --round 0
   ```

   This writes `profile_brief.md` into the baseline directory.

8. **Hand off baseline artifacts (no automatic commit)**: Do not run `git add`, `git commit` or `git push`. List the files in `kernel-evolve/examples/kernels/${KERNEL_NAME}_baseline/` in the summary (Step 10b) and tell the user to review and commit them. Before committing, the user should check the IR dumps and `eval_result.json` for absolute paths, hostnames or bucket names they do not want to publish.

## Step 10 — Cleanup and summary

### 10a. Delete temporary clone

**This step MUST execute regardless of whether previous steps succeeded or failed.**

```bash
rm -rf "$TMPDIR"
```

### 10b. Output summary

Print a summary of all generated files:

```
=== init-kernel complete: {KERNEL_NAME} ===

Source: {REPO} @ {BRANCH} ({COMMIT_SHA})

Generated files:
  kernel-evolve/examples/{KERNEL_NAME}.yaml              — config
  kernel-evolve/examples/kernels/{KERNEL_NAME}.py         — template (with EVOLVE-BLOCK)
  kernel-evolve/examples/kernels/{KERNEL_NAME}_ref.py     — reference (self-contained)
  kernel-evolve/examples/kernels/{KERNEL_NAME}_baseline/  — baseline profiling artifacts
  kernel-evolve/upstream/{KERNEL_NAME}/                   — unmodified upstream source
  kernel-evolve/tests/test_{KERNEL_NAME}.py               — pytest convention tests
  kernel-evolve/tests/standalone_{KERNEL_NAME}_test.py    — TPU integration test

Nothing has been committed. Review the generated files above and commit the
ones you want to keep (check the baseline IR dumps for local paths first).

Next step:
  /pallas-evolve:start {KERNEL_NAME}.yaml
```
