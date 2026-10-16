# Example kernel: K-tiled Pallas matmul (TPU).
"""K-tiled Pallas matmul (VMEM scratch accumulator) — template for evolution.

Same math as jnp.dot (C = A @ B, f32), hand-tiled as a Pallas kernel: the grid
is (M/BM, N/BN, K/BK); each program computes one (BM x BN) output tile by
looping the K grid axis, accumulating partials in a VMEM scratch buffer, and
flushing on the last K step (the grid runs sequentially so the accumulator
persists).

The optimizer may change anything in the EVOLVE-BLOCK: block sizes, grid,
BlockSpecs, the accumulator/scratch strategy, pipelining, dtypes. Frozen:
imports, _make_test_data, and the optimized_compute(M, N, K) signature.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _make_test_data(M=512, N=512, K=512):
  """Deterministic f32 inputs. MUST be identical to the reference's copy."""
  ka, kb = jax.random.split(jax.random.PRNGKey(0))
  a = jax.random.normal(ka, (M, K), dtype=jnp.float32)
  b = jax.random.normal(kb, (K, N), dtype=jnp.float32)
  return a, b


# EVOLVE-BLOCK-START
BM, BN, BK = 256, 256, 256


def _matmul_kernel(a_ref, b_ref, o_ref, acc_ref):
  """Runs once per grid point. a_ref:(BM,BK) b_ref:(BK,BN) o_ref:(BM,BN)."""
  k = pl.program_id(2)  # position along the K (reduction) grid axis

  @pl.when(k == 0)
  def _init():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  # Accumulate this K-slice's partial product into VMEM scratch.
  acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

  @pl.when(k == pl.num_programs(2) - 1)
  def _flush():
    o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def optimized_compute(M=512, N=512, K=512):
  a, b = _make_test_data(M, N, K)
  grid = (M // BM, N // BN, K // BK)
  return pl.pallas_call(
    _matmul_kernel,
    grid=grid,
    in_specs=[
      # A tile depends on (i, k); B tile depends on (k, j).
      pl.BlockSpec((BM, BK), lambda i, j, k: (i, k)),
      pl.BlockSpec((BK, BN), lambda i, j, k: (k, j)),
    ],
    out_specs=pl.BlockSpec((BM, BN), lambda i, j, k: (i, j)),
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    # VMEM scratch accumulator, persists across the sequential grid loop.
    scratch_shapes=[pltpu.VMEM((BM, BN), jnp.float32)],
  )(a, b)


# EVOLVE-BLOCK-END
