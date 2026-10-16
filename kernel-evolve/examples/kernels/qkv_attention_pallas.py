# Example kernel: qkv self-attention with a Pallas attention core (TPU).
"""qkv self-attention with a Pallas attention core — template for evolution.

Fused-QKV projection (jnp) -> Pallas attention kernel -> output projection (jnp).
The baseline kernel is deliberately NON-flash: one program per (batch*head),
loads that head's full Q/K/V into VMEM, forms the full (S x S) score matrix,
softmaxes, and multiplies by V. Fits VMEM only for modest S (scores are S*S*4
bytes). A flash-attention tiling (loop K/V blocks with running max/sum) is the
obvious evolution direction.

The optimizer may change anything in the EVOLVE-BLOCK: the kernel (tiling,
online softmax, block sizes), the pallas_call launch config, dtypes. Frozen:
imports, _make_test_data, and the optimized_compute(B, S, H, D) signature.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _make_test_data(B=1, S=512, H=8, D=128):
  """Deterministic f32 inputs. MUST be identical to the reference's copy."""
  d_model = H * D
  kx, kq, ko = jax.random.split(jax.random.PRNGKey(0), 3)
  x = jax.random.normal(kx, (B, S, d_model), dtype=jnp.float32) * 0.1
  w_qkv = jax.random.normal(kq, (d_model, 3 * d_model), dtype=jnp.float32) * 0.02
  w_o = jax.random.normal(ko, (d_model, d_model), dtype=jnp.float32) * 0.02
  return x, w_qkv, w_o


# EVOLVE-BLOCK-START
def _attn_kernel(q_ref, k_ref, v_ref, o_ref, *, scale):
  """One (batch*head) slice. Each ref is (1, S, D); index [0] -> 2D."""
  q = q_ref[0]  # (S, D)
  k = k_ref[0]  # (S, D)
  v = v_ref[0]  # (S, D)
  scores = jnp.dot(q, k.T, preferred_element_type=jnp.float32) * scale  # (S, S)
  attn = jax.nn.softmax(scores, axis=-1)
  o_ref[0] = jnp.dot(attn, v, preferred_element_type=jnp.float32)  # (S, D)


def optimized_compute(B=1, S=512, H=8, D=128):
  x, w_qkv, w_o = _make_test_data(B, S, H, D)
  b, s, d = x.shape
  scale = 1.0 / (D**0.5)

  qkv = jnp.einsum("bsd,de->bse", x, w_qkv)
  q, k, v = jnp.split(qkv, 3, axis=-1)

  def to_heads(t):
    return t.reshape(b, s, H, D).transpose(0, 2, 1, 3)

  q, k, v = to_heads(q), to_heads(k), to_heads(v)  # (b, H, s, D)

  bh = b * H
  qf, kf, vf = (t.reshape(bh, s, D) for t in (q, k, v))
  block = pl.BlockSpec((1, s, D), lambda i: (i, 0, 0))
  out = pl.pallas_call(
    functools.partial(_attn_kernel, scale=scale),
    grid=(bh,),
    in_specs=[block, block, block],
    out_specs=block,
    out_shape=jax.ShapeDtypeStruct((bh, s, D), jnp.float32),
  )(qf, kf, vf)

  ctx = out.reshape(b, H, s, D).transpose(0, 2, 1, 3).reshape(b, s, d)  # merge heads
  return jnp.einsum("bsd,de->bse", ctx, w_o)


# EVOLVE-BLOCK-END
