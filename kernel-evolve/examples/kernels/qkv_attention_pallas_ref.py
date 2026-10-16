# Reference implementation for the qkv self-attention Pallas example kernel (TPU).
"""All-jnp reference for the qkv self-attention Pallas kernel.

Fused-QKV projection -> multi-head scaled-dot-product attention (full softmax)
-> output projection, all in plain jnp. Ground truth for the correctness check.
Inputs come from the SAME _make_test_data as the template.
"""

import jax
import jax.numpy as jnp


def _make_test_data(B=1, S=512, H=8, D=128):
  """Deterministic f32 inputs. MUST be identical to the template's copy."""
  d_model = H * D
  kx, kq, ko = jax.random.split(jax.random.PRNGKey(0), 3)
  x = jax.random.normal(kx, (B, S, d_model), dtype=jnp.float32) * 0.1
  w_qkv = jax.random.normal(kq, (d_model, 3 * d_model), dtype=jnp.float32) * 0.02
  w_o = jax.random.normal(ko, (d_model, d_model), dtype=jnp.float32) * 0.02
  return x, w_qkv, w_o


def simple_compute(B=1, S=512, H=8, D=128):
  x, w_qkv, w_o = _make_test_data(B, S, H, D)
  b, s, d = x.shape
  scale = 1.0 / (D**0.5)
  qkv = x @ w_qkv
  q, k, v = jnp.split(qkv, 3, axis=-1)

  def reshape(t):
    return t.reshape(b, s, H, D).transpose(0, 2, 1, 3)

  q, k, v = reshape(q), reshape(k), reshape(v)
  scores = (q @ k.transpose(0, 1, 3, 2)) * scale
  attn = jax.nn.softmax(scores, axis=-1)
  ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, d)
  return ctx @ w_o


def reference_fn(**kwargs):
  return simple_compute(**kwargs)
