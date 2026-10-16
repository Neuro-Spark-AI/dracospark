# Reference implementation for the K-tiled Pallas matmul example kernel (TPU).
"""Reference matmul (jnp.dot) for the K-tiled Pallas matmul kernel.

Ground-truth C = A @ B in float32, used as the correctness baseline for the
evolving Pallas kernel in matmul_pallas.py. Inputs come from the SAME
_make_test_data as the template so the two are compared on identical data.
"""

import jax
import jax.numpy as jnp


def _make_test_data(M=512, N=512, K=512):
  """Deterministic f32 inputs. MUST be identical to the template's copy."""
  ka, kb = jax.random.split(jax.random.PRNGKey(0))
  a = jax.random.normal(ka, (M, K), dtype=jnp.float32)
  b = jax.random.normal(kb, (K, N), dtype=jnp.float32)
  return a, b


def simple_compute(M=512, N=512, K=512):
  a, b = _make_test_data(M, N, K)
  return jnp.dot(a, b, preferred_element_type=jnp.float32)


def reference_fn(**kwargs):
  return simple_compute(**kwargs)
