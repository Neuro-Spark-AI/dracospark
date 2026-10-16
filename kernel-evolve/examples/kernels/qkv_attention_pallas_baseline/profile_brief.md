## Profile Brief for Round 0

### Source
- Kernel: examples/kernels/qkv_attention_pallas.py (baseline, non-flash full-score attention)
- Speedup: 0.985x vs jnp reference | Latency: 0.152 ms (ref 0.1498 ms)
- Compute ratio: 0.492 | Memory transfer ratio: 0.508
- Shape: B=1, S=1024, H=8, D=128 (roofline: TPU v6e — 918 TFLOP/s, 1759 GB/s HBM, 128 MiB VMEM)

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 26095 | high — full S×S score matrix materialized per head |
| MXU dual ratio | 0.0 (mxu0=512, mxu1=0) | poor — only one matrix unit used |
| Avg ops/bundle (ILP) | 1.21 | poor (<2) — weak VLIW packing |
| HBM bandwidth | 4.19 MB | 1.57% of peak — not HBM-bandwidth-bound |
| VMEM utilization | 3.9% of 128 MiB (5.25 MB alloc) | very low — large headroom for on-chip reuse |
| HBM capacity used | 0.09% | negligible |
| DMA transfers | 1792 | double_buffered: no |
| Register fills/spills | 0 / 0 | ideal — no register pressure |
| Pipeline NOPs | 0 | fine |
| Fusions (Pallas kernel) | 0 | ideal (single custom_call; the QKV/out projections are separate jnp einsums) |

### Bottleneck Diagnosis
**Primary bottleneck**: memory-bound, at parity with the jnp reference.
**Evidence**: compute_ratio 0.492 (<0.50) with memory_transfer_ratio 0.508; the kernel
builds the full (1024×1024) f32 score matrix per head (26095 VLIW bundles, avg 1.21
ops/bundle). VMEM is only 3.9% used, so there is no on-chip-capacity limit — the cost is
in materializing and re-reading the full score matrix, not in fitting it.
**Combined patterns**: memory-bound + single-MXU (dual_ratio 0.0) + low ILP (1.21) +
VMEM heavily underutilized (3.9%). The kernel does the math correctly but leaves both the
second MXU and ~96% of VMEM idle.

### LLO / HLO Key Observations
- 512 mxu0 ops, 0 mxu1 ops → the two matmuls (QKᵀ and attn·V) are serialized on one MXU.
- 1792 DMA transfers, no double buffering → K/V movement not overlapped with compute.
- 0 NOPs, 0 spills → schedule is clean; the loss is structural (full-score materialization), not pressure.
- HLO: 0 fusions inside the Pallas custom_call; the surrounding QKV projection and output
  projection are ordinary jnp einsums (34 fusion/custom_call matches are those XLA matmuls).

### Optimization Priorities (derived from profile)
1. **Flash-style tiling / online softmax**: the baseline materializes the full S×S score
   matrix (memory-bound, 26k bundles). Tiling K/V into blocks with a running max/sum avoids
   ever holding the full (1024×1024) scores, cutting memory traffic and bundle count. This is
   the highest-leverage change and the reason the non-flash kernel only matches jnp.
2. **Double-buffer K/V DMA**: double_buffering=false with 1792 transfers — prefetch the next
   K/V block into VMEM while computing the current one to hide DMA behind the matmuls.
3. **Use both MXUs**: dual_ratio 0.0. With tiling in place, schedule QKᵀ and attn·V (or
   independent head-tiles) so mxu0 and mxu1 run concurrently.

### What NOT to try (profile evidence)
- **Reducing block sizes / shrinking VMEM footprint**: VMEM is at 3.9% and there are 0
  register spills — there is nothing to relieve; smaller blocks only add loop overhead.
- **HBM-bandwidth tricks**: HBM bandwidth is at 1.57% of peak — the kernel is not
  HBM-bandwidth-bound, so compression/layout-for-bandwidth won't move the needle.
