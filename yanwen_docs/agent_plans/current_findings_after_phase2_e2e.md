# Current Findings After Phase 2 E2E Storage Study

Phase 2 is complete. The current source report is:

```text
yanwen_docs/agent_reports/real_llama_e2e_storage_study.md
```

The generated browser report is:

```text
yanwen_docs/agent_results/real_llama_e2e_storage_study/analysis/phase2_real_llama_e2e_web_report.html
```

It is also copied under the Phase 1 served directory:

```text
yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/analysis/phase2/
```

## Main Result

Texture3D linear coopmat transfers to E2E on real LLaMA at the measured size.

Real LLaMA 3.1 8B, 4 layers, fp16, seq=256:

| Variant | Steady mean | Speedup vs texture Stephen |
| --- | ---: | ---: |
| texture Stephen | 442.63 ms | 1.00x |
| texture coopmat | 250.43 ms | 1.77x |
| buffer disable coopmat | 393.94 ms | 1.12x |
| buffer coopmat | 205.29 ms | 2.16x |

Synthetic one-block LLaMA-shaped fp16 seq=256:

| Variant | Steady mean | Speedup vs texture Stephen |
| --- | ---: | ---: |
| texture Stephen | 93.74 ms | 1.00x |
| texture coopmat | 46.53 ms | 2.01x |

## Routing

For real LLaMA texture3D `.pte`:

- 28 prefill linears route to `linear_coopmat_texture3d_buffer` when texture
  coopmat is enabled.
- The LM head routes to `linear_vec_buffer_buffer_half` because `M=1` and
  `N=128256` exceeds the texture width limit.
- Attention BMMs remain on `matmul_vec_texture3d_half`; no texture matmul
  coopmat shader exists yet.

For real LLaMA buffer `.pte`:

- 28 prefill linears route to `linear_coopmat_half`.
- The buffer path is fastest at seq=256, but it is not the recommended
  production default because previous seq=2048 evidence showed a severe
  whole-graph buffer-storage regression.

## ETDump Takeaways

Real LLaMA 4L fp16 seq=256 ETDump leaf-event totals:

| Category | tex Stephen | tex coopmat | buf coopmat | buf disable coop |
| --- | ---: | ---: | ---: | ---: |
| linear | 525.11 | 354.05 | 241.83 | 455.09 |
| CPU/GPU copy | 274.66 | 251.98 | 250.95 | 257.06 |
| bmm / matmul | 7.91 | 8.27 | 9.97 | 8.78 |
| softmax | 0.62 | 0.65 | 1.85 | 1.79 |
| total leaf | 873.43 | 682.82 | 580.59 | 797.68 |

Linear-category speedups:

```text
texture coopmat vs texture Stephen: 525.11 / 354.05 = 1.48x
buffer coopmat vs buffer disable: 455.09 / 241.83 = 1.88x
buffer coopmat vs texture Stephen: 525.11 / 241.83 = 2.17x
```

Important interpretation:

- softmax-buffer is only 1.85 ms at seq=256, so the seq=2048 softmax-buffer
  tax is effectively hidden at this smaller sequence length.
- ETVK copy is roughly flat across variants at seq=256.
- BMM/matmul is unchanged because no texture coopmat matmul shader exists.

## Caveats

- The primary requested seq=2048 workload was not measured in Phase 2. The
  process is OOM-killed on the 28 GB host after exporting a 3.85 GB `.pte`.
- Synthetic buffer `.pte` did not actually route linears as buffer because
  fp16 RMSNorm casts caused storage propagation back to texture3D.
- Real LLaMA helper needed a local torchao import workaround.
- Phase 2 correctness is E2E fp16-output tolerance plus Phase 1 sampled
  large-shape shader correctness, not a new full large-shape shader proof.

## Phase 3 Starting Point

Use texture3D linear coopmat as the production default candidate for
texture-backed graphs:

```text
dtype == fp16
device supports required KHR coopmat config
op == linear
input/output storage == texture3D
shape passes M % 64 == 0, N % 64 == 0, K % 32 == 0
fp32 accumulator
K-step 32
```

Keep buffer coopmat available only when the graph is already buffer-backed for
reasons other than coopmat. Do not force whole-graph buffer storage.

The highest-value follow-up shader work is texture coopmat matmul for attention
BMMs (`Q x K^T`, `attention x V`).
