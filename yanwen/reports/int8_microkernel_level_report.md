# Int8 microkernel-level report for LLaMA 3.1 8B prefill

**Updated:** 2026-05-11 · **Author:** Yanwen Xu  
**Device:** AMD Radeon 780M, RADV PHOENIX, RDNA3+ mobile iGPU, wave64, Mesa 25.0.7  
**Scope:** shader / microkernel benchmark only, not an end-to-end model run  
**Model shapes:** LLaMA 3.1 8B, L=32, prefill S=128 shapes

This report explains the int8 microkernel results in plain language, then maps the numbers back to LLaMA. The main point is:

> Int8 cooperative matrix is very good for the big FFN matmuls, but it is not the best shader for every LLaMA linear. The best projected int8 path is a hybrid: **KHR coopmat for FFN** and **q8ta scalar int8-dot-product for Q/O/K/V**.

---

## 1. Vocabulary: what is being measured?

### Microkernel

A **microkernel** here means one low-level GPU shader that performs one matrix multiplication shape. It is not the full LLaMA model.

For example, instead of running the whole transformer block, the benchmark asks:

```text
How long does this one shader take for:

(128 x 4096) @ (4096 x 14336) = (128 x 14336)?
```

That shape corresponds to the LLaMA FFN gate/up projection at sequence length 128.

### GEMM shape: M, K, N

Most linear layers are matrix multiplies:

```text
A[M, K] @ B[K, N] = C[M, N]
```

For LLaMA prefill:

- `M` is usually the number of tokens processed together. At S=128, `M=128`.
- `K` is the input hidden width.
- `N` is the output width of the projection.

Example:

```text
FFN gate/up:
A = activations = 128 x 4096
B = weights     = 4096 x 14336
C = output      = 128 x 14336
```

### Required vocabulary

| Term | Meaning in this report | Why you should care |
|---|---|---|
| **Q8CSW** | The non-coopmat int8 weight path: 8-bit weights, fp16 activations. In this report I also call it **W8A16**. | It saves memory bandwidth for weights, but it still has to dequantize int8 weights and multiply with fp16 activations. That scalar conversion work is why it can be slower than expected. |
| **W8A8** | 8-bit weights and 8-bit activations. Both sides of the matmul are int8. | This is needed for true int8 dot-product or int8 cooperative-matrix math. It is a stronger int8 setup than Q8CSW/W8A16. |
| **Q8TA** | A W8A8 non-coopmat shader that uses packed int8 dot-product instructions. | It is still not coopmat, but it is much faster than Q8CSW. It wins over int8 coopmat on smaller Q/O and K/V shapes because it has lower tile/setup overhead. |
| **CM wave64** | The corrected int8 **cooperative matrix** shader for AMD wave64 hardware. "CM" means cooperative matrix; "wave64" means each GPU subgroup has 64 lanes. | This is the fast int8 tensor-core-like path. It wins on large FFN shapes. The older wave32-assumed CM shader was invalid on the 780M because it did incomplete work. |
| **FFN gate/up** | Two FFN projections in each transformer layer. At S=128 each has shape `(128, 4096, 14336)`, and together they appear 64 times per forward. | This is the biggest `N` shape in the repeated linears. It gives coopmat enough work, so int8 CM is fastest here. |
| **FFN down** | The FFN projection that maps the large intermediate vector back to hidden size. Shape `(128, 14336, 4096)`, 32 dispatches per forward. | Still a heavy FFN matmul. Int8 CM wins, but less strongly than gate/up because the output width `N=4096` is smaller. |
| **Q/O** | Attention query and output projections. Shape `(128, 4096, 4096)`, 64 dispatches per forward. | Medium-size output width. Int8 CM beats Q8CSW, but Q8TA is faster because the CM tile overhead is too high for this shape. |
| **K/V** | Attention key and value projections. Shape `(128, 4096, 1024)`, 64 dispatches per forward. | Small output width. Int8 CM launches only about 8 output workgroups, so the GPU is under-filled. Q8TA is better here. |

One naming trap: **Q8CSW** and **Q8TA** are both int8-related, but they are not the same thing. Q8CSW is effectively W8A16 and pays dequantization overhead. Q8TA is W8A8 and uses packed int8 dot-product instructions.

---

## 2. What was benchmarked?

Three custom microbench binaries were used:

| Binary | What it benchmarks |
|---|---|
| `linear_coopmat_bench` | fp16/fp32 vec and fp16/fp32 coopmat linears |
| `q8csw_linear` | W8A16 q8csw and W8A8 q8ta non-coopmat int8 paths |
| `khr_cm_gemm_int8` | KHR cooperative-matrix int8 path, including wave64-correct shader |

All timings are GPU timestamp means over 10 measured runs after warmup. They are **per dispatch**, not full-model wallclock.

Latest source logs:

| Artifact | Description |
|---|---|
| `yanwen/artifacts/int8_microbench/linear_coopmat_bench_20260510_204428_fp16_llama_baseline.log` | real fp16 coopmat LLaMA-shape baseline |
| `yanwen/artifacts/int8_microbench/q8csw_linear_20260510_182522.log` | q8csw W8A16 and q8ta W8A8 non-coopmat int8 |
| `yanwen/artifacts/int8_microbench/khr_cm_gemm_int8_20260510_182522.log` | int8 KHR coopmat, original and wave64 variants |
| `yanwen/artifacts/int8_microbench/khr_cm_gemm_int8_sweep_20260510_205654.log` | tile-schedule sweep |
| `yanwen/artifacts/int8_microbench/khr_cm_validate_sweep_20260510_205654.log` | tight-tolerance validation |

---

## 3. Full latency matrix

Times are milliseconds per shader dispatch. Bold marks the fastest correct path for that shape.

| Shape | fp32 vec | fp32 cm | fp16 vec | fp16 cm | int8 q8csw W8A16 | int8 q8ta W8A8 scalar | int8 cm wave64 W8A8 | int8 cm orig broken |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN gate/up `(128,4096,14336)` | 15.18 | 6.78 | 15.33 | 3.42 | 9.23 | 3.27 | **2.00** | 1.35 |
| FFN down `(128,14336,4096)` | 16.76 | 6.16 | 16.21 | 3.52 | 9.62 | 3.19 | **2.40** | 1.54 |
| Q/O `(128,4096,4096)` | 4.59 | 1.78 | 4.60 | 0.92 | 2.74 | **0.78** | 1.27 | 0.80 |
| K/V `(128,4096,1024)` | 1.24 | 0.31 | 1.24 | **0.26** | 0.82 | 0.32 | 0.53 | 0.31 |

Important reading:

- For **FFN gate/up**, int8 coopmat is fastest: 2.00 ms.
- For **FFN down**, int8 coopmat is fastest: 2.40 ms.
- For **Q/O**, q8ta scalar W8A8 is fastest: 0.78 ms.
- For **K/V**, fp16 coopmat is slightly fastest: 0.26 ms; q8ta is close at 0.32 ms; int8 coopmat is slower at 0.53 ms.

The broken column is shown only to explain the earlier false lead. It should not be used for performance conclusions.

---

## 4. The main comparisons

### 4.1 Coopmat vs non-coopmat, same precision

This asks: if we stay in the same broad precision family, does coopmat help?

| Comparison | FFN gate/up | FFN down | Q/O | K/V | Readout |
|---|---:|---:|---:|---:|---|
| fp16 cm vs fp16 vec | 4.5x faster | 4.6x faster | 5.0x faster | 4.8x faster | fp16 coopmat is a clear win |
| int8 cm wave64 vs int8 q8csw | 4.6x faster | 4.0x faster | 2.2x faster | 1.5x faster | int8 coopmat beats W8A16 q8csw everywhere |
| int8 cm wave64 vs int8 q8ta | 1.6x faster | 1.3x faster | 1.6x slower | 1.7x slower | coopmat wins only at FFN scale |

The key nuance: `q8csw` is the weaker int8 baseline because it is W8A16 with scalar dequant work. `q8ta` is a stronger W8A8 scalar path. Against q8ta, coopmat wins only when the output is large enough.

### 4.2 int8 coopmat vs int8 non-coopmat

There are two useful "int8 no coopmat" baselines:

1. `q8csw`: W8A16. Weights are int8, activations are fp16. This is closer to the currently available weight-only int8 path.
2. `q8ta`: W8A8. Weights and activations are int8, using packed int8 dot-product instructions, but not cooperative matrix.

These answer different questions.

#### Against q8csw W8A16

| Shape | int8 q8csw | int8 coopmat wave64 | Coopmat speedup | Readout |
|---|---:|---:|---:|---|
| FFN gate/up | 9.23 ms | **2.00 ms** | **4.6x** | huge win |
| FFN down | 9.62 ms | **2.40 ms** | **4.0x** | huge win |
| Q/O | 2.74 ms | **1.27 ms** | **2.2x** | still wins |
| K/V | 0.82 ms | **0.53 ms** | **1.5x** | modest win |

This is the cleanest "does coopmat help the current W8A16-style int8 path?" comparison. The answer is yes. Coopmat beats q8csw on every shape because q8csw pays scalar dequantization overhead: it has to load int8 weights, convert/dequantize them, then multiply with fp16 activations.

The win is largest at FFN because there is enough arithmetic to amortize coopmat's tile setup cost. It shrinks at K/V because K/V has only `N=1024`, so there are only 8 output tiles across N.

#### Against q8ta W8A8 scalar

| Shape | int8 q8ta | int8 coopmat wave64 | Coopmat speedup | Readout |
|---|---:|---:|---:|---|
| FFN gate/up | 3.27 ms | **2.00 ms** | **1.6x** | coopmat wins |
| FFN down | 3.19 ms | **2.40 ms** | **1.3x** | coopmat wins |
| Q/O | **0.78 ms** | 1.27 ms | **0.6x** | q8ta wins by 1.6x |
| K/V | **0.32 ms** | 0.53 ms | **0.6x** | q8ta wins by 1.7x |

This is the more competitive comparison because both paths are W8A8. q8ta is still scalar/non-coopmat, but it uses hardware packed int8 dot-product instructions. That makes it much stronger than q8csw.

The result is mixed:

- Use int8 coopmat for FFN.
- Use q8ta for Q/O and K/V.

This is why the recommended path is hybrid dispatch, not "turn on coopmat for every int8 linear."

### 4.3 int8 coopmat vs fp16 coopmat

This is the user's original "can int8 be 2x faster than fp16 coopmat?" question.

| Shape | int8 cm / fp16 cm | Speedup | Verdict |
|---|---:|---:|---|
| FFN gate/up | 0.58 | **1.72x faster** | close, but below 2x |
| FFN down | 0.68 | **1.47x faster** | useful but not 2x |
| Q/O | 1.38 | 0.73x | int8 coopmat is slower |
| K/V | 2.06 | 0.49x | int8 coopmat is about 2x slower |

So the honest answer is:

> Int8 coopmat is not universally 2x faster than fp16 coopmat. It is meaningfully faster for the FFN shapes, but worse for the smaller attention projection shapes.

---

## 5. Why the answer changes by shape

The wave64 int8 coopmat shader uses a fixed 128 x 128 output tile. That means each workgroup computes a chunk of the output matrix.

For `M=128`, the number of workgroups is mostly driven by `N / 128`:

| Shape | N | Approx workgroups | What happens |
|---|---:|---:|---|
| FFN gate/up | 14336 | 112 | enough work to keep the 12-WGP GPU busy |
| FFN down | 4096 | 32 | enough work, but less than gate/up |
| Q/O | 4096 | 32 | borderline: q8ta's lower overhead wins |
| K/V | 1024 | 8 | not enough work; part of the GPU is idle |

Why this matters:

- Coopmat has fixed setup overhead per workgroup.
- That overhead is worth paying when each dispatch has lots of output tiles.
- At small `N`, there are too few tiles, so q8ta's simpler scalar dot-product path wins.

In plain terms:

```text
coopmat = faster engine, but needs enough road
q8ta    = smaller engine, but starts quickly and handles short roads well
```

For LLaMA, FFN has enough road. K/V does not.

### Performance-analysis view: why slower cases happen

The slower cases are not mysterious "int8 is bad" cases. They are cases where the overheads around the fast int8 matrix instruction dominate the useful arithmetic.

#### 1. Occupancy and parallelism

The 780M has 12 WGPs. A dispatch with only 8 workgroups, like K/V with `N=1024`, cannot fill the GPU. Even if each workgroup uses fast coopmat instructions, there simply are not enough workgroups to occupy the hardware.

For K/V:

```text
M = 128
N = 1024
tile = 128 x 128
workgroups = (M / 128) x (N / 128)
           = 1 x 8
           = 8 workgroups
```

That is less than the number of WGPs. Some GPU capacity sits idle.

#### 2. Fixed tile overhead

The coopmat shader does more than just multiply:

- loads tiles from global memory
- stages data through shared memory
- synchronizes lanes
- performs cooperative matrix loads
- stores the output tile

Those costs are paid per tile. At large `N`, the arithmetic dominates. At small `N`, the setup cost becomes a large fraction of total latency.

q8ta has less setup. It uses packed int8 dot-product instructions in a simpler scalar pipeline. For small shapes, lower overhead beats higher peak throughput.

#### 3. Tile shape mismatch

The int8 coopmat shader is tuned around a fixed 128 x 128 output tile. That is a natural fit for FFN gate/up because `N=14336`, which gives 112 tiles across N.

It is a poor fit for K/V because `N=1024`, only 8 tiles across N. The shader's geometry cannot shrink its fixed overhead in proportion to the smaller output.

#### 4. Memory and dequantization behavior

q8csw is slow for a different reason: it is W8A16. It saves weight bandwidth, but it still has fp16 activations and scalar dequantization work. That overhead is why int8 coopmat beats q8csw even at Q/O and K/V.

q8ta removes much of that dequantization penalty by using W8A8 and packed int8 dot-product instructions. Once q8ta is the comparison point, coopmat needs enough tile-level arithmetic to pull ahead. FFN has enough; Q/O and K/V do not.

#### 5. Launch and dispatch overhead are not the main issue, but per-workgroup overhead is

These timings are GPU timestamp timings for one dispatch. The important overhead here is not Python or model-level launch overhead. It is inside the shader work itself: tile prefetch, barriers, cooperative matrix load/store, and limited workgroup count.

That is why tuning `TILE_K` or subgroup layout was tested. The result showed the existing 2x2, 128x128x32 schedule is already near the best point among those simple knobs.

---

## 6. Tile-schedule sweep

After the wave64 correction, the next question was whether tile tuning could reach 2x over fp16 coopmat at FFN shapes.

The tested variants changed:

- `TILE_K`: how deep each K chunk is
- subgroup layout: `1x4`, `2x2`, `4x1`
- some variants tried alternate `TILE_N` or column-major B

Only four variants passed correctness validation:

| Variant | TILE_M | TILE_N | TILE_K | sg_w | sg_h | FFN gate/up | FFN down | Q/O | K/V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v0_baseline` | 128 | 128 | 32 | 2 | 2 | **2.006** | **2.340** | **1.241** | 0.462 |
| `v1_deepK` | 128 | 128 | 64 | 2 | 2 | 2.391 | 2.521 | 1.343 | 0.451 |
| `v3_sg1x4` | 128 | 128 | 32 | 1 | 4 | 2.139 | 2.447 | 1.274 | **0.445** |
| `v4_sg4x1` | 128 | 128 | 32 | 4 | 1 | 2.136 | 2.401 | 1.361 | 0.548 |

The baseline schedule was still best for the FFN shapes.

Decision gate:

```text
weighted_int8 = 64 x 2.006 + 32 x 2.340 = 203.3 ms
weighted_fp16 = 64 x 3.421 + 32 x 3.517 = 331.5 ms
ratio         = 203.3 / 331.5 = 0.613
speedup       = 1 / 0.613 = 1.63x
target        = 2.00x
```

The tile tuning did **not** meet the 2x target. It landed at 1.63x for FFN-weighted work.

Why not?

- Deeper K tiles increased register/shared-memory pressure faster than they reduced overhead.
- Subgroup reshuffles did not materially improve FFN.
- Smaller-N variants either failed correctness validation or did not help the main FFN shapes.

---

## 7. Mapping microkernel results back to a full LLaMA forward

The full model uses these kernels many times. Multiply per-dispatch latency by the dispatch count:

| Component | Dispatches | Shape | int8 cm wave64 ms/dispatch | Total |
|---|---:|---|---:|---:|
| FFN gate + up | 64 | `(128,4096,14336)` | 2.00 | 128.0 ms |
| FFN down | 32 | `(128,14336,4096)` | 2.40 | 76.8 ms |
| Q + O | 64 | `(128,4096,4096)` | 1.27 | 81.3 ms |
| K + V | 64 | `(128,4096,1024)` | 0.53 | 33.7 ms |
| **All non-lm-head linears** | 224 | - | - | **~320 ms** |

If using pure int8 coopmat everywhere:

```text
linears ≈ 320 ms
unchanged non-linear / copy / view work ≈ 170 ms
projected forward ≈ 490 ms
projected throughput ≈ 128 / 0.490 ≈ 261 tok/s
```

Compared with measured fp16 coopmat:

```text
fp16 coopmat measured ≈ 583 ms ≈ 220 tok/s
pure int8 coopmat projection ≈ 490 ms ≈ 261 tok/s
speedup ≈ 1.19x
```

That is a real improvement, but modest.

### Hybrid projection

Use the best shader per shape:

| Component | Dispatches | Best path | ms/dispatch | Total |
|---|---:|---|---:|---:|
| FFN gate + up | 64 | int8 coopmat | 2.00 | 128.0 ms |
| FFN down | 32 | int8 coopmat | 2.40 | 76.8 ms |
| Q + O | 64 | q8ta | 0.78 | 49.9 ms |
| K + V | 64 | q8ta | 0.32 | 20.5 ms |
| **All non-lm-head linears** | 224 | mixed | - | **~275 ms** |

Then:

```text
linears ≈ 275 ms
unchanged non-linear / copy / view work ≈ 170 ms
projected forward ≈ 445 ms
projected throughput ≈ 128 / 0.445 ≈ 288 tok/s
```

This is the best short-term int8 story:

- ~1.31x over measured fp16 coopmat
- ~3.97x over fp16 baseline
- requires W8A8 activation quantization and real model dispatch integration

---

## 8. Practical conclusion

1. **Do not use the broken wave32 int8 coopmat timings.** They looked faster because the shader did incomplete work on wave64 hardware.
2. **The corrected wave64 int8 coopmat path is worthwhile for FFN.** It gives 1.72x over fp16 coopmat for gate/up and 1.47x for down.
3. **Do not route every linear through int8 coopmat.** Q/O and K/V are too small for that tile geometry; q8ta or fp16 coopmat is better there.
4. **Tile-schedule tuning alone did not reach 2x over fp16 coopmat.** The measured FFN-weighted speedup is 1.63x.
5. **The recommended int8 direction is hybrid dispatch.** Use coopmat when `N` is large, roughly FFN-sized; use q8ta for smaller attention projections.

The remaining work is not more microbenchmarking of the same knobs. It is integration work:

- wire the wave64 int8 KHR coopmat shader into a real LLaMA linear dispatch path
- compose it with q8ta for smaller projections
- add/export a W8A8 quantization recipe so activations are int8, not just weights
- measure end-to-end overhead from quantization, layout transitions, and staging
