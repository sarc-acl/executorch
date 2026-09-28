# SARC 1.5-r2 prefill results: what they are, why they look this way, and how strong the evidence is

Technical report for engineering review, 2026-09-28.

The headline benchmark is in [REPORT.md](REPORT.md): 30 configurations and 300 timed runs, with SARC's
cooperative-matrix kernels 1.48–5.35× faster than stock ExecuTorch 1.5 at 2048-token prefill. This document
explains those numbers.

An independent audit re-derived every number from the raw data; its findings are applied here.

Evidence labels:

| Label | Meaning |
|---|---|
| **[M]** | Measured in this campaign: timed runs, warm ETDump kernel traces, logits |
| **[R]** | Measured by the roofline tool: confirmed campaign roofs, not this campaign |
| **[S]** | Read from shader or selection source; not ISA-verified |
| **[I]** | Inference from [M]/[R]/[S] data. Plausible, but alternatives are not excluded |
| **[O]** | Open: not measured |

## 1. Summary of findings

1. **Stock 1.5 prefill runs on scalar units; SARC runs on the matrix units.** [M][R][S]
   - Stock's GEMM kernels use scalar fp16 FMA (4w) and int8 dot (8da4w), and reach 22–59 % of those scalar
     roofs.
   - SARC's coopmat kernels reach 22–70 % of the matrix roofs.
   - The matrix roofs are 1.2–9.3× the scalar roofs stock is bound by.
   - The GEMM kernel speedup ranges from 1.40× (780M 8da4w) to 10.6× (Orin 4w). It splits exactly into a
     hardware-headroom term and an efficiency term (§4). That split is an accounting identity, not independent
     confirmation.
   - Headroom is the dominant term on the Orin and on Intel.
2. **The end-to-end gain is smaller than the GEMM gain, and it grows with model size.** [M]
   - GEMMs are 43–91 % of stock prefill time. Attention and everything else run at about 1× outside the 780M.
   - Where the GEMM speedup is roughly constant across sizes (780M, 4070 Ti, Orin), the growing GEMM share alone
     makes the end-to-end speedup grow.
   - On Intel the GEMM speedup is **not** constant. For 8da4w it rises with size (B580 2.30 → 2.90) because the
     stock kernel loses efficiency; for 4w it falls slightly (5.16 → 4.83). See §5.
3. **On the 780M, 8B speeds up slightly less than 3B.** [M]
   - It is the only GPU where SARC also replaces attention, and attention gains more there than GEMMs do
     (4.0–4.2× vs 2.5×).
   - Attention's share of stock time falls from 34 % at 3B to 24 % at 8B (§5).
4. **Stock 8da4w beats stock 4w on AMD and Intel, but not on NVIDIA.** [M][R]
   - On the 780M and Intel, the int8 dot roof is 1.2–1.4× the fp16 FMA roof, and stock 8da4w also runs at a
     higher fraction of its roof.
   - On the 4070 Ti and Orin, stock 8da4w reaches only 22–24 % of its dot roof and runs at nearly the same
     absolute rate as stock 4w.
   - **Why** is **[O]**; §6 gives two candidates.
5. **After tuning, the faster scheme depends on the GPU.** (§7)
   - **780M: 4w wins by 7.5 %.** [M][R]
     - The int8 matrix roof is 0.97× the fp16 roof SARC 4w uses, and the int8 kernel is 6.8 % slower (+188 ms).
     - 8da4w adds activation quantization (+166 ms) and saves some view copies (−92 ms).
     - The quantize pass and the copies are software, so part of the gap is recoverable [I].
   - **Orin: 4w wins.** [M] SARC 8da4w reaches 24 % of the int8 register roof, against 61 % for 4w on its roof.
     Why is **[I]/[O]** (§7).
   - **Intel: a tie, or 8da4w by 12 %.** [M]
   - **4070 Ti: 8da4w wins by 12 %.** [M]
6. **The three next-token differences are consistent with a near-tie in the quantized model.** [I], built on [M]
   measurements:
   - in the 8da4w model the two candidate tokens are 0.148 logit apart (stock kernels, one GPU);
   - stock itself returns different tokens on different GPUs;
   - SARC's token ("bullying") is the one the fp32 model ranks higher of the two.
   - SARC's own logits were not measured **[O]** (§8).
7. **Where the next speedup is.** [M]→[I]
   - After SARC, attention is 33–41 % of 8B prefill time on the four GPUs without a SARC attention row.
   - SARC 8da4w GEMMs reach only 22–24 % of the int8 register roof on Intel and Orin (§9).
8. **One disclosed limitation affects absolute tok/s.** [M]
   - The timed prompt `prompt_2048.txt` is the word "the" repeated 2048 times.
   - Prefill work does not depend on token values, but power draw can. On a power-limited GPU (the 4070 Ti), a
     low-toggle input may run at higher clocks than real text, for both builds.
   - A real-text timing check has not been run **[O]** (§10).

## 2. Evidence base

| Source | What it gives | Quality notes |
|---|---|---|
| Timed runs ([REPORT.md](REPORT.md), `raw/*/runs.csv`) | End-to-end tok/s, n = 5 per build, 300/300 accepted | Clocks not pinned; B580 noisier (display GPU); prompt is repeated "the" (§10) |
| Warm ETDump traces (`raw/*/trace2/*.etdp`, 60 files) | GPU time per dispatch, with operator name and tensor shapes | One `--warmup` run per cell. Each event has 2 `raw` entries; the last is the warm execution (checked against `start_time` order and the logged run time). Trace graph time vs timed median: −6.7 % to +3.8 %. [trace_analysis.py](results/scripts/trace_analysis.py) → `evidence/trace/{families,gemm,totals}.csv` |
| Confirmed roofs ([evidence/roofline.md](evidence/roofline.md), `roofline.json`) | Measured matrix, scalar, fed-matrix and memory peaks per GPU, each with its REPORT.md line | `fast` plan, 3 fresh-process repeats, spread ≤ 3.2 %, sentinel healthy on all 34 checks. **Not ISA-verified.** Sustained roofs unconfirmed. 780M `standard` campaign agrees within 3.4 %. Measured 2026-09-26/27, not on the benchmark day |
| Kernel efficiency ([evidence/efficiency.py](evidence/efficiency.py) → `efficiency.csv`) | Achieved prefill-GEMM rate = Σ 2·M·N·K / Σ GPU time, and % of the matched roof | Kernel level only. The 8da4w quantize dispatch and the stock 4w input transpose are reported separately (§5). Logical FLOPs |
| Logits ([evidence/logits/](evidence/logits/)) | fp32 CPU reference; 8da4w Vulkan logits at the disputed position | Vulkan logits come from the ExecuTorch Python runtime built from the SARC dev tree, whose B580 path runs stock kernels. Device index 0 is the B580 by vulkaninfo order; the device name is not logged (§8) |
| Source | What each kernel computes; which SARC tile runs where | Not ISA-verified |

**Kernel, operator and model levels are reported separately**, as the project workflow requires:
- **kernel:** GEMM dispatches only;
- **operator:** all dispatches issued by the linear op, i.e. GEMM plus the 8-bit activation quantize plus stock
  4w's input-transpose dispatch;
- **model:** end to end.

**Are the GEMMs compute-bound?** [I]
- Whole-tensor arithmetic intensity is ≈ 1,640 FLOP/B: an 8B 4096×4096 projection does 68.7 GFLOP over ≈ 42 MB.
- DRAM ridge points:
  - fp16 matrix: Orin 156, 780M 170, B580 240, 4070 Ti 258, B70 287 FLOP/B;
  - int8 matrix: 780M 166, Orin 313, B580 497, 4070 Ti 518, B70 595 FLOP/B.
- That alone does **not** prove the kernels are compute-bound. Per-tile reuse is much lower:
  `JETSON-WMMA-LESSONS.md:115-121` estimates ~102 op/B with fp16 A and 171 op/B with int8 A for a 128×128 tile,
  which is below several int8 ridges.
- The register-resident matrix roof is therefore an *upper bound*, not a proven operating point. Where it
  matters, the report also gives the fed roofs (matrix fed from shared memory or cache).

## 3. Measured roofs and where the kernels run

Confirmed short-run roofs [R], with sources in [roofline.md](evidence/roofline.md) §2–§4. Units: TFLOP/s for
fp16, TOP/s for int8.

| GPU | fp16 FMA | int8 dot | fp16 matrix (accumulator used by SARC 4w) | int8 matrix | DRAM read |
|---|---|---|---|---|---|
| Radeon 780M | 8.14 | 11.72 | 14.77 (fp32 acc) | **14.39** | 86.7 GB/s |
| Arc B580 | 27.3 | 32.1 | 111.7 (fp16 acc) | 231.4 | 465 GB/s |
| Arc Pro B70 | 42.9 | 50.4 | 173.3 (fp16 acc) | 359.9 | 604 GB/s |
| RTX 4070 Ti SUPER | 45.0 | 78.1 | 183.7 (fp16 acc) | 369.2 | 713 GB/s |
| Jetson Orin Nano | 1.77 | 2.09 | 9.70 (fp16 acc)* | 19.52 | 62.3 GB/s |

\* One Orin 8B tile uses `ACC_FP32` [S]: the w2 projection, K = 14336, 30 % of Orin SARC 4w GEMM time. Its roof
is 9.736, 0.4 % higher, so the Orin figures below are about 0.1 % optimistic.

**Achieved prefill-GEMM rate, Llama 3.1 8B, and % of the matched roof** [M] (`evidence/efficiency.csv`):

| GPU | stock 4w (vs fp16 FMA) | SARC 4w (vs fp16 matrix) | stock 8da4w (vs int8 dot) | SARC 8da4w (vs int8 matrix) |
|---|---|---|---|---|
| Radeon 780M | 4.11 (50.5 %) | 10.29 (69.7 %) | 6.90 (58.9 %) | 9.64 (67.0 %) |
| Arc B580 | 8.80 (32.2 %) | 42.5 (38.1 %) | 17.3 (54.1 %) | 50.3 (21.8 %) |
| Arc Pro B70 | 13.7 (31.9 %) | 64.6 (37.2 %) | 28.0 (55.7 %) | 82.2 (22.8 %) |
| RTX 4070 Ti SUPER | 17.8 (39.5 %) | 119.0 (64.7 %) | 17.6 (22.5 %) | 155.5 (42.1 %) |
| Jetson Orin Nano | 0.55 (31.4 %) | 5.88 (60.6 %) | 0.50 (23.8 %) | 4.72 (24.2 %) |

**What stock computes** [S]
- Stock 4w (`q4gsw_linear_gemm__tin__w_4x8`) is fp16 FMA with fp16 accumulation: 32 FMAs per four dequant steps,
  plus a separate input-transpose dispatch per linear op.
- Stock 8da4w (`linear_dq8ca_q4gsw_tiled`) uses `dotPacked4x8AccSatEXT`, 32 per k4 step, with a per-group
  zero-point correction.
- Neither uses cooperative matrix. Both are unchanged from upstream `985c1ceccc`.

![Measured roofs and achieved kernel rates](evidence/figures/e1_roofs_vs_kernels.png)

*Figure E1. Measured roofs (bars) and achieved 8B prefill-GEMM rates (markers), each on its matched roof, log
scale. Figure labels round to integers (e.g. 4070 Ti stock 40 % / 22 %); the tables carry one decimal.*

## 4. Why the kernel speedup differs across GPUs

For the prefill GEMMs:

```
SARC rate / stock rate  ≡  (SARC roof / stock roof)  ×  (SARC % of roof / stock % of roof)
                              hardware headroom H         efficiency gain E
```

**What this identity does and doesn't show**
- The roofs cancel, so H × E equals the measured rate ratio by construction. Only H is independent information
  [R]; E is the measured ratio divided by H.
- The split depends on which roof each kernel is compared with. Against the 780M's fp16-accumulate roof (10.96)
  instead of the fp32-accumulate roof (14.77), 780M 4w would be H = 1.35 and E = 1.86.
- What it shows: how much of each GPU's gain the hardware makes *available*, and how well each kernel *uses* it.

8B values [M][R]. "e2e" is the timed end-to-end speedup from `cells.csv`.

| GPU · scheme | H | E | H × E | measured GEMM rate ratio | e2e (timed) |
|---|---|---|---|---|---|
| 780M · 4w | 14.77/8.14 = 1.82 | 69.7/50.5 = 1.38 | 2.51 | 2.51 | 2.66 |
| 780M · 8da4w | 14.39/11.72 = 1.23 | 67.0/58.9 = 1.14 | 1.40 | 1.40 | 1.73 |
| B580 · 4w | 4.09 | 38.1/32.2 = 1.18 | 4.84 | 4.83 | 3.21 |
| B580 · 8da4w | 7.22 | 21.8/54.1 = 0.40 | 2.91 | 2.90 | 1.87 |
| B70 · 4w | 4.04 | 37.2/31.9 = 1.17 | 4.71 | 4.72 | 3.11 |
| B70 · 8da4w | 7.15 | 22.8/55.7 = 0.41 | 2.92 | 2.93 | 1.98 |
| 4070 Ti · 4w | 4.09 | 64.7/39.5 = 1.64 | 6.69 | 6.70 | 4.04 |
| 4070 Ti · 8da4w | 4.73 | 42.1/22.5 = 1.87 | 8.85 | 8.84 | 4.53 |
| Orin · 4w | 5.49 | 60.6/31.4 = 1.93 | 10.60 | 10.61 | 5.35 |
| Orin · 8da4w | 9.33 | 24.2/23.8 = 1.02 | 9.49 | 9.48 | 5.27 |

The last-digit differences between H × E and the measured ratio are rounding in the percentages.

**Reading the table**
- The Orin and Intel gain most because their matrix units offer 4–9× the scalar rate.
- The 780M's matrix units offer only 1.2–1.8×, which caps its GEMM gain whatever the kernel quality.
- On Intel, SARC 8da4w converts less of its roof than stock does of its own (E ≈ 0.4), yet the 7.2× headroom
  still yields 2.9×.
- End-to-end speedups are lower than GEMM speedups because GEMMs are not all of prefill (§5).

![Speedup decomposition](evidence/figures/e4_speedup_decomposition.png)

*Figure E4. The GEMM kernel speedup as hardware headroom (grey) × efficiency gain (blue above 1, pink below 1).
The black tick is the measured rate ratio. The factorisation is an identity (§4).*

## 5. From kernel to operator to model: the model-size trend

Per-family speedups from the warm traces [M] (`evidence/trace/modelsize.txt`, with operator-level attribution):
- **GEMM×**: kernel level.
- **Op×**: linear-operator level, i.e. GEMM + activation quantize + input transpose.
- **Attn×**: attention (QKᵀ, softmax, AV, KV update).
- **e2e× (trace)**: stock total ÷ SARC total, which differs from the timed ratio by a few %.
- **Share**: fraction of *stock* trace time.
- The remainder (norms, RoPE, elementwise, copies) is 0.97–1.02× everywhere except the 4070 Ti (1.01–1.13×).

| GPU · scheme | GEMM share 1B/3B/8B | GEMM× 1B/3B/8B | Op× 1B/3B/8B | Attn× 1B/3B/8B | e2e× (trace) |
|---|---|---|---|---|---|
| 780M · 4w | 55 / 57 / 67 % | 2.48 / 2.52 / 2.51 | 2.63 / 2.68 / 2.62 | **2.29 / 4.00 / 4.17** | 2.22 / 2.69 / 2.64 |
| 780M · 8da4w | 43 / 45 / 57 % | 1.35 / 1.37 / 1.40 | 1.33 / 1.35 / 1.38 | 2.30 / 3.97 / 4.14 | 1.56 / 1.84 / 1.74 |
| B580 · 4w | 73 / 73 / 84 % | 5.16 / 4.89 / 4.83 | 5.44 / 5.38 / 4.98 | ≈1.0 | 2.66 / 2.92 / 3.18 |
| B580 · 8da4w | 52 / 61 / 74 % | **2.30 / 2.51 / 2.90** | 2.21 / 2.43 / 2.84 | ≈1.0 | 1.42 / 1.57 / 1.95 |
| B70 · 4w | 70 / 72 / 82 % | 5.07 / 4.92 / 4.72 | 5.37 / 5.43 / 4.88 | ≈1.0 | 2.56 / 2.83 / 3.06 |
| B70 · 8da4w | 50 / 58 / 72 % | **2.49 / 2.48 / 2.93** | 2.37 / 2.40 / 2.86 | ≈1.0 | 1.43 / 1.54 / 1.92 |
| 4070 Ti · 4w | 76 / 82 / 87 % | 6.57 / 6.63 / 6.70 | 6.60 / 6.67 / 6.74 | ≈1.0 | 2.91 / 3.44 / 4.06 |
| 4070 Ti · 8da4w | 77 / 83 / 88 % | 8.42 / 8.75 / 8.84 | 8.02 / 8.43 / 8.57 | ≈1.0 | 3.13 / 3.80 / 4.55 |
| Orin · 4w | 81 / 84 / 89 % | 10.9 / 11.0 / 10.6 | 10.9 / 11.1 / 10.6 | ≈1.0 | 3.90 / 4.37 / 5.36 |
| Orin · 8da4w | 83 / 86 / 91 % | 9.20 / 8.99 / 9.48 | 8.79 / 8.69 / 9.25 | ≈1.0 | 3.88 / 4.23 / 5.28 |

**Operator vs kernel level** [M]
- **4w:** SARC also removes stock's input-transpose dispatch, which is 2.6–7.7 % of stock time on Intel and 3.1–3.6 % on
  the 780M (320.6 ms of the 780M 8B total). So the operator speedup is higher than the kernel speedup: B580 3B 5.38× vs 4.89×.
- **8da4w:** SARC's activation-quantize dispatch costs more than stock's, so the operator speedup is 1.5–4 % below
  the kernel speedup.

**Why the model-size trend exists**
- **780M, 4070 Ti, Orin** [M]: the GEMM and operator speedups stay within ±4 % across sizes while the GEMM share
  of stock time rises. The end-to-end speedup rises with the share (Amdahl's law).
- **Intel** [M]: the kernel speedup itself changes with size.
  - 8da4w rises (B580 2.30 → 2.90) because the **stock** 8da4w kernel loses efficiency with size: B580 71 → 63
    → 54 % and B70 71 → 68 → 56 % of the dot roof. Why is **[O]**.
  - 4w falls slightly (5.16 → 4.83).
  - Both effects add to the share effect.
- **Why the GEMM share rises** [I]:
  - Per-layer linear-to-attention FLOP ratios are 3.6 / 4.0 / 6.5 for 1B / 3B / 8B.
  - From 1B to 3B the ratio barely moves, which is why the 780M and Intel shares are nearly flat there (55 → 57
    %, 72.5 → 73 %).
  - The 8B jump comes mainly from its wider FFN (14336 = 3.5 × d_model).

**The 780M's attention** [M], per kernel (4w):

| 780M 4w attention | 1B | 3B | 8B |
|---|---|---|---|
| QKᵀ× | 2.08 | 3.51 | 3.72 |
| AV× | 5.11 | 9.86 | 10.04 |
| softmax× | 1.40 | 1.38 | 1.38 |

- **3B → 8B** [M]: attention gains 4.0–4.2× and GEMMs 2.5×. Attention's share of stock time falls from 34 % to
  24 %, so the end-to-end speedup moves toward the GEMM figure. That is why 8B (2.64×) is below 3B (2.69×).
- **1B attention gains only 2.3×.** Two effects of similar size contribute [M]/[I]:
  - **The per-kernel speedups drop at 1B:** QKᵀ 2.08× vs 3.5–3.7×; AV 5.1× vs 9.9–10×. Why is **[O]**. A
    candidate is Llama 3.2 1B's head_dim of 64 (vs 128 for 3B/8B), which halves the coopmat tile's K-work per
    head.
  - **The mix shifts toward softmax,** which is sped up only 1.4×. The SARC softmax is stock's plus causal
    truncation [S] (`openspec/changes/sarc-1.5-sdpa-port/proposal.md`). Its cost does not scale with head_dim,
    so it is a larger share at 1B.
  - Counterfactuals: 1B with the 3B per-kernel speedups gives 2.93× attention; the 3B mix with the 1B per-kernel
    speedups gives 2.74×; the measured 1B value is 2.29×.

![Model-size decomposition](evidence/figures/e3_amdahl_model_size.png)

*Figure E3. Kernel-level GEMM speedup (open circles), the GEMM share of stock time (squares) and the timed
end-to-end speedup (filled), by model size. On the 780M, the diamonds show attention speedup and share. The GEMM
speedup is roughly flat on the 780M, 4070 Ti and Orin. On Intel, 8da4w rises with size and 4w falls slightly.*

![Where prefill time goes](evidence/figures/e2_time_breakdown.png)

*Figure E2. 8B prefill GPU time by kernel family, SARC normalised to the stock bar of the same GPU and scheme.
After SARC, attention is 33–41 % of the remaining time on the four GPUs without a SARC attention row.*

## 6. Why stock 8da4w beats stock 4w on AMD and Intel, but not on NVIDIA

Stock 8da4w ÷ stock 4w GEMM rate = (dot roof ÷ FMA roof) [R] × (efficiency ratio) [M]. 8B values:

| GPU | dot/FMA roof | stock 8da4w % / stock 4w % | stock rate ratio |
|---|---|---|---|
| 780M | 1.44 | 58.9/50.5 = 1.17 | 1.68 |
| B580 | 1.17 | 54.1/32.2 = 1.68 | 1.97 |
| B70 | 1.17 | 55.7/31.9 = 1.75 | 2.05 |
| 4070 Ti | 1.74 | 22.5/39.5 = 0.57 | 0.99 |
| Orin | 1.18 | 23.8/31.4 = 0.76 | 0.90 |

**Measured** [M]
- On both NVIDIA GPUs, stock 8da4w runs at 22–24 % of the dot roof at every model size.
- It reaches almost the same absolute rate as stock 4w: 17.6 vs 17.8 TOP/s on the 4070 Ti, 0.50 vs 0.55 on the
  Orin.

**Why is [O].** Two candidates, neither tested:
- **A shared limiter.** Both stock kernels, although different, hit the same wall on NVIDIA, e.g. texture/L1
  feed or occupancy. The near-identical absolute rates favour this.
- **A lower real roof.** The stock shader uses the *signed saturating* `dotPacked4x8AccSatEXT`, while the
  roofline dot kernel measured the non-saturating form. If NVIDIA lowers the saturating form to more
  instructions, the matched roof is lower than measured. There is no SASS route on these GPUs to check this.

Either way, this affects only the baseline's speed, not any SARC claim.

## 7. After tuning, why 4w or 8da4w wins depends on the GPU

SARC throughput [M], 8B, tok/s:

| GPU | 4w | 8da4w |
|---|---|---|
| 780M | 525.8 | 489.1 |
| Orin | 190 | 170 |
| B580 | 1,694 | 1,680 |
| B70 | 2,438 | 2,738 |
| 4070 Ti | 4,491 | 5,032 |

### Radeon 780M: 4w is 7.5 % faster

**Roofs** [R]
- The int8 matrix roof is 14.39 TOP/s, 0.974× the fp16-with-fp32-accumulate roof (14.77 TFLOP/s) that SARC 4w
  uses. The 780M `standard` campaign agrees within 0.2 %.
- SARC 4w uses fp32 accumulation because RDNA3 WMMA with an fp16 accumulator is slower (10.96 TFLOP/s): the packed
  accumulator is repacked in the loop (`780M-WMMA-LESSONS.md`).
- Against that fp16-accumulate roof, int8 would be 1.31× higher. It is not the roof SARC 4w runs on.

**Where the 262 ms goes** (SARC 8da4w − SARC 4w, 8B) [M]:

| Component | Δ ms | Cause |
|---|---|---|
| GEMM | +188 | the int8 kernel is 6.8 % slower (0.974 roof ratio × 67 % vs 70 % efficiency) |
| Activation quantize | +166 | 8da4w only |
| View copies | −92 | `view_texture_half`: 160 dispatches in 8da4w vs 256 in 4w |

**Interpretation** [I]
- The hardware gives int8 no throughput advantage on this GPU.
- The rest of the gap is quantization and copy overhead, which is software. Fusing the quantize into the
  preceding op would close part of it.
- "No tuning gap" would overclaim, since the roofs are not ISA-verified.

### Jetson Orin: 4w is faster; the 8-bit kernel converts less of its roof

**Measured** [M][R]
- The int8 register roof is 2.01× fp16.
- SARC 8da4w runs at 24.2 % of it, against 60.6 % for 4w on its roof.
- So 8da4w reaches 0.80× the 4w GEMM rate (4.72 vs 5.88).

**Provenance** [S]
- The Orin 8da4w row uses `zpgtr t128x128k64g44s32mk32ra`. `table_nvidia.cpp:88-90` credits the 1.4 branches
  `-4070ti` and `-jetson`.
- The Jetson study screened this K64 / raw-A / paired-B / g44 family on the Orin (≈ 2.32× over its baseline;
  `JETSON-WMMA-LESSONS.md:150-162`).
- So the kernel *was* evaluated on the Orin. It was not re-tuned for this release.

**Why it is low** [I]/[O]
- The only counter evidence is 1.4-era and measured over submission windows: Tensor Active 55.4 % for the
  *original* 4w kernel and 20.85 % for the *final* 8da4w kernel (`JETSON-WMMA-LESSONS.md:95-102`). Those are
  different optimisation stages, not a like-for-like comparison.
- The study calls staging and unpacking "the first hypothesis to test". It remains a hypothesis.
- Against the int8 *cache-fed* roof (8.15 TOP/s [R]), SARC 8da4w is at 58 %. If per-tile reuse, rather than the
  tensor cores, is the limit, the practical ceiling is well below the register roof (§2).

### Intel B580/B70: int8 headroom mostly unused

**Measured** [M]
- SARC 8da4w runs at 22–23 % of the int8 register roof; SARC 4w runs at 37–38 % of the fp16 roof.
- The 2.07× hardware advantage becomes 1.18× (B580) and 1.27× (B70) in GEMM rate.
- Activation quantization then costs 5.3–5.7 % of time.
- Result: a tie on the B580, and 8da4w 12 % faster on the B70.

**Headroom** [R][I]
- The Intel int8 *fed* roofs are high: B580 `matrix_int8_feed_shared` 206 and `feed_cache` 198 TOP/s, 85–89 % of
  the register roof. So the matrix unit can be kept fed at those rates.
- The gap is therefore plausibly in the kernel, not the hardware. This is not demonstrated with counters for the
  1.5 kernels.

### RTX 4070 Ti: 8da4w wins, as the hardware predicts

[M]: 42 % of the int8 roof against 65 % of the fp16 roof gives a 1.31× GEMM rate and +12 % end to end.

## 8. Correctness: the three next-token differences

**The check.** Both builds generate one token from a 1972-token real-text prompt, and the outputs are compared.
27/30 cells match. The 3 differences are all Llama 3.1 8B 8da4w (780M, 4070 Ti, Orin), at "… threatening, or
___".

| Evidence | Result |
|---|---|
| Stock 1.5 across GPUs [M] | "otherwise" on 780M/4070 Ti/Orin, **"bullying" on B580/B70**: the unmodified build disagrees with itself |
| SARC across GPUs [M] | "bullying" on all 5 GPUs, from both SARC 8-bit kernel families |
| 8da4w Vulkan logits, device 0 (B580 by vulkaninfo order), stock kernels [M] | bullying 15.469, otherwise 15.320: **Δ = 0.148 logit**, probabilities 11.2 % vs 9.6 %. One measurement, via the Python runtime of the SARC dev tree; its B580 path has no SARC rows, and default is bit-identical to FORCE_TILED |
| fp32 CPU reference, unquantized [M] | Top-1 " stalking" (26 %), then " intimidation" (18 %) and " bullying" (11 %); " otherwise" is rank 6 (1.9 %). Of the two, the float model prefers **bullying**, by 1.76 logit |
| Reference vs 8da4w, full vocabulary [M] | Up to 3.8 logits apart (mean 0.62). This measures quantization plus fp16 error, not kernel-to-kernel accumulation-order noise |

**Interpretation** [I]
- In the quantized model the two tokens are 0.15 logit apart on one measured path.
- Stock itself flips between them across GPUs, and SARC consistently returns the one the float model ranks
  higher.
- This is consistent with a near-tie decided by accumulation order, not with a SARC error.
- Neither token is the float model's first choice.

**Still [O]**
- SARC's own 8da4w logits, and a direct SARC-vs-stock logit diff on one GPU with the device logged. This is a
  follow-up that needs hardware.
- The promotion-time production-diff (1B/3B/8B, nonzero zero points) passed within tolerance [M, earlier
  campaign], but that checks tensor-level error, not the argmax.

## 9. Where the remaining headroom is

Projections [I], not measurements.

| Target | Evidence | Rough size |
|---|---|---|
| Attention on the four GPUs without a SARC attention row | [M] share of SARC 8B 4w time: B580 33.0 %, B70 34.5 %, 4070 Ti 33.7 %, Orin 41.3 %. Stock QKᵀ runs at ≈ 5–9 % of the matrix roof there | If attention reached the 780M's 4×: ≈ +30–45 % end to end (1 / (1 − 0.75·share)). Assumes the 780M result transfers; the 780M's stock AV was unusually slow |
| SARC 8da4w GEMM on Intel and Orin | [M] 22–24 % of the int8 register roof; Intel fed roofs are 85–89 % of it [R] | Reaching the 4070 Ti's 42 % would be ≈ 1.8× on those GEMMs; the practical ceiling is uncertain (§2, §7) |
| SARC 4w GEMM on Intel | [M] 37–38 % of the fp16 roof, vs 61–70 % on the 780M/Orin/4070 Ti | Up to ≈ 1.6× on those GEMMs, if 60 % is achievable on Xe2 [O] |
| 780M 8da4w overheads | [M] quantize +166 ms, plus copies | Fusing the quantize could remove most of the 7.5 % gap to 4w [I] |

## 10. Threats to validity and follow-ups

**Prompt content** [M]
- The timed prompt is "the" × 2048, and the trace runs used it too.
- Arithmetic is data-independent, but power is not. On the power-limited 4070 Ti (285 W,
  `4070TI-WMMA-LESSONS`), low-toggle inputs can raise clocks.
- Both builds see the same input, so speedups are less exposed than absolute tok/s.
- **Follow-up:** time a real-text 2048-token prompt on each GPU (the B70 once it is free).

**Roofs**
- `fast`-plan short-run roofs: confirmed, but not sustained and **not ISA-verified**.
- Measured on a different day from the benchmark.
- Two suspect roofs are excluded (roofline.md §2, §11).

**Clocks**
- Not pinned in either campaign.
- Clocks under load were not recorded comparably. The Orin ran in its 15 W mode.

**Kernel rates** use logical FLOPs, so dequant and zero-point work is not counted. There is one warm trace per
cell, with no repeat.

**Operator attribution** relies on ETDump operator names. The LM-head gemv (1 dispatch) is inside the linear-op
total.

**Counter evidence** (Nsight, Intel OA, 780M phase timing) exists only for 1.4-era SARC kernels, never for the
stock 1.5 kernels.

**Per-shape efficiency** is in `evidence/trace/gemm.csv` but not analysed. Small-N K/V projections and partial
waves likely differ from the aggregate.

**Open questions**
- Why stock 8da4w is slow on NVIDIA (§6).
- Why stock 8da4w loses efficiency with size on Intel (§5).
- Why 780M 1B attention kernels gain less (§5).
- SARC's own logits (§8).

## 11. Files

Everything is under `sarc-acl/.artifacts/e2e-1.5-2026-09-28/`.

| Path | Content |
|---|---|
| `report/REPORT.md` | Benchmark report |
| `report/TECHNICAL-REPORT.md` | This document |
| `report/evidence/roofline.md`, `roofline.json` | Every roof with source file and line; derived ratios; gaps |
| `report/evidence/efficiency.py`, `efficiency.csv` | Kernel rate and % of roof per cell |
| `report/trace_analysis.py`, `report/evidence/trace/` | Warm-trace time per family, per GEMM dispatch (with shapes) and per linear operator; `modelsize.txt` |
| `report/evidence/logits/` | fp32 reference and Vulkan logits: scripts, logs, tensors, build check |
| `report/evidence/figures/` | E1–E4, their scripts and `*.values.csv` |
| `raw/<gpu>/trace2/` | Warm ETDumps (60) and run logs |
