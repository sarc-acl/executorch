# 1.5 vs 1.4 end-to-end gap on the Radeon 780M (2026-09-27)

Answer: SDPA. On 1B 4w (2048-token prefill), SDPA accounts for 307 of the 309 ms per prefill (99 %).

Setup:
- Timing: ETDump per-kernel GPU timestamps, last run of 2 per file, 3 repeats per cell (median, with min–max).
- Builds: traced llama_main of 1.4 `yanwen/release14-quant-shaders-780m` e4786474b9, and of 1.5 `dev/1.5`.
- Device state: gpu-lab lock held; default DVFS; host idle apart from a user's `nvtop`.

## 1B per-family GPU ms per prefill (measured)

| family | 1.4 tuned | 1.5 tuned | delta | 1.4 tiled | 1.5 tiled | delta |
|---|---:|---:|---:|---:|---:|---:|
| SDPA attn*V | 43.6 (coopmat) | 224.7 (tiled) | +181.1 | 43.6 | 219.6 | +176.0 |
| SDPA QK^T | 75.3 (coopmat) | 156.7 (tiled) | +81.4 | 75.2 | 157.3 | +82.1 |
| SDPA softmax | 115.1 | 161.6 | +46.5 | 115.3 | 161.5 | +46.3 |
| SDPA kv_cache_update | 5.3 | 3.1 | -2.2 | 5.3 | 3.1 | -2.2 |
| SDPA subtotal | 239.2 | 546.1 | +306.9 | 239.4 | 541.6 | +302.2 |
| linear 4w | 372.5 | 373.3 | +0.8 | 2173.5 | 2188.0 | +14.4 |
| copy/view/slice | 59.2 | 60.8 | +1.6 | 59.6 | 61.1 | +1.5 |
| add/mul, silu, rms_norm, rope | 81.2 | 81.6 | +0.4 | 81.3 | 81.7 | +0.4 |
| GPU sum | 752.1 (752.1-752.3) | 1061.5 (1059.6-1062.8) | +309.4 | 2553.8 (2543-2561) | 2872.8 (2861-2890) | +319.0 |
| graph-execute wall | 754.2 | 1063.9 | +309.7 | 2559.3 | 2879.5 | +320.2 |
| prefill tok/s | 2702 | 1916 (1916-1919) | | 799 (797-803) | 710 (706-713) | |

- Dispatch counts are identical in the two versions.
- Wall minus GPU sum is ~2 ms (tuned) and ~6 ms (tiled) in both versions, so 1.5 adds no CPU or submission overhead.
- The gap is the same in tiled mode because 1.4 runs its SDPA coopmat there too; `ET_VK_FORCE_TILED_LINEAR` only affects the linears.

## SDPA microbench per call (measured; `--sdpa --regime=prefill`, excludes kv-cache update)

| model | 1.4 | 1.5 | predicted e2e delta | observed |
|---|---:|---:|---:|---:|
| 1B | 15.0 ms | 33.6 ms | x16 layers = 0.30 s | 0.31 s |
| 3B | 14.6 ms | 52.7 ms | x28 = 1.07 s | 1.1 s |
| 8B | 18.7 ms | 70.6 ms | x32 = 1.66 s | 1.8 s |

The 3B/8B attribution is inferred from the microbench, not traced end to end. About 0.14 s of the 8B gap
is unexplained.

## What recovers it (SARC 1.4 commits, none in upstream 1.5)

1. **attn*V coopmat**: `sdpa_compute_out_coopmat` (t64x64k32g22s64), ~181 ms. Commits:
   8be6567b3a, 7f43322dc1, 788617961a, be70eee109.
2. **QK^T coopmat**: `sdpa_compute_attn_weights_coopmat` (t128x64k32g22s64, fast path df5b8dca1b),
   ~81 ms.
3. **Softmax causal-tail truncation** (582ec23f95, an edit to upstream `sdpa_attn_weights_softmax.glsl`),
   ~46 ms.
4. **Supporting `SDPA.cpp` changes** (+310 lines):
   - gate `supports_cooperative_matrix() && subgroup_size()==64`, with the `ET_VK_DISABLE_COOPMAT` kill
     switch;
   - shape eligibility;
   - shader pick and workgroup sizes;
   - extra spec constants.

Expected after porting: ~2705 tok/s tuned (inferred).

## Port to dev/1.5 (estimate from reading the code; not attempted)

- **Shaders:** about 670 GLSL lines, nearly verbatim; `common.glslh` is unchanged from 1.4 to 1.5.
  About 250 lines of C++ need re-plumbing, plus the softmax edit.
- **Workgroup API (74d024cde):** the pickers must return `GlobalWorkGrid` / `LocalWorkGroup`.
  Mechanical.
- **Spec-constant conflict on the AV node (real):**
  - 1.5's GQA decode shader passes `{1.0, group_size}` to the same dynamic node.
  - The 1.4 AV coopmat uses slot 2 for `num_k_chunks_arg`.
  - The layouts must be unified.
- **`pick_sdpa_av_shader`:** a small textual conflict. Re-check 1.5's `align_up_4` padding against
  `aw_row_width`.
- **SARC zones:**
  - kernels go into `glsl/sarc` + `impl/sarc/SdpaCoopmat.cpp`;
  - rows for sdpa_qk and sdpa_av;
  - hook lines in `SDPA.cpp` (listed in `sarc/HOOKS`).
  - The softmax edit changes upstream SPIR-V for every device: ship it as a SARC copy selected by a row,
    or get owner sign-off as an upstreamable fix.
- **After porting:** run `--sdpa-correctness-only` (all tiers) and the unaligned-prompt e2e check.

Not checked: SDPA coopmat correctness on 1.5 (not ported).

Raw data: `t14-*`, `t14b-*`, `t15-*` (ETDumps and logs), `summary.txt`, `analyze.py`, `run.sh`, `mb/`.
Host copies are in `~/.cache/et-e2e/sdpa-gap/`.
