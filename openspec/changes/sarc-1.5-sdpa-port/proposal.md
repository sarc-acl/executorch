# sarc-1.5-sdpa-port

## Why

On release 1.5 the Radeon 780M ran 1B 4w prefill at 1930 tok/s, against 2702 on the 1.4 SARC branch,
although the linear kernels matched.

ETDump per-kernel timing (`results/sdpa-gap-REPORT.md`) shows the cause. SDPA accounts for +307 of the
+309 ms per 1B prefill:
- the 1.4 branch ran SARC's SDPA coopmat kernels (AMD subgroup 64);
- 1.4 also ran a causally truncated softmax;
- release 1.5 has neither.

## What

**Kernels** (`glsl/sarc/`):
- `sarc_sdpa_qk_coopmat` (t128x64k32g22s64) and `sarc_sdpa_av_coopmat` (t64x64k32g22s64), both from the
  1.4 branches.
- The attn*V kernel declares a placeholder in spec slot 2, so its node shares the spec-constant list with
  release 1.5's GQA decode shader.
- `sarc_sdpa_attn_weights_softmax`: release 1.5's LLM softmax plus the 1.4 causal-tail truncation. The
  upstream SPIR-V is unchanged.

**Hooks** (`impl/sarc/SdpaCoopmat.cpp` + `SDPA.cpp`):
- They cover the pickers, the work-group grids, the spec constants and the softmax name.
- LLM mode only; they do nothing on devices without an SDPA row.

**Rows:**
- 780M: verified.
- M51 (Xclipse): unverified, owned by the M51 agent.

## Results (2026-09-27, Radeon 780M, release 1.5)

**SDPA correctness** (`test_llama_microbench --sdpa-correctness-only`): 4/4 PASSED with 0 mismatches.
Both coopmat kernels were dispatched in every case (`results/780m/sdpa2`).

**End to end**, 2048-token prefill tok/s, 1B / 3B / 8B, with the 4w/8da4w rows and the SDPA rows
(`results/780m/sdpa-e2e`):

| scheme | 1.5 tiled | 1.5 SARC | 1.4 tiled | 1.4 tuned |
|---|---|---|---|---|
| 4w | 797 / 283 / 114 | 2695 / 1138 / 516 | 801 / 283 / 114 | 2702 / 1142 / 516 |
| 8da4w | 2156 / 850 / 377 | 2544 / 1048 / 487 | 2151 / 858 / 375 | 2538 / 1055 / 486 |

- The 780M is back at its 1.4 numbers (within 0.7 %).
- The next token matches tiled on the real-text prompt and on the 1792-token prompt, for both schemes.
- Decode works.
- 8da4w production-diff passes 6/6.

**Other devices:** on the B580 (no SDPA row), 1B prefill is unchanged by this change (4w 8605, 8da4w
8790 tok/s).

**Not verified:** the M51 rows (no access here).
