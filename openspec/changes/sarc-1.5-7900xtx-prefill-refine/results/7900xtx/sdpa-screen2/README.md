# Second attention screen (2026-10-09 03:33 to 03:41 UTC)

`tools/q-sdpa2.sh` -> `tools/sdpa_screen2.sh`: `test_llama_microbench --sdpa`, build `c6b`, exact kernel names (`ET_VK_SARC_7900XTX_QK` / `_AV`), 3 rounds, order rotated,
one `gl.sh` job per (round, item); the incumbents are candidate 5's kernels (QK^T `pk_t128x128k32g42s32nf`, attn*V `sweep_t64x64k32g42s32`). Variants added
for it: QK^T `pk` 128 x 128 with grids 2x4, 2x2 (subgroup 64), 4x4, 2x2 (subgroup 32) and 64 x 128 with 4x2; attn*V `sweep` 128 x 128 with 4x4 / 8x2 / 4x2 (subgroup 64) and
32 x 32 with 2x2 (a ninth variant, 32 x 32 with 4x1, did not compile: one subgroup row gives a zero-sized array; its build is in the superseded directory).
Rule (R8, `tools/sdpa_pick.py`): at least 3 % faster than the incumbent in every round on every model of the head dimension. Result: only attn*V `sweep_t32x32k32g22s32` for
head dimension 64 (1B), 1.16x in every round on that op; for head dimension 128 nothing qualifies (the large tiles are 0.62 to 0.90x, the QK^T grids 0.72 to 1.02x).
