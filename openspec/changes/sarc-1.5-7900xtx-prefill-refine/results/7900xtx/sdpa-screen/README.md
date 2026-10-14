# Unfused attention kernel screen (2026-10-09 00:33 to 00:47 UTC)

`tools/q-sdpa.sh` -> `tools/sdpa_screen.sh`: `test_llama_microbench --sdpa` (`ET_VK_SDPA_PERF_RUNS=20,8`) with the dev zone's named
`ET_VK_SARC_DEV_PROFILE` profiles for QK^T (no mask fill `nf`, packed K staging `pk`) and attn*V (`av`, multi-load `ml`), on the parent build with
`ET_VK_SARC_780M_PROFILE=c7` (softmax r3), 3 rounds, order rotated per round, one `gl.sh` job per (round, profile); per model at S = 2048 the mean time of
the qk, softmax and av ops. Rule (R8): a kernel replaces the table kernel of an op only if at least 3 % faster in every round on every model of the head
dimension (d64: 1B; d128: 3B, 8B); the best worst-round speedup wins (`tools/sdpa_pick.py`: `pick-analysis.txt`, `pick.out`).
Result: QK^T `pk_t128x128k32g42s32nf` (1.30x for d64, 1.45x for d128, worst round) and attn*V `sweep_t64x64k32g42s32` (1.08x / 1.10x).
