#!/usr/bin/env python3
"""Batch 1 of 780M sweep candidates (dev zone only): appends variants to the zpg and q4gsw sweep yamls and
candidate rows to impl/sarc_dev/Overrides.cpp. Idempotent. usage: add_batch1.py <executorch tree>"""
import pathlib, sys
root = pathlib.Path(sys.argv[1]) / "backends/vulkan/runtime/graph/ops"
MARK = "780M prefill refine 2026-10-03, batch 1"

# (token, M, N, K, sgx, sgy, a_blocks) -- 8da4w zpg, MMA 16x16x16, subgroup 32, A map full by construction:
# A blocks per chunk (M/4)*(K/4) == a_blocks * WG_SIZE, B slots per thread (K/4)*N / WG_SIZE is an integer.
DQ = [("t128x64k64g22s32", 128, 64, 64, 2, 2, 4), ("t128x64k64g42s32", 128, 64, 64, 4, 2, 2),
      ("t64x64k32g21s32", 64, 64, 32, 2, 1, 2), ("t128x128k32g22s32", 128, 128, 32, 2, 2, 2),
      ("t64x64k32g22s32", 64, 64, 32, 2, 2, 1), ("t64x128k32g22s32", 64, 128, 32, 2, 2, 1),
      ("t128x64k32g24s32", 128, 64, 32, 2, 4, 1), ("t128x32k32g22s32", 128, 32, 32, 2, 2, 2),
      ("t256x64k32g42s32", 256, 64, 32, 4, 2, 2), ("t128x64k32g21s32", 128, 64, 32, 2, 1, 4),
      ("t128x64k32g12s32", 128, 64, 32, 1, 2, 4)]
y = root / "glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpg_sweep.yaml"; t = y.read_text()
if MARK not in t:
    t += f"    # {MARK}: per-wave tile shape / workgroup size / chunk depth around the\n    # release t128x64k32g42s32 (phase timing: staging 40 %, barrier 13 %, MMA 38 % of a wave).\n"
    for tok, m, n, k, sx, sy, ab in DQ:
        wg = sx * sy * 32
        assert (m // 4) * (k // 4) == ab * wg and ((k // 4) * n) % wg == 0, tok
        t += f"    - NAME: sarc_linear_dq8ca_coopmat_zpg_sweep_{tok}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n"
        t += f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      A_MAP_FULL: true\n"
        if ab > 1: t += f"      A_MULTI_BLOCK: true\n      A_BLOCKS: {ab}\n"
    y.write_text(t)

# (token, M, N, K, sgx, sgy) -- 4w, ACC_FP32 + CSH_IN_ASH (the 780M flags)
Q4 = [("t128x128k32g22s32f32c", 128, 128, 32, 2, 2), ("t128x128k64g42s32f32c", 128, 128, 64, 4, 2),
      ("t256x128k32g42s32f32c", 256, 128, 32, 4, 2), ("t128x256k64g42s32f32c", 128, 256, 64, 4, 2),
      ("t128x128k64g22s32f32c", 128, 128, 64, 2, 2)]
y = root / "glsl/sarc_dev/sarc_linear_q4gsw_coopmat_sweep.yaml"; t = y.read_text()
if MARK not in t:
    t += f"    # {MARK}: workgroup size / chunk depth around the release t128x128k32g42s32f32c\n    # (phase timing: staging 29 %, barrier 16 %, MMA 51 % of a wave).\n"
    for tok, m, n, k, sx, sy in Q4:
        wg = sx * sy * 32
        assert m % (wg // (k // 8)) == 0 and k % (wg // (n // 8)) == 0, tok          # A_PASSES, B_PASSES integral
        assert 2 * m * ((k + 8) // 8) >= sy * 16 * (n // 8), tok                       # drain band fits in Ash
        t += f"    - NAME: sarc_linear_q4gsw_coopmat_sweep_{tok}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n      WEIGHT_STORAGE: texture2d\n"
        t += f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: 32\n      ACC_FP32: true\n      CSH_IN_ASH: true\n"
    y.write_text(t)

o = root / "impl/sarc_dev/Overrides.cpp"; t = o.read_text()
if MARK not in t:
    a = "    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_q4gsw.yaml):"
    assert t.count(a) == 1
    rows = f"    // {MARK} (texture3d screen).\n"
    for tok, m, n, k, sx, sy in Q4:
        rows += f'    {{"", nullptr, Op::kQ4gswLinear,\n     "sarc_linear_q4gsw_coopmat_sweep_{tok}", tile_mnk({m}, {n}, {k}, {sx}, {sy}, true),\n     kTex3dTex2d, nullptr, Status::kUnverified}},\n'
    t = t.replace(a, rows + a)
    a = "    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_dq8ca_zpg.yaml)."
    assert t.count(a) == 1
    rows = f"    // {MARK} (texture3d screen).\n"
    for tok, m, n, k, sx, sy, ab in DQ:
        rows += f'    {{"", nullptr, Op::kDq8caLinear,\n     "sarc_linear_dq8ca_coopmat_zpg_sweep_{tok}",\n     {{{m}, {n}, {k}, {sx}, {sy}, 32, 16, false}}, kTex3dTex2d, nullptr,\n     Status::kUnverified}},\n'
    t = t.replace(a, rows + a)
    o.write_text(t)
print("batch 1 added:", [x[0] for x in DQ], [x[0] for x in Q4])
