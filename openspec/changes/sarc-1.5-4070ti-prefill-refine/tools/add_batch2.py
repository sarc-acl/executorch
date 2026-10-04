#!/usr/bin/env python3
"""Batch 2 of 780M sweep candidates (dev zone only): wave64 (SUBGROUP_SIZE 64) tiles. Idempotent.
Every tile is checked against the 64 KiB shared-memory limit before it is written (batch 1's 4w k64 tiles
exceeded it and hung the GPU). usage: add_batch2.py <executorch tree>"""
import pathlib, sys
root = pathlib.Path(sys.argv[1]) / "backends/vulkan/runtime/graph/ops"
MARK = "780M prefill refine 2026-10-03, batch 2"
LDS_MAX = 60000
# 8da4w zpg: (token, M, N, K, sgx, sgy, sg, a_blocks)
DQ = [("t128x64k32g22s64", 128, 64, 32, 2, 2, 64, 1), ("t128x64k32g21s64", 128, 64, 32, 2, 1, 64, 2),
      ("t128x64k32g12s64", 128, 64, 32, 1, 2, 64, 2), ("t128x128k32g22s64", 128, 128, 32, 2, 2, 64, 1),
      ("t256x64k32g22s64", 256, 64, 32, 2, 2, 64, 2), ("t128x64k64g22s64", 128, 64, 64, 2, 2, 64, 2)]
y = root / "glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpg_sweep.yaml"; t = y.read_text()
if MARK not in t:
    t += f"    # {MARK}: wave64 tiles (a 16x16 accumulator takes 4 VGPRs instead of 8).\n"
    for tok, m, n, k, sx, sy, sg, ab in DQ:
        wg = sx * sy * sg
        assert (m // 4) * (k // 4) == ab * wg and ((k // 4) * n) % wg == 0, tok
        lds = 2 * m * k + 2 * n * k + 8 * m + 12 * n + sy * 16 * n * 2          # Ash, Bsh, izp+ifs, wsc+wcorr, Csh_out
        assert lds <= LDS_MAX, (tok, lds)
        t += f"    - NAME: sarc_linear_dq8ca_coopmat_zpg_sweep_{tok}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n"
        t += f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {sg}\n      A_MAP_FULL: true\n"
        if ab > 1: t += f"      A_MULTI_BLOCK: true\n      A_BLOCKS: {ab}\n"
    y.write_text(t)
# 4w: (token, M, N, K, sgx, sgy, sg)
Q4 = [("t128x128k32g22s64f32c", 128, 128, 32, 2, 2, 64), ("t128x256k32g22s64f32c", 128, 256, 32, 2, 2, 64),
      ("t128x128k32g21s64f32c", 128, 128, 32, 2, 1, 64), ("t128x128k32g12s64f32c", 128, 128, 32, 1, 2, 64),
      ("t256x128k32g22s64f32c", 256, 128, 32, 2, 2, 64)]
y = root / "glsl/sarc_dev/sarc_linear_q4gsw_coopmat_sweep.yaml"; t = y.read_text()
if MARK not in t:
    t += f"    # {MARK}: wave64 tiles.\n"
    for tok, m, n, k, sx, sy, sg in Q4:
        wg = sx * sy * sg
        assert m % (wg // (k // 8)) == 0 and k % (wg // (n // 8)) == 0, tok
        assert 2 * m * ((k + 8) // 8) >= sy * 16 * (n // 8), tok
        lds = 4 * (m * (k + 8) + k * (n + 8)); assert lds <= LDS_MAX, (tok, lds)
        t += f"    - NAME: sarc_linear_q4gsw_coopmat_sweep_{tok}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n      WEIGHT_STORAGE: texture2d\n"
        t += f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {sg}\n      ACC_FP32: true\n      CSH_IN_ASH: true\n"
    y.write_text(t)
o = root / "impl/sarc_dev/Overrides.cpp"; t = o.read_text()
if MARK not in t:
    a = "    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_q4gsw.yaml):"
    assert t.count(a) == 1
    rows = f"    // {MARK} (wave64, texture3d screen).\n"
    for tok, m, n, k, sx, sy, sg in Q4:
        rows += f'    {{"", nullptr, Op::kQ4gswLinear,\n     "sarc_linear_q4gsw_coopmat_sweep_{tok}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {sg}, 16, true}}, kTex3dTex2d, nullptr,\n     Status::kUnverified}},\n'
    t = t.replace(a, rows + a)
    a = "    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_dq8ca_zpg.yaml)."
    assert t.count(a) == 1
    rows = f"    // {MARK} (wave64, texture3d screen).\n"
    for tok, m, n, k, sx, sy, sg, ab in DQ:
        rows += f'    {{"", nullptr, Op::kDq8caLinear,\n     "sarc_linear_dq8ca_coopmat_zpg_sweep_{tok}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {sg}, 16, false}}, kTex3dTex2d, nullptr,\n     Status::kUnverified}},\n'
    t = t.replace(a, rows + a)
    o.write_text(t)
print("batch 2 added:", [x[0] for x in DQ], [x[0] for x in Q4])
