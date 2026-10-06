#!/usr/bin/env python3
"""Candidate 5 (B, SDPA QK^T): packed staging, as a new dev-zone family sarc_sdpa_qk_coopmat_pk.

usage: gen_qkpk.py <executorch tree>      (run gen_sdpa.py first)

The release QK^T kernel stages Q and K^T into fp16-typed shared arrays: 4 scalar 16-bit LDS stores per loaded
f16vec4, and K is transposed on write (4 stores to 4 different LDS rows). With head_dim = 64 or 128 the kernel
runs only 2 or 4 K-chunks, so staging, not the MMA, dominates a tile. Here Q rows and K rows are staged the way
they are stored (d contiguous), 8 fp16 per uvec4 store, and K^T comes from a ColumnMajor coopMatLoad of the K
rows (the attn*V kernel already stages this way). WG_TILE_K may be 64: the shader derives its chunk count from
the release spec constant (head_dim / 32). Includes NO_MASK_FILL (gen_sdpa.py). Arithmetic is unchanged: the
same fp16 x fp16 -> fp32 MMAs in the same order per output element."""
import pathlib, sys
root = pathlib.Path(sys.argv[1]) / "backends/vulkan/runtime/graph/ops"
g = root / "glsl"
def sub(text, old, new, count=1):
    n = text.count(old); assert n == count, f"anchor occurs {n} times, expected {count}: {old[:70]!r}"
    return text.replace(old, new)
q = (g / "sarc_dev/sarc_sdpa_qk_coopmat_sweep.glsl").read_text()
q = q[q.index("#version 450 core"):]
q = sub(q, """const uint FP16_PER_VEC4 = 4; // we read native tensors as f16vec4 (4 fp16)
""", """const uint F16_PER_UV4 = 8; // one shared uvec4 holds 8 fp16 (two native f16vec4)
""")
a0 = q.index("// fp16 shared tiles with skew padding."); a1 = q.index("shared float16_t Csh[WG_TILE_M * WG_TILE_N];")
q = q[:a0] + """// Packed staging: A = Q rows [s][d], B = K rows [c][d] (K^T comes from a
// ColumnMajor load), both d-contiguous with one uvec4 of skew padding per row.
// C = scaled result [s][c] scratch for the masked scalar store (fallback path).
const uint AB_STRIDE_UV4 = (WG_TILE_K + F16_PER_UV4) / F16_PER_UV4;

shared uvec4 Ash[WG_TILE_M * AB_STRIDE_UV4];
shared uvec4 Bsh[WG_TILE_N * AB_STRIDE_UV4];
""" + q[a1:]
q = sub(q, "    for (uint chunk = 0; chunk < uint(num_k_chunks_arg); ++chunk) {",
        "    // num_k_chunks_arg counts 32-wide chunks (head_dim / 32, from the table row).\n    const uint num_chunks = uint(num_k_chunks_arg) * 32u / WG_TILE_K;\n    const uint UV4_PER_CHUNK = WG_TILE_K / F16_PER_UV4;\n    for (uint chunk = 0; chunk < num_chunks; ++chunk) {")
s0 = q.index("        // --- Stage A = Q [s][d] into fp16 shared (contiguous d) ---"); s1 = q.index("        // coopmat-lds-fence (WRITE->READ): the Ash/Bsh staging stores above")
q = q[:s0] + """        // --- Stage A = Q rows, 8 fp16 per uvec4 store ---
        for (uint idx = gl_LocalInvocationID.x; idx < WG_TILE_M * UV4_PER_CHUNK;
             idx += WG_SIZE) {
            const uint ls = idx / UV4_PER_CHUNK;      // local s row
            const uint l8 = idx % UV4_PER_CHUNK;      // uvec4 within the chunk
            const uint src = (s_tile_base + ls) * q_row_stride + q_head_base +
                d4_chunk + l8 * 2u;
            const f16vec4 v0 = t_q[src];
            const f16vec4 v1 = t_q[src + 1u];
            Ash[ls * AB_STRIDE_UV4 + l8] = uvec4(
                packFloat2x16(v0.xy), packFloat2x16(v0.zw),
                packFloat2x16(v1.xy), packFloat2x16(v1.zw));
        }

        // --- Stage B = K rows [c][d] (no transpose; loaded ColumnMajor) ---
        for (uint idx = gl_LocalInvocationID.x; idx < WG_TILE_N * UV4_PER_CHUNK;
             idx += WG_SIZE) {
            const uint lc = idx / UV4_PER_CHUNK;      // local c row of K
            const uint l8 = idx % UV4_PER_CHUNK;
            const uint src = (c_tile_base + lc) * k_row_stride + k_head_base +
                d4_chunk + l8 * 2u;
            const f16vec4 v0 = t_k[src];
            const f16vec4 v1 = t_k[src + 1u];
            Bsh[lc * AB_STRIDE_UV4 + l8] = uvec4(
                packFloat2x16(v0.xy), packFloat2x16(v0.zw),
                packFloat2x16(v1.xy), packFloat2x16(v1.zw));
        }

""" + q[s1:]
q = sub(q, """                coopMatLoad(
                    matA[i], Ash,
                    row_a * A_ROW + k_start,
                    A_ROW,
                    gl_CooperativeMatrixLayoutRowMajor);""", """                coopMatLoad(
                    matA[i], Ash,
                    row_a * AB_STRIDE_UV4 + k_start / F16_PER_UV4,
                    AB_STRIDE_UV4,
                    gl_CooperativeMatrixLayoutRowMajor);""")
q = sub(q, """                coopMatLoad(
                    matB, Bsh,
                    k_start * B_ROW + col_b,
                    B_ROW,
                    gl_CooperativeMatrixLayoutRowMajor);""", """                coopMatLoad(
                    matB, Bsh,
                    col_b * AB_STRIDE_UV4 + k_start / F16_PER_UV4,
                    AB_STRIDE_UV4,
                    gl_CooperativeMatrixLayoutColumnMajor);""")
assert "A_ROW" not in q and "B_ROW" not in q and "FP16_PER_VEC4" not in q
(g / "sarc_dev/sarc_sdpa_qk_coopmat_pk.glsl").write_text("""/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone: QK^T with packed staging (generated by gen_qkpk.py from the sweep twin of
 * glsl/sarc/sarc_sdpa_qk_coopmat.glsl; openspec/changes/sarc-1.5-780m-prefill-refine). See that shader for the
 * algorithm; the differences are the uvec4 staging of Q and K rows, the ColumnMajor B load and WG_TILE_K 64.
 */

""" + q)
# (token, M, N, K, sgx, sgy, subgroup)
V = [("t128x64k32g22s64nf", 128, 64, 32, 2, 2, 64), ("t128x64k64g22s64nf", 128, 64, 64, 2, 2, 64),
     ("t128x64k32g42s32nf", 128, 64, 32, 4, 2, 32), ("t128x64k64g42s32nf", 128, 64, 64, 4, 2, 32)]
y = "# SARC development zone: QK^T with packed staging (generated by gen_qkpk.py). Not shipped.\n\nsarc_sdpa_qk_coopmat_pk:\n"
y += "  parameter_names_with_default_values:\n    DTYPE: half\n    PRECISION: highp\n    IO_STORAGE: buffer\n    K_CACHE_STORAGE: buffer\n    MMA_M: 16\n    MMA_N: 16\n    MMA_K: 16\n"
y += "    WG_TILE_M: 128\n    WG_TILE_N: 64\n    WG_TILE_K: 32\n    SG_GRID_X: 2\n    SG_GRID_Y: 2\n    SUBGROUP_SIZE: 64\n    NO_MASK_FILL: true\n"
y += "  generate_variant_forall:\n    combination:\n      parameter_names: [IO_STORAGE, K_CACHE_STORAGE]\n      combos:\n        - parameter_values: [buffer, buffer]\n    DTYPE:\n      - VALUE: half\n  shader_variants:\n"
for tok, m, n, k, sx, sy, sg in V:
    assert 16 * (m + n) * ((k + 8) // 8) + 2 * m * n <= 60000, tok
    y += f"    - NAME: sarc_sdpa_qk_coopmat_pk_{tok}\n      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {sg}\n"
(g / "sarc_dev/sarc_sdpa_qk_coopmat_pk.yaml").write_text(y)
o = root / "impl/sarc_dev/Overrides.cpp"; t = o.read_text()
if "sarc_sdpa_qk_coopmat_pk" not in t:
    a = "};\n\nstd::string& requested_variant() {"
    rows = "    // 780M prefill refine 2026-10-03, candidate 5: QK^T with packed staging\n    // (glsl/sarc_dev/sarc_sdpa_qk_coopmat_pk.yaml).\n"
    for tok, m, n, k, sx, sy, sg in V:
        rows += f'    {{"", nullptr, Op::kSdpaQk,\n     "sarc_sdpa_qk_coopmat_pk_{tok}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {sg}, 16, false}}, kBufBuf, nullptr,\n     Status::kUnverified}},\n'
    t = sub(t, a, rows + a)
    a = "struct Profile {\n"
    prefs = "".join(f'const Preference kQkPk_{tok}[] = {{{{Op::kSdpaQk, "pk_{tok}", nullptr}}}};\n' for tok, *_ in V)
    t = sub(t, a, prefs + a)
    a = '    {"780m-refine1-dq",'
    t = sub(t, a, "".join(f'    {{"qkpk-{tok}", kQkPk_{tok}, 1}},\n' for tok, *_ in V) + a)
    o.write_text(t)
print("qk pk family generated")
