#!/usr/bin/env python3
"""Xe2 (Arc B580 / Arc Pro B70) dev-zone content of sarc-1.5-xe2-prefill-refine, generated.

usage: gen_xe2.py <executorch tree>

Writes, and rewrites idempotently:
  - the xe2 blocks appended to the shared dev-zone sweep yamls (SDPA QK^T sweep / pk, attn*V sweep / ml):
    the 780M twins compiled for the shapes Xe2 exposes (MMA 8x16x16 fp16 x fp16 -> fp32, subgroup 16);
  - glsl/sarc_dev/sarc_sdpa_{qk,av}_coopmat_xe2.{glsl,yaml}: new Xe2 families with a fragment-contiguous
    shared-memory layout (what FRAG_LAYOUT is for the 4w kernel): A as [k16 block][m][16 fp16] and B as
    [n16 block][k][16 fp16], uvec4 stores, RowMajor loads with a 32-byte stride. QK^T builds K^T by gathering
    8 context rows per thread (4 uvec4 stores) instead of 4 scalar fp16 stores per loaded texel. Generated from
    the dev twins; the arithmetic (fp16 x fp16 -> fp32 MMAs, same order per output element) is unchanged;
  - impl/sarc_dev/Xe2Sdpa.cpp: the Xe2 SDPA base rows (kUnverified, active only while an xe2-* profile is
    requested) and the Xe2 candidate rows;
  - the xe2 blocks of impl/sarc_dev/Overrides.cpp (profiles) and of test_llama_microbench.cpp (pairing check).
Never writes a release-zone file. Blocks are delimited by "xe2 begin" / "xe2 end" comments.

Static pruning: a variant is emitted only if its shared memory fits the device limit (49152 bytes on the B70; the
B580 was not queried here) with margin, its workgroup has at most 1024 invocations, the subgroup tile is a whole number of 8x16 MMA tiles
and (single-pass attn*V) WG_SIZE == 4 * WG_TILE_M == WG_TILE_K * WG_TILE_N / 8.
"""
import pathlib, re, sys
root = pathlib.Path(sys.argv[1]) / "backends/vulkan"
g = root / "runtime/graph/ops/glsl/sarc_dev"
impl = root / "runtime/graph/ops/impl/sarc_dev"
LDS_MAX = 46000      # bytes; maxComputeSharedMemorySize is 49152 on BMG G31 (B70)
SG = 16              # subgroup size of every Xe2 kernel
MMA_M = 8            # Xe2 exposes 8x16x16 only (a 16x16x16 pipeline miscomputes)

def block(text, begin, end, body, anchor=None, before=True):
    """Replace the delimited block, or insert it at the anchor (or at the end of the text)."""
    if begin in text:
        a = text.index(begin); b = text.index(end, a) + len(end)
        return text[:a] + begin + body + end + text[b:]
    new = begin + body + end
    if anchor is None:
        return text.rstrip("\n") + "\n" + new
    assert text.count(anchor) == 1, f"anchor occurs {text.count(anchor)} times: {anchor[:60]!r}"
    return text.replace(anchor, (new + anchor) if before else (anchor + new))

def tok(m, n, k, sx, sy, suffix=""):
    return f"t{m}x{n}k{k}g{sx}{sy}s{SG}m8{suffix}"

def geometry_ok(m, n, sx, sy):
    wg = sx * sy * SG
    return wg <= 1024 and m % sy == 0 and n % sx == 0 and (m // sy) % MMA_M == 0 and (n // sx) % 16 == 0

# family -> list of (token, M, N, K, sgx, sgy, extra yaml lines)
QK_SWEEP, QK_PK, AV_SWEEP, AV_ML = [], [], [], []
for m, n, sx, sy, nf in [(128, 64, 4, 4, True), (128, 64, 4, 4, False), (128, 64, 4, 8, True), (128, 64, 2, 4, True),
                         (64, 64, 4, 4, True), (64, 64, 2, 4, True), (64, 128, 4, 4, True), (64, 128, 8, 4, True)]:
    assert geometry_ok(m, n, sx, sy), (m, n, sx, sy)
    assert 2 * (m * 40 + 32 * (n + 8) + m * n) <= LDS_MAX, (m, n)       # Ash + Bsh + Csh, fp16
    QK_SWEEP.append((tok(m, n, 32, sx, sy, "nf" if nf else ""), m, n, 32, sx, sy, "      NO_MASK_FILL: true\n" if nf else ""))
for m, n, k, sx, sy in [(128, 64, 32, 4, 4), (128, 64, 64, 4, 4), (128, 64, 32, 4, 8), (128, 64, 32, 2, 4),
                        (64, 64, 32, 4, 4), (64, 64, 64, 4, 4), (64, 128, 32, 4, 4), (64, 128, 32, 8, 4)]:
    assert geometry_ok(m, n, sx, sy), (m, n, sx, sy)
    assert 16 * (m + n) * ((k + 8) // 8) + 2 * m * n <= LDS_MAX, (m, n, k)
    QK_PK.append((tok(m, n, k, sx, sy, "nf"), m, n, k, sx, sy, ""))       # the pk family defaults to NO_MASK_FILL
for m, n, sx, sy in [(64, 64, 4, 4)]:
    assert geometry_ok(m, n, sx, sy) and sx * sy * SG == 4 * m == 32 * n // 8
    AV_SWEEP.append((tok(m, n, 32, sx, sy), m, n, 32, sx, sy, ""))
for m, n, sx, sy in [(64, 64, 4, 4), (64, 64, 2, 4), (64, 64, 4, 8), (128, 64, 4, 4), (128, 64, 4, 8), (64, 128, 4, 4),
                     (64, 128, 8, 4), (128, 128, 4, 8), (128, 128, 8, 8), (32, 64, 4, 2)]:
    assert geometry_ok(m, n, sx, sy), (m, n, sx, sy)
    assert 16 * (m * 5 + 32 * ((n + 8) // 8)) <= LDS_MAX, (m, n)
    AV_ML.append((tok(m, n, 32, sx, sy), m, n, 32, sx, sy, ""))

QK_XE2, AV_XE2 = [], []
for m, n, k, sx, sy in [(128, 64, 32, 4, 4), (128, 64, 32, 4, 8), (128, 64, 32, 2, 4), (128, 64, 64, 4, 4),
                        (64, 64, 32, 4, 4), (64, 64, 32, 2, 4), (64, 64, 64, 4, 4), (64, 128, 32, 4, 4), (64, 128, 32, 8, 4)]:
    assert geometry_ok(m, n, sx, sy), (m, n, sx, sy)
    assert 2 * (m * k + k * n + m * n) <= LDS_MAX, (m, n, k)             # Ash + Bsh (unpadded) + Csh
    QK_XE2.append((tok(m, n, k, sx, sy, "nf"), m, n, k, sx, sy, ""))
for t, m, n, k, sx, sy, _ in AV_ML:
    assert 2 * (m * k + k * n) <= LDS_MAX
    AV_XE2.append((t, m, n, k, sx, sy, ""))
# screen 2: around the screen-1 best attn*V tile (t128x64k32g44)
for m, n, sx, sy in [(128, 64, 2, 4), (256, 64, 4, 8), (256, 64, 4, 4), (128, 128, 4, 4), (256, 128, 4, 8)]:
    assert geometry_ok(m, n, sx, sy) and 2 * (m * 32 + 32 * n) <= LDS_MAX, (m, n, sx, sy)
    AV_XE2.append((tok(m, n, 32, sx, sy), m, n, 32, sx, sy, ""))
# QK^T, packed staging + ColumnMajor B (the pk twin, the screen-1 best) with the fragment-contiguous layout
QK_XE2C = []
for m, n, k, sx, sy in [(128, 64, 32, 4, 4), (128, 64, 32, 2, 4), (64, 128, 32, 4, 4), (64, 64, 32, 4, 4), (128, 64, 64, 4, 4), (64, 128, 32, 4, 2)]:
    assert geometry_ok(m, n, sx, sy) and 2 * (m * k + k * n + m * n) <= LDS_MAX, (m, n, k, sx, sy)
    QK_XE2C.append((tok(m, n, k, sx, sy, "nf"), m, n, k, sx, sy, ""))

def sub(text, old, new, count=1):
    c = text.count(old); assert c == count, f"anchor occurs {c} times, expected {count}: {old[:70]!r}"
    return text.replace(old, new)
HDR = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, Intel Xe2 (openspec/changes/sarc-1.5-xe2-prefill-refine, generated by its
 * tools/gen_xe2.py from glsl/sarc_dev/%s.glsl; see glsl/sarc/%s.glsl for the algorithm).
 * %s
 */

"""
FRAG = """// Fragment-contiguous shared memory (Intel Xe2): A as [k16 block][m][16 fp16]
// and B as [n16 block][k][16 fp16], 8 fp16 per uvec4, so every MMA_M x 16 A
// fragment and 16 x MMA_N B fragment is one contiguous run with a 32-byte row
// stride and no padding. Requires MMA_K == MMA_N == 16.
const uint FRAG_ROW_UV4 = 2u;
// (row, uvec4 column) -> index
#define A_SH_IDX(m, c) ((((c) / FRAG_ROW_UV4) * WG_TILE_M + (m)) * FRAG_ROW_UV4 + ((c) % FRAG_ROW_UV4))
#define B_SH_IDX(k, c) ((((c) / FRAG_ROW_UV4) * WG_TILE_K + (k)) * FRAG_ROW_UV4 + ((c) % FRAG_ROW_UV4))

shared uvec4 Ash[WG_TILE_M * WG_TILE_K / 8u];
shared uvec4 Bsh[WG_TILE_K * WG_TILE_N / 8u];
"""
# --- QK^T ---
q = (g / "sarc_sdpa_qk_coopmat_sweep.glsl").read_text(); q = q[q.index("#version 450 core"):]
q = sub(q, "const uint FP16_PER_VEC4 = 4; // we read native tensors as f16vec4 (4 fp16)\n",
        "const uint F16_PER_UV4 = 8; // one shared uvec4 holds 8 fp16 (two native f16vec4)\n")
a0 = q.index("// fp16 shared tiles with skew padding."); a1 = q.index("shared float16_t Csh[WG_TILE_M * WG_TILE_N];")
q = q[:a0] + FRAG + "// C = scaled result [s][c] scratch for the masked scalar store (fallback path).\n" + q[a1:]
q = sub(q, "    for (uint chunk = 0; chunk < uint(num_k_chunks_arg); ++chunk) {",
        "    // num_k_chunks_arg counts 32-wide chunks (head_dim / 32, from the table row).\n"
        "    const uint num_chunks = uint(num_k_chunks_arg) * 32u / WG_TILE_K;\n"
        "    const uint UV4_PER_CHUNK = WG_TILE_K / F16_PER_UV4;\n"
        "    for (uint chunk = 0; chunk < num_chunks; ++chunk) {")
s0 = q.index("        // --- Stage A = Q [s][d] into fp16 shared (contiguous d) ---")
s1 = q.index("        // coopmat-lds-fence (WRITE->READ): the Ash/Bsh staging stores above")
q = q[:s0] + """        // --- Stage A = Q rows [s][d], 8 fp16 per uvec4 store ---
        for (uint idx = gl_LocalInvocationID.x; idx < WG_TILE_M * UV4_PER_CHUNK;
             idx += WG_SIZE) {
            const uint ls = idx / UV4_PER_CHUNK;      // local s row
            const uint l8 = idx % UV4_PER_CHUNK;      // uvec4 within the chunk
            const uint src = (s_tile_base + ls) * q_row_stride + q_head_base +
                d4_chunk + l8 * 2u;
            const f16vec4 v0 = t_q[src];
            const f16vec4 v1 = t_q[src + 1u];
            Ash[A_SH_IDX(ls, l8)] = uvec4(
                packFloat2x16(v0.xy), packFloat2x16(v0.zw),
                packFloat2x16(v1.xy), packFloat2x16(v1.zw));
        }

        // --- Stage B = K^T [d][c]: one thread gathers the same 4 d of 8
        // consecutive context rows and writes 4 uvec4 (rows d..d+3, 8 c each) ---
        for (uint idx = gl_LocalInvocationID.x;
             idx < (WG_TILE_N / F16_PER_UV4) * VEC4_PER_CHUNK; idx += WG_SIZE) {
            const uint c8 = idx / VEC4_PER_CHUNK;     // uvec4 column (8 context rows)
            const uint ld4 = idx % VEC4_PER_CHUNK;    // local d4 within chunk
            const uint src = (c_tile_base + c8 * 8u) * k_row_stride + k_head_base +
                d4_chunk + ld4;
            f16vec4 v[8];
            [[unroll]] for (uint i = 0u; i < 8u; ++i) {
                v[i] = t_k[src + i * k_row_stride];
            }
            [[unroll]] for (uint j = 0u; j < 4u; ++j) {
                Bsh[B_SH_IDX(ld4 * 4u + j, c8)] = uvec4(
                    packFloat2x16(f16vec2(v[0][j], v[1][j])),
                    packFloat2x16(f16vec2(v[2][j], v[3][j])),
                    packFloat2x16(f16vec2(v[4][j], v[5][j])),
                    packFloat2x16(f16vec2(v[6][j], v[7][j])));
            }
        }

""" + q[s1:]
q = sub(q, """                coopMatLoad(
                    matA[i], Ash,
                    row_a * A_ROW + k_start,
                    A_ROW,
                    gl_CooperativeMatrixLayoutRowMajor);""", """                coopMatLoad(
                    matA[i], Ash,
                    A_SH_IDX(row_a, k_start / F16_PER_UV4),
                    FRAG_ROW_UV4,
                    gl_CooperativeMatrixLayoutRowMajor);""")
q = sub(q, """                coopMatLoad(
                    matB, Bsh,
                    k_start * B_ROW + col_b,
                    B_ROW,
                    gl_CooperativeMatrixLayoutRowMajor);""", """                coopMatLoad(
                    matB, Bsh,
                    B_SH_IDX(k_start, col_b / F16_PER_UV4),
                    FRAG_ROW_UV4,
                    gl_CooperativeMatrixLayoutRowMajor);""")
assert "A_ROW" not in q and "B_ROW" not in q and "FP16_PER_VEC4" not in q
(g / "sarc_sdpa_qk_coopmat_xe2.glsl").write_text(HDR % ("sarc_sdpa_qk_coopmat_sweep", "sarc_sdpa_qk_coopmat",
    "Differences: fragment-contiguous uvec4 staging of Q and K^T (K^T gathered per thread), WG_TILE_K 32 or 64.") + q)
# --- QK^T, ColumnMajor B: K rows staged as stored, [d16 block][c][16 d] ---
qc = (g / "sarc_sdpa_qk_coopmat_pk.glsl").read_text(); qc = qc[qc.index("#version 450 core"):]
qc = sub(qc, """const uint AB_STRIDE_UV4 = (WG_TILE_K + F16_PER_UV4) / F16_PER_UV4;

shared uvec4 Ash[WG_TILE_M * AB_STRIDE_UV4];
shared uvec4 Bsh[WG_TILE_N * AB_STRIDE_UV4];
""", """// Fragment-contiguous (Intel Xe2): A as [d16 block][s][16 d] and B = K rows as
// [d16 block][c][16 d], no padding, so each MMA_M x 16 A fragment and each
// 16 x 16 B fragment (loaded ColumnMajor: one context row per column) is one
// contiguous run with a 32-byte stride. Requires MMA_K == MMA_N == 16.
const uint FRAG_ROW_UV4 = 2u;
#define A_SH_IDX(m, c) ((((c) / FRAG_ROW_UV4) * WG_TILE_M + (m)) * FRAG_ROW_UV4 + ((c) % FRAG_ROW_UV4))
#define BT_SH_IDX(n, c) ((((c) / FRAG_ROW_UV4) * WG_TILE_N + (n)) * FRAG_ROW_UV4 + ((c) % FRAG_ROW_UV4))

shared uvec4 Ash[WG_TILE_M * WG_TILE_K / 8u];
shared uvec4 Bsh[WG_TILE_N * WG_TILE_K / 8u];
""")
qc = sub(qc, "Ash[ls * AB_STRIDE_UV4 + l8] = uvec4(", "Ash[A_SH_IDX(ls, l8)] = uvec4(")
qc = sub(qc, "Bsh[lc * AB_STRIDE_UV4 + l8] = uvec4(", "Bsh[BT_SH_IDX(lc, l8)] = uvec4(")
qc = sub(qc, """                    row_a * AB_STRIDE_UV4 + k_start / F16_PER_UV4,
                    AB_STRIDE_UV4,""", """                    A_SH_IDX(row_a, k_start / F16_PER_UV4),
                    FRAG_ROW_UV4,""")
qc = sub(qc, """                    col_b * AB_STRIDE_UV4 + k_start / F16_PER_UV4,
                    AB_STRIDE_UV4,""", """                    BT_SH_IDX(col_b, k_start / F16_PER_UV4),
                    FRAG_ROW_UV4,""")
assert "AB_STRIDE_UV4" not in qc
(g / "sarc_sdpa_qk_coopmat_xe2c.glsl").write_text(HDR % ("sarc_sdpa_qk_coopmat_pk", "sarc_sdpa_qk_coopmat",
    "Difference from the pk twin: fragment-contiguous shared-memory layout (no row padding).") + qc)
# --- attn*V ---
a = (g / "sarc_sdpa_av_coopmat_ml.glsl").read_text(); a = a[a.index("#version 450 core"):]
a = sub(a, """const uint A_STRIDE_VEC4 = (WG_TILE_K + FP16_PER_VEC4) / FP16_PER_VEC4;
const uint B_STRIDE_VEC4 = (WG_TILE_N + FP16_PER_VEC4) / FP16_PER_VEC4;

shared uvec4 Ash[WG_TILE_M * A_STRIDE_VEC4];
shared uvec4 Bsh[WG_TILE_K * B_STRIDE_VEC4];
""", FRAG)
a = sub(a, "Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(", "Ash[A_SH_IDX(a_row_offset, a_col)] = uvec4(")
a = sub(a, "Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(", "Bsh[B_SH_IDX(b_row_offset, b_col)] = uvec4(")
a = sub(a, """                    row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
                    A_STRIDE_VEC4,""", """                    A_SH_IDX(row_a, k_start / FP16_PER_VEC4),
                    FRAG_ROW_UV4,""")
a = sub(a, """                    k_start * B_STRIDE_VEC4 + col_b,
                    B_STRIDE_VEC4,""", """                    B_SH_IDX(k_start, col_b),
                    FRAG_ROW_UV4,""")
assert "STRIDE_VEC4" not in a
(g / "sarc_sdpa_av_coopmat_xe2.glsl").write_text(HDR % ("sarc_sdpa_av_coopmat_ml", "sarc_sdpa_av_coopmat",
    "Difference: fragment-contiguous shared-memory layout; multi-pass staging as in the ml twin.") + a)
def family_yaml(name, cache, extra_default, variants):
    y = f"# SARC development zone, Intel Xe2: {name} (generated by sarc-1.5-xe2-prefill-refine/tools/gen_xe2.py). Not shipped.\n\n{name}:\n"
    y += f"  parameter_names_with_default_values:\n    DTYPE: half\n    PRECISION: highp\n    IO_STORAGE: buffer\n    {cache}: buffer\n    MMA_M: {MMA_M}\n    MMA_N: 16\n    MMA_K: 16\n"
    y += f"    WG_TILE_M: 64\n    WG_TILE_N: 64\n    WG_TILE_K: 32\n    SG_GRID_X: 4\n    SG_GRID_Y: 4\n    SUBGROUP_SIZE: {SG}\n{extra_default}"
    y += f"  generate_variant_forall:\n    combination:\n      parameter_names: [IO_STORAGE, {cache}]\n      combos:\n        - parameter_values: [buffer, buffer]\n    DTYPE:\n      - VALUE: half\n  shader_variants:\n"
    for t, m, n, k, sx, sy, _ in variants:
        y += f"    - NAME: {name}_{t}\n      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n"
    return y
(g / "sarc_sdpa_qk_coopmat_xe2.yaml").write_text(family_yaml("sarc_sdpa_qk_coopmat_xe2", "K_CACHE_STORAGE", "    NO_MASK_FILL: true\n", QK_XE2))
(g / "sarc_sdpa_qk_coopmat_xe2c.yaml").write_text(family_yaml("sarc_sdpa_qk_coopmat_xe2c", "K_CACHE_STORAGE", "    NO_MASK_FILL: true\n", QK_XE2C))
(g / "sarc_sdpa_av_coopmat_xe2.yaml").write_text(family_yaml("sarc_sdpa_av_coopmat_xe2", "V_CACHE_STORAGE", "", AV_XE2))

FAMILIES = [  # (yaml file, kernel prefix, op, profile prefix, token prefix, variants)
    ("sarc_sdpa_qk_coopmat_sweep", "sarc_sdpa_qk_coopmat_sweep", "kSdpaQk", "qk", "sweep", QK_SWEEP),
    ("sarc_sdpa_qk_coopmat_pk", "sarc_sdpa_qk_coopmat_pk", "kSdpaQk", "qkpk", "pk", QK_PK),
    ("sarc_sdpa_av_coopmat_sweep", "sarc_sdpa_av_coopmat_sweep", "kSdpaAv", "av", "sweep", AV_SWEEP),
    ("sarc_sdpa_av_coopmat_ml", "sarc_sdpa_av_coopmat_ml", "kSdpaAv", "avml", "ml", AV_ML),
    (None, "sarc_sdpa_qk_coopmat_xe2", "kSdpaQk", "qkfr", "xe2", QK_XE2),
    (None, "sarc_sdpa_av_coopmat_xe2", "kSdpaAv", "avfr", "xe2", AV_XE2),
    (None, "sarc_sdpa_qk_coopmat_xe2c", "kSdpaQk", "qkc", "xe2c", QK_XE2C),
]
YB = "# xe2 begin: Intel Xe2 variants (openspec/changes/sarc-1.5-xe2-prefill-refine, tools/gen_xe2.py)\n"
YE = "# xe2 end\n"
for fname, prefix, op, pp, tp, variants in FAMILIES:
    if fname is None:
        continue
    y = ""
    for t, m, n, k, sx, sy, extra in variants:
        y += (f"    - NAME: {prefix}_{t}\n      MMA_M: {MMA_M}\n      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n"
              f"      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {SG}\n{extra}")
    p = g / f"{fname}.yaml"; p.write_text(block(p.read_text(), YB, YE, y))

# The profiles. Base rows = the table choice on Xe2 while an xe2-* profile is requested.
BASE_QK = QK_SWEEP[0]; BASE_AV = AV_SWEEP[0]
REFINE = {
    # name: [(op, token with family prefix, shape predicate or nullptr)]
    "xe2-sdpa0": [],   # the base rows alone
    # candidate 1 (SDPA prefill): screen 1, best kernel per shape. QK^T with packed staging for every
    # head_dim; attn*V 128-row fragment-layout tile for head_dim 128 (3B, 8B), the base 64 x 64 tile for 64 (1B).
    "xe2-refine1": [("kSdpaQk", "pk_t128x64k32g44s16m8nf", "nullptr"), ("kSdpaAv", "xe2_t128x64k32g44s16m8", "xe2_head_dim_128")],
}
PREDS = """// attn*V: ShapeInfo::N is head_dim.
bool xe2_head_dim_128(const ShapeInfo& s) {
  return s.N >= 128;
}
"""
rows = ""
def row(dev, pred, op, name, m, n, k, sx, sy):
    return (f'    {{"{dev}", {pred}, Op::{op},\n     "{name}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {SG}, {MMA_M}, false}}, kBufBuf, nullptr,\n'
            f"     Status::kUnverified}},\n")
base = ""
for dev in ("bmg g21", "bmg g31"):
    t, m, n, k, sx, sy, _ = BASE_QK; base += row(dev, "xe2_profile_requested", "kSdpaQk", f"sarc_sdpa_qk_coopmat_sweep_{t}", m, n, k, sx, sy)
    t, m, n, k, sx, sy, _ = BASE_AV; base += row(dev, "xe2_profile_requested", "kSdpaAv", f"sarc_sdpa_av_coopmat_sweep_{t}", m, n, k, sx, sy)
cand = ""; prefs = ""; profs = ""
for fname, prefix, op, pp, tp, variants in FAMILIES:
    for t, m, n, k, sx, sy, _ in variants:
        cand += row("", "nullptr", op, f"{prefix}_{t}", m, n, k, sx, sy)
        ident = f"kXe2_{pp}_{t}"
        prefs += f'const Preference {ident}[] = {{{{Op::{op}, "{tp}_{t}", nullptr}}}};\n'
        profs += f'    {{"xe2-{pp}-{t}", {ident}, 1}},\n'
for name, items in REFINE.items():
    ident = "kXe2_" + name[4:].replace("-", "_")
    if items:
        prefs += f"const Preference {ident}[] = {{\n" + "".join(f'    {{Op::{op}, "{t}", {pred}}},\n' for op, t, pred in items) + "};\n"
        profs += f'    {{"{name}", {ident}, sizeof({ident}) / sizeof(Preference)}},\n'
    else:
        profs += f'    {{"{name}", nullptr, 0}},\n'

(impl / "Xe2Sdpa.cpp").write_text(f"""/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC development zone, Intel Xe2 (device tag xe2): SDPA prefill rows for the
// Arc B580 / Arc Pro B70 (openspec/changes/sarc-1.5-xe2-prefill-refine;
// generated by its tools/gen_xe2.py). Not part of a release.
//
// The release tables have no Intel SDPA row, and impl/sarc/SdpaCoopmat.cpp
// builds the SARC SDPA path (spec constants, truncated softmax) only on a
// device with an active SDPA row. The base rows below are that row, from the
// dev zone: kUnverified (so they need ET_VK_SARC_UNVERIFIED=1) and matching
// only while ET_VK_SARC_DEV_PROFILE names an xe2-* profile, so every other
// configuration of a dev build selects exactly what the release tables select.
// Kernels: the 780M sweep twins built for the shapes Xe2 exposes (MMA 8x16x16
// fp16 x fp16 -> fp32, subgroup 16); WG_TILE_K of the base rows is 32, which
// the spec constants assume.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <cstring>

namespace vkcompute {{
namespace sarc {{
namespace {{

bool xe2_profile_requested(const DeviceInfo&) {{
  const char* e = std::getenv("ET_VK_SARC_DEV_PROFILE");
  return e != nullptr && std::strncmp(e, "xe2-", 4) == 0;
}}

const Row kXe2SdpaRows[] = {{
{base}}};

// Selected only through an xe2-* profile (impl/sarc_dev/Overrides.cpp).
const Row kXe2SdpaCandidates[] = {{
{cand}}};

struct Registrar {{
  Registrar() {{
    register_rows(kXe2SdpaRows, sizeof(kXe2SdpaRows) / sizeof(kXe2SdpaRows[0]));
    register_candidates(
        kXe2SdpaCandidates,
        sizeof(kXe2SdpaCandidates) / sizeof(kXe2SdpaCandidates[0]));
  }}
}} registrar;

}} // namespace
}} // namespace sarc
}} // namespace vkcompute
""")

# ---------------- linear: phase-timing (PROF) twins of the shipped Xe2 tiles, MEASUREMENT ONLY ----------------
# (kernel base, op, dims, storages, yaml file, per-storage yaml lines)
Q4_FLAGS = "      MMA_M: 8\n      WG_TILE_K: 16\n      SG_GRID_X: 4\n      SG_GRID_Y: 4\n      SUBGROUP_SIZE: 16\n      FRAG_LAYOUT: true\n      IMG_A: true\n"
DQ_FLAGS = "      WG_TILE_M: 256\n      SG_GRID_X: 4\n      SG_GRID_Y: 8\n      SUBGROUP_SIZE: 16\n      MMA_M: 8\n      MMA_K: 32\n      A_MAP_FULL: true\n      A_MULTI_BLOCK: true\n"
LIN = [  # (kernel base, op, "{dims}", storages, yaml, variant flags)
    ("sarc_dev_prof_q4gsw_t128x128k16g44s16m8flip", "kQ4gswLinear", "{128, 128, 16, 4, 4, 16, 8, false}", "kTex3dTex2d | kBufTex2d", "sarc_dev_prof_q4gsw", Q4_FLAGS),
    ("sarc_dev_prof_dq8ca_zpg_t256x64k32g48s16m8p", "kDq8caLinear", "{256, 64, 32, 4, 8, 16, 8, false}", "kTex3dTex2d | kBufTex2d", "sarc_dev_prof_dq8ca_zpg", DQ_FLAGS),
]
lin_rows = ""; ybody = {}
for kb, op, dims, st, yf, flags in LIN:
    lin_rows += f'    {{"", nullptr, Op::{op},\n     "{kb}", {dims},\n     {st}, nullptr, Status::kUnverified}},\n'
    ybody[yf] = ybody.get(yf, "") + f"    - NAME: {kb}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n{flags}    - NAME: {kb}_buffer_texture2d_half\n{flags}"

# ---------------- linear sweep tiles (texture3d = the model path), statically pruned ----------------
# 8da4w zpg, MMA 8x16x32, subgroup 16. (M, N, K, sgx, sgy). A map full: A blocks per chunk (M/4)*(K/4) is a
# whole number of blocks per thread; B slots per thread (K/4)*N / WG_SIZE is an integer; shared memory =
# 2 slices of A (M*32 bytes per K slab) and B (N*32 per slab) + izp/ifs + wsc/wcorr + the drain band.
def dq_lds(m, n, k, sy): return 2 * (k // 32) * (m * 32 + n * 32) + m * 8 + n * 12 + sy * MMA_M * n * 2
DQ = [(128, 64, 32, 4, 4), (128, 64, 32, 2, 4), (128, 64, 32, 4, 2), (256, 64, 32, 4, 4), (256, 64, 32, 2, 8),
      (128, 128, 32, 4, 4), (256, 128, 32, 4, 8), (128, 64, 64, 4, 4), (256, 64, 32, 4, 8)]
dq_tiles = []
for m, n, k, sx, sy in DQ:
    wg = sx * sy * SG; ab, rem = divmod((m // 4) * (k // 4), wg)
    assert geometry_ok(m, n, sx, sy) and rem == 0 and ab >= 1 and ((k // 4) * n) % wg == 0 and k % 32 == 0, (m, n, k, sx, sy)
    assert dq_lds(m, n, k, sy) <= LDS_MAX, (m, n, k, dq_lds(m, n, k, sy))
    dq_tiles.append((m, n, k, sx, sy, ab))
def dq_yaml(m, n, k, sx, sy, ab):
    return (f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {SG}\n"
            f"      MMA_M: {MMA_M}\n      MMA_K: 32\n      A_MAP_FULL: true\n      A_MULTI_BLOCK: true\n      A_BLOCKS: {ab}\n")
for m, n, k, sx, sy, ab in dq_tiles:
    if (m, n, k, sx, sy) == (256, 64, 32, 4, 8): continue        # the shipped tile; only its bt twin is new (never: 512 threads)
    kb = f"sarc_linear_dq8ca_coopmat_zpg_sweep_{tok(m, n, k, sx, sy)}"
    lin_rows += f'    {{"", nullptr, Op::kDq8caLinear,\n     "{kb}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {SG}, {MMA_M}, false}}, kTex3dTex2d, nullptr, Status::kUnverified}},\n'
    ybody["sarc_linear_dq8ca_coopmat_zpg_sweep"] = ybody.get("sarc_linear_dq8ca_coopmat_zpg_sweep", "") + f"    - NAME: {kb}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n" + dq_yaml(m, n, k, sx, sy, ab)
    # texel-wise weight staging (the 780M bt family): a (texel, parity) slot per thread, 2*(K/4)*(N/8) slots
    if (2 * (k // 4) * (n // 8)) % (sx * sy * SG) == 0:
        kb = f"sarc_dev_linear_dq8ca_coopmat_zpg_bt_{tok(m, n, k, sx, sy)}"
        lin_rows += f'    {{"", nullptr, Op::kDq8caLinear,\n     "{kb}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {SG}, {MMA_M}, false}}, kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified}},\n'
        ybody["sarc_dev_linear_dq8ca_coopmat_zpg_bt"] = ybody.get("sarc_dev_linear_dq8ca_coopmat_zpg_bt", "") + "".join(
            f"    - NAME: {kb}_{io}_texture2d_half\n      IO_STORAGE: {io}\n" + dq_yaml(m, n, k, sx, sy, ab) for io in ("texture3d", "buffer"))
# 4w, FRAG_LAYOUT + IMG_A (the shipped Xe2 flags), MMA 8x16x16, subgroup 16. Staging passes must be integral:
# A rows per pass = WG_SIZE / (K/8) divides M, B rows per pass = WG_SIZE / (N/8) divides K. Shared memory =
# 2 slices of A (M*K*2 bytes) and B (K*N*2) + the drain band (SG_GRID_Y * 8 rows x N fp16).
Q4 = [(128, 128, 32, 4, 4), (128, 256, 16, 4, 4), (64, 128, 16, 4, 2), (128, 64, 16, 2, 4), (128, 128, 16, 2, 4),
      (256, 128, 32, 4, 8), (128, 128, 32, 2, 4)]
for m, n, k, sx, sy in Q4:
    wg = sx * sy * SG
    if 2 * 2 * (m * k + k * n) + sy * MMA_M * n * 2 > LDS_MAX: continue          # pruned: over the shared-memory limit
    assert geometry_ok(m, n, sx, sy) and wg % (k // 8) == 0 and m % (wg // (k // 8)) == 0 and wg % (n // 8) == 0 and k % (wg // (n // 8)) == 0, (m, n, k, sx, sy)
    kb = f"sarc_linear_q4gsw_coopmat_sweep_{tok(m, n, k, sx, sy)}fli"
    lin_rows += f'    {{"", nullptr, Op::kQ4gswLinear,\n     "{kb}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {SG}, {MMA_M}, false}}, kTex3dTex2d, nullptr, Status::kUnverified}},\n'
    ybody["sarc_linear_q4gsw_coopmat_sweep"] = ybody.get("sarc_linear_q4gsw_coopmat_sweep", "") + (
        f"    - NAME: {kb}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n      WEIGHT_STORAGE: texture2d\n      MMA_M: {MMA_M}\n"
        f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {SG}\n      FRAG_LAYOUT: true\n      IMG_A: true\n")
for yf, y in ybody.items():
    p = g / f"{yf}.yaml"; p.write_text(block(p.read_text(), YB, YE, y))
(impl / "Xe2Linear.cpp").write_text(f"""/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC development zone, Intel Xe2 (device tag xe2): linear candidate rows of
// openspec/changes/sarc-1.5-xe2-prefill-refine (generated by its tools/gen_xe2.py).
// Selected only by name: ET_VK_SARC_Q4GSW_VARIANT / ET_VK_SARC_DQ8CA_VARIANT or
// an xe2-* profile. The sarc_dev_prof_* rows are MEASUREMENT ONLY: they write
// shader-clock phase counters over their output.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {{
namespace sarc {{
namespace {{

const Row kXe2LinearCandidates[] = {{
{lin_rows}}};

struct Registrar {{
  Registrar() {{
    register_candidates(
        kXe2LinearCandidates,
        sizeof(kXe2LinearCandidates) / sizeof(kXe2LinearCandidates[0]));
  }}
}} registrar;

}} // namespace
}} // namespace sarc
}} // namespace vkcompute
""")

o = impl / "Overrides.cpp"; t = o.read_text()
CB = "// xe2 begin: Intel Xe2 profiles (openspec/changes/sarc-1.5-xe2-prefill-refine, tools/gen_xe2.py)\n"
CE = "// xe2 end\n"
t = block(t, CB, CE, "// Single-kernel screening profiles and the xe2-refineN candidates. They take effect on a device whose\n"
          "// SDPA base rows are active (impl/sarc_dev/Xe2Sdpa.cpp, ET_VK_SARC_UNVERIFIED=1).\n" + PREDS + prefs, "struct Profile {\n")
PB = "    // xe2 begin: Intel Xe2 profiles (tools/gen_xe2.py)\n"; PE = "    // xe2 end\n"
t = block(t, PB, PE, profs, "};\nconst Profile* requested_profile() {")
o.write_text(t)

c = root / "test/sarc_dev/test_llama_microbench.cpp"; t = c.read_text()
TB = "      // xe2 begin: Xe2 tile tokens end in s16m8nf (openspec/changes/sarc-1.5-xe2-prefill-refine)\n"; TE = "      // xe2 end\n"
old = '      qk_name.find("s64nf") != std::string::npos;\n'
if TB not in t:
    assert t.count(old) == 1
    t = t.replace(old, '      qk_name.find("s64nf") != std::string::npos\n' + TB + '      || qk_name.find("m8nf") != std::string::npos\n' + TE + "      ;\n")
    c.write_text(t)
print(f"xe2: {len(QK_SWEEP)} qk sweep, {len(QK_PK)} qk pk, {len(QK_XE2)} qk xe2, {len(AV_SWEEP)} av sweep, {len(AV_ML)} av ml, {len(AV_XE2)} av xe2 variants; {len(REFINE)} refine profiles")
