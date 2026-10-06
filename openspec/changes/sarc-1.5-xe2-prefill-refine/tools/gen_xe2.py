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
# --- truncated softmax that reads the row once (NOT reachable from the dev zone: the softmax shader name is
# fixed in impl/sarc/SdpaCoopmat.cpp; measured only through the local hook in tools/hook-sdpa-softmax.patch) ---
sm = (g.parent / "sarc/sarc_sdpa_attn_weights_softmax.glsl").read_text(); sm = sm[sm.index("#version 450 core"):]
LOAD = """    SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(
        c4, s, q_h, context_texel_len, attn_S, Q_H);
"""
sm = sub(sm, "shared SOFTMAX_ACC_T shared_exp_sum[NUM_WORKERS_PER_WG];\n", """shared SOFTMAX_ACC_T shared_exp_sum[NUM_WORKERS_PER_WG];

// xe2: a worker keeps its texels of the row between the passes (at most XS_KEEP;
// 16 covers a context of 4096), so the row is read once and exp is evaluated
// once per element. Longer rows take the release path. Same reductions in the
// same order and the same exp(x - max) / sum per element as the release shader.
#define XS_KEEP 16
""")
sm = sub(sm, """  for (int c4 = worker_id; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG) {
""" + LOAD + """
    for (int comp = 0; comp < 4; comp++) {
      local_max = max(local_max, SOFTMAX_ACC_T(in_texel[comp]));
    }
  }
""", """  SOFTMAX_IN_VEC4_T kept[XS_KEEP];
  const bool keep = R4_limit <= NUM_WORKERS_PER_WG * XS_KEEP;
  for (int c4 = worker_id, i = 0; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG, ++i) {
""" + LOAD + """    if (keep) {
      kept[i] = in_texel;
    }

    for (int comp = 0; comp < 4; comp++) {
      local_max = max(local_max, SOFTMAX_ACC_T(in_texel[comp]));
    }
  }
""")
sm = sub(sm, """  for (int c4 = worker_id; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG) {
""" + LOAD + """
    for (int comp = 0; comp < 4; comp++) {
      local_exp_sum += exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max);
    }
  }
""", """  for (int c4 = worker_id, i = 0; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG, ++i) {
    SOFTMAX_IN_VEC4_T in_texel;
    if (keep) {
      in_texel = kept[i];
    } else {
      in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, attn_S, Q_H);
    }

    SOFTMAX_IN_VEC4_T e;
    for (int comp = 0; comp < 4; comp++) {
      e[comp] = exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max);
      local_exp_sum += SOFTMAX_ACC_T(e[comp]);
    }
    if (keep) {
      kept[i] = e;
    }
  }
""")
sm = sub(sm, """  for (int c4 = worker_id; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG) {
""" + LOAD + """
    VEC4_T out_texel;
    [[unroll]] for (int comp = 0; comp < 4; comp++) {
      out_texel[comp] = T(
          exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max) / local_exp_sum);
    }
""", """  for (int c4 = worker_id, i = 0; c4 < R4_limit; c4 += NUM_WORKERS_PER_WG, ++i) {
    VEC4_T out_texel;
    if (keep) {
      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        out_texel[comp] = T(SOFTMAX_ACC_T(kept[i][comp]) / local_exp_sum);
      }
    } else {
      SOFTMAX_IN_VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, attn_S, Q_H);
      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        out_texel[comp] = T(
            exp(SOFTMAX_ACC_T(in_texel[comp]) - global_max) / local_exp_sum);
      }
    }
""")
(g / "sarc_dev_sdpa_attn_weights_softmax_xe2.glsl").write_text("""/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, Intel Xe2 (openspec/changes/sarc-1.5-xe2-prefill-refine, generated by its
 * tools/gen_xe2.py from glsl/sarc/sarc_sdpa_attn_weights_softmax.glsl): the truncated LLM softmax reading each
 * row once. MEASUREMENT ONLY: no dev-zone path selects it (the softmax name is fixed in
 * impl/sarc/SdpaCoopmat.cpp); see tools/hook-sdpa-softmax.patch for the hook it would need.
 */

""" + sm)
(g / "sarc_dev_sdpa_attn_weights_softmax_xe2.yaml").write_text("""# SARC development zone, Intel Xe2: single-read truncated softmax (generated by sarc-1.5-xe2-prefill-refine/tools/gen_xe2.py).
# Not shipped and not selected by any dev-zone path; LLM mode, fp16 buffers only.

sarc_dev_sdpa_attn_weights_softmax_xe2:
  parameter_names_with_default_values:
    IN_DTYPE: half
    OUT_DTYPE: half
    STORAGE: buffer
    MODE: llm
  generate_variant_forall:
    STORAGE:
      - VALUE: buffer
    combination:
      parameter_names: [IN_DTYPE, OUT_DTYPE]
      combos:
        - parameter_values: [half, half]
          suffix: half
  shader_variants:
    - NAME: sarc_dev_sdpa_attn_weights_softmax_xe2
""")

# --- second hook-only softmax: the two reductions with subgroup operations (2 barriers per row, not 14) ---
# Screen 10: the single-read variant above is not faster (0.79 against 0.80 ms per layer on 8B), so the reads
# and the second exp are not what the kernel spends its time on; each row is one 64-thread workgroup that
# passes 14 barriers. Here every subgroup reduces its lanes with subgroupMax / subgroupAdd and the workgroup
# combines the few subgroup values after one barrier. The max is exact; the sum of exp is added in a different
# order and its subgroup partial sums are carried in fp32, so the result is NOT bit-identical to the release
# softmax: an arithmetic change, to be judged against the fp32 reference.
sg = (g.parent / "sarc/sarc_sdpa_attn_weights_softmax.glsl").read_text(); sg = sg[sg.index("#version 450 core"):]
sg = sub(sg, "#extension GL_EXT_control_flow_attributes : require\n", "#extension GL_EXT_control_flow_attributes : require\n#extension GL_KHR_shader_subgroup_basic : require\n#extension GL_KHR_shader_subgroup_arithmetic : require\n")
sg = sub(sg, "shared SOFTMAX_ACC_T shared_exp_sum[NUM_WORKERS_PER_WG];\n", "shared SOFTMAX_ACC_T shared_exp_sum[NUM_WORKERS_PER_WG];\n// xe2sg: one fp32 value per subgroup for each reduction (separate arrays: no barrier between the two uses).\nshared float xs_sg[NUM_WORKERS_PER_WG];\nshared float xs_sg2[NUM_WORKERS_PER_WG];\n")
sg = sub(sg, """  shared_max[worker_id] = local_max;

  memoryBarrierShared();
  barrier();

  // Tree reduction to find the global max
  for (int i = NUM_WORKERS_PER_WG / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_max[worker_id] = max(
          shared_max[worker_id], shared_max[worker_id + i]);
    }
    memoryBarrierShared();
    barrier();
  }

  const SOFTMAX_ACC_T global_max = shared_max[0];
""", """  // xe2sg: max over the subgroup's lanes, then over the subgroups (exact).
  {
    const float sg_max = subgroupMax(float(local_max));
    if (subgroupElect()) {
      xs_sg[gl_SubgroupID] = sg_max;
    }
  }
  memoryBarrierShared();
  barrier();
  float xs_max = xs_sg[0];
  for (uint i = 1u; i < gl_NumSubgroups; ++i) {
    xs_max = max(xs_max, xs_sg[i]);
  }
  const SOFTMAX_ACC_T global_max = SOFTMAX_ACC_T(xs_max);
""")
sg = sub(sg, """  shared_exp_sum[worker_id] = local_exp_sum;

  memoryBarrierShared();
  barrier();

  // Tree reduction to compute the overall exp sum
  for (int i = NUM_WORKERS_PER_WG / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_exp_sum[worker_id] = shared_exp_sum[worker_id] +
          shared_exp_sum[worker_id + i];
    }
    memoryBarrierShared();
    barrier();
  }

  local_exp_sum = shared_exp_sum[0];
""", """  // xe2sg: sum over the subgroup's lanes (fp32), then over the subgroups in index order.
  {
    const float sg_sum = subgroupAdd(float(local_exp_sum));
    if (subgroupElect()) {
      xs_sg2[gl_SubgroupID] = sg_sum;
    }
  }
  memoryBarrierShared();
  barrier();
  float xs_sum = xs_sg2[0];
  for (uint i = 1u; i < gl_NumSubgroups; ++i) {
    xs_sum += xs_sg2[i];
  }
  local_exp_sum = SOFTMAX_ACC_T(xs_sum);
""")
(g / "sarc_dev_sdpa_attn_weights_softmax_xe2sg.glsl").write_text("""/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, Intel Xe2 (openspec/changes/sarc-1.5-xe2-prefill-refine, generated by its
 * tools/gen_xe2.py from glsl/sarc/sarc_sdpa_attn_weights_softmax.glsl): the truncated LLM softmax with
 * subgroup reductions. MEASUREMENT ONLY and an arithmetic change (the exp sum is added in another order): no
 * dev-zone path selects it; see tools/hook-sdpa-softmax.patch for the hook it would need.
 */

""" + sg)
(g / "sarc_dev_sdpa_attn_weights_softmax_xe2sg.yaml").write_text((g / "sarc_dev_sdpa_attn_weights_softmax_xe2.yaml").read_text()
    .replace("single-read truncated softmax", "truncated softmax with subgroup reductions").replace("sarc_dev_sdpa_attn_weights_softmax_xe2", "sarc_dev_sdpa_attn_weights_softmax_xe2sg"))

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

# Attention tiles confirmed by the enumeration of the legal spaces (results/xe2/sweep/{av,qk}/confirm.csv), used
# by candidate 6. Unlike everything above they may use subgroup size 32, so the size is part of the entry:
# (yaml file or None for a generated Xe2 family, kernel prefix, op, token prefix, M, N, K, sgx, sgy, subgroup size, nf)
FOUND = [
    (None, "sarc_sdpa_av_coopmat_xe2", "kSdpaAv", "xe2", 64, 64, 64, 4, 4, 32, False),     # search id 400163: head_dim 64
    (None, "sarc_sdpa_av_coopmat_xe2", "kSdpaAv", "xe2", 128, 64, 64, 4, 4, 32, False),    # search id 400819: head_dim 128
    (None, "sarc_sdpa_qk_coopmat_xe2c", "kSdpaQk", "xe2c", 64, 128, 32, 8, 2, 16, True),   # search id 303340
]
def found_tok(m, n, k, sx, sy, sg, nf): return f"t{m}x{n}k{k}g{sx}{sy}s{sg}m8" + ("nf" if nf else "")
def found_yaml(prefix, op, m, n, k, sx, sy, sg, nf):
    return (f"    - NAME: {prefix}_{found_tok(m, n, k, sx, sy, sg, nf)}\n      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n"
            f"      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {sg}\n      MMA_M: {MMA_M}\n"
            + (f"      NO_MASK_FILL: {'true' if nf else 'false'}\n" if op == "kSdpaQk" else ""))
for fname, prefix, op, tp, m, n, k, sx, sy, sg, nf in FOUND:
    assert sx * sy * sg <= 1024 and m % sy == 0 and n % sx == 0 and (m // sy) % MMA_M == 0 and (n // sx) % 16 == 0, (prefix, m, n, sx, sy, sg)
    if fname is None:
        p = g / f"{prefix}.yaml"; p.write_text(p.read_text() + found_yaml(prefix, op, m, n, k, sx, sy, sg, nf))
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
    y += "".join(found_yaml(f[1], f[2], *f[4:]) for f in FOUND if f[0] == fname)
    p = g / f"{fname}.yaml"; p.write_text(block(p.read_text(), YB, YE, y))

# The profiles. Base rows = the table choice on Xe2 while an xe2-* profile is requested.
BASE_QK = QK_SWEEP[0]; BASE_AV = AV_SWEEP[0]
REFINE = {
    # name: [(op, token with family prefix, shape predicate or nullptr)]
    "xe2-sdpa0": [],   # the base rows alone
    # candidate 1 (SDPA prefill): screen 1, best kernel per shape. QK^T with packed staging for every
    # head_dim; attn*V 128-row fragment-layout tile for head_dim 128 (3B, 8B), the base 64 x 64 tile for 64 (1B).
    "xe2-refine1": [("kSdpaQk", "pk_t128x64k32g44s16m8nf", "nullptr"), ("kSdpaAv", "xe2_t128x64k32g44s16m8", "xe2_head_dim_128")],
    # candidate 2 (8da4w linear): refine1 + the balanced K = 64 tile with texel-wise weight staging (screen 7:
    # 1.15 to 1.39x per shape at kernel level; every thread stages one A block and one weight texel per chunk).
    "xe2-refine2": [("kSdpaQk", "pk_t128x64k32g44s16m8nf", "nullptr"), ("kSdpaAv", "xe2_t128x64k32g44s16m8", "xe2_head_dim_128"),
                    ("kDq8caLinear", "xe2bt_t128x128k64g84s16m8", "nullptr")],
    # candidate 3 (SDPA QK^T): refine2 with the fragment-contiguous ColumnMajor QK^T (screen 2: 0.38 against
    # 0.41 ms per layer on 8B, 0.29 against 0.31 on 3B, equal on 1B). A layout change, meant to be bit-identical.
    # (No 4w candidate exists: every 4w variant of screens 4, 6, 8, 9 and 11 is slower than the shipped tile.)
    "xe2-refine3": [("kSdpaQk", "xe2c_t128x64k32g44s16m8nf", "nullptr"), ("kSdpaAv", "xe2_t128x64k32g44s16m8", "xe2_head_dim_128"),
                    ("kDq8caLinear", "xe2bt_t128x128k64g84s16m8", "nullptr")],
    # candidate 4 (8da4w, per shape): refine3 with the 64-column K = 64 tile for the one shape class where it
    # measured faster than the 128-column tile (screen 7: N >= 4 K, i.e. 1B w1/w3, 622 against 647 us).
    "xe2-refine4": [("kSdpaQk", "xe2c_t128x64k32g44s16m8nf", "nullptr"), ("kSdpaAv", "xe2_t128x64k32g44s16m8", "xe2_head_dim_128"),
                    ("kDq8caLinear", "bt_t128x64k64g44s16m8", "xe2_wide_output"), ("kDq8caLinear", "xe2bt_t128x128k64g84s16m8", "nullptr")],
    # candidate 5 (4w linear, from the sampled parameter search; results/xe2/sweep/4w/confirm-c.csv): refine2 with
    # the shipped 128 x 128 K = 16 tile on a subgroup grid of 8 x 2 (subgroup tile 16 x 64 instead of 32 x 32)
    # and the band drain (search id 150312: 1.02 to 1.06x of the shipped kernel on every shape of the three
    # models except 1B wk / wv, 0.98x), and for an output of at most 512 columns the K = 32 tile on a grid of
    # 8 x 4 with IMG_W (search id 170243: 1.18x on 1B wk / wv, the only such shape; 0.71 and 0.89x on the
    # 1024-column wk / wv of 3B and 8B, which therefore keep the first tile).
    "xe2-refine5": [("kSdpaQk", "pk_t128x64k32g44s16m8nf", "nullptr"), ("kSdpaAv", "xe2_t128x64k32g44s16m8", "xe2_head_dim_128"),
                    ("kDq8caLinear", "xe2bt_t128x128k64g84s16m8", "nullptr"),
                    ("kQ4gswLinear", "sweep_t128x128k32g84s16m8flw", "xe2_narrow_output"),
                    ("kQ4gswLinear", "sweep_t128x128k16g82s16m8flib", "nullptr")],
    # candidate 6 (attention, from the enumeration of the legal spaces; results/xe2/sweep/{qk,av}/confirm.csv):
    # refine5 with QK^T on the fragment-contiguous ColumnMajor 64 x 128 tile, grid 8 x 2 (search id 303340: 1.03 /
    # 1.12 / 1.12x of the refine1 kernel on 1B / 3B / 8B) and attn*V on K = 64 tiles with subgroup size 32:
    # 128 x 64 for head_dim 128 (400819: 1.20 / 1.22x on 3B / 8B), 64 x 64 for head_dim 64 (400163: 1.13x on 1B).
    "xe2-refine6": [("kSdpaQk", "xe2c_t64x128k32g82s16m8nf", "nullptr"), ("kSdpaAv", "xe2_t128x64k64g44s32m8", "xe2_head_dim_128"),
                    ("kSdpaAv", "xe2_t64x64k64g44s32m8", "nullptr"),
                    ("kDq8caLinear", "xe2bt_t128x128k64g84s16m8", "nullptr"),
                    ("kQ4gswLinear", "sweep_t128x128k32g84s16m8flw", "xe2_narrow_output"),
                    ("kQ4gswLinear", "sweep_t128x128k16g82s16m8flib", "nullptr")],
    # the 4w part of refine5 alone (kernel attribution; not a candidate)
    "xe2-q4-g82": [("kQ4gswLinear", "sweep_t128x128k32g84s16m8flw", "xe2_narrow_output"), ("kQ4gswLinear", "sweep_t128x128k16g82s16m8flib", "nullptr")],
    # the 8da4w part of refine2 alone (kernel attribution; not a candidate)
    "xe2-dq-k64": [("kDq8caLinear", "xe2bt_t128x128k64g84s16m8", "nullptr")],
}
PREDS = """// attn*V: ShapeInfo::N is head_dim.
bool xe2_head_dim_128(const ShapeInfo& s) {
  return s.N >= 128;
}
// 8da4w linear: an output at least four times as wide as the input (1B w1 / w3).
bool xe2_wide_output(const ShapeInfo& s) {
  return s.N >= 4 * s.K;
}
// 4w linear: an output of at most 512 columns (1B wk / wv).
bool xe2_narrow_output(const ShapeInfo& s) {
  return s.N <= 512;
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
for fname, prefix, op, tp, m, n, k, sx, sy, sg, nf in FOUND:
    cand += (f'    {{"", nullptr, Op::{op},\n     "{prefix}_{found_tok(m, n, k, sx, sy, sg, nf)}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {sg}, {MMA_M}, false}}, kBufBuf, nullptr,\n'
             f"     Status::kUnverified}},\n")
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
    # phase timing of the candidate-2 kernel (balanced K = 64 tile, texel-wise weight staging)
    ("sarc_dev_prof_dq8ca_coopmat_zpg_bt_t128x128k64g84s16m8p", "kDq8caLinear", "{128, 128, 64, 8, 4, 16, 8, false}", "kTex3dTex2d | kBufTex2d", "sarc_dev_prof_dq8ca_coopmat_zpg_bt",
     "      WG_TILE_M: 128\n      WG_TILE_N: 128\n      WG_TILE_K: 64\n      SG_GRID_X: 8\n      SG_GRID_Y: 4\n      SUBGROUP_SIZE: 16\n      MMA_M: 8\n      MMA_K: 32\n      A_MAP_FULL: true\n      A_MULTI_BLOCK: true\n"),
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
# ---- 8da4w, batch 2 (after screen 3): texel-wise weight staging on tiles whose slot count is below the
# workgroup size (family sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt). Screen 3: every tile with more than 4 x 1 MMA
# tiles per subgroup lost (0.26 to 0.52x) and doubling the weight fetches per thread cost 20 %; the bt staging
# made the K = 64 tile 1.65x faster than its zpg twin. The 780M bt body needs 2 * (K/4) * (N/8) slots to be a
# multiple of WG_SIZE, which excludes the shipped 512-thread tile (128 slots); here a thread whose slot index
# is past the last slot stages nothing. Same LDS layout, values and MMA loop as the release kernel.
xb = (g / "sarc_dev_linear_dq8ca_coopmat_zpg_bt_body.glslh").read_text()
xb = xb.replace("SARC_DEV_LINEAR_DQ8CA_COOPMAT_ZPG_BT_BODY_GLSLH", "SARC_DEV_LINEAR_DQ8CA_COOPMAT_ZPG_XE2BT_BODY_GLSLH")
xb = sub(xb, "  const uint B_SLOTS_PER_THREAD = B_TOTAL_SLOTS / WG_SIZE;\n",
         "  // xe2bt: round up; B_SLOT_ACTIVE masks the slots past the last one.\n"
         "  const uint B_SLOTS_PER_THREAD = (B_TOTAL_SLOTS + WG_SIZE - 1u) / WG_SIZE;\n"
         "#define B_SLOT_ACTIVE(si) (B_TOTAL_SLOTS % WG_SIZE == 0u || gl_LocalInvocationID.x + (si) * WG_SIZE < B_TOTAL_SLOTS)\n")
loop = "[[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {"
assert xb.count(loop) == 5
head, rest = xb.split(loop, 1)          # the first loop only computes indices
xb = head + loop + rest.replace(loop, "[[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) if (B_SLOT_ACTIVE(si)) {")
(g / "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt_body.glslh").write_text(xb)
xw = (g / "sarc_dev_linear_dq8ca_coopmat_zpg_bt.glsl").read_text(); xw = xw[xw.index("#version 450 core"):]
xw = sub(xw, '#include "sarc_dev_linear_dq8ca_coopmat_zpg_bt_body.glslh"', '#include "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt_body.glslh"')
(g / "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt.glsl").write_text(HDR % ("sarc_dev_linear_dq8ca_coopmat_zpg_bt", "sarc_linear_dq8ca_coopmat_zpg",
    "8da4w zpg with texel-wise weight staging where the slot count may be below the workgroup size.") + xw)
ys = (g / "sarc_linear_dq8ca_coopmat_zpg_sweep.yaml").read_text()
ydef = ys[ys.index("  parameter_names_with_default_values:"):ys.index("  shader_variants:")]
XB = [(256, 64, 32, 4, 8), (128, 64, 64, 4, 4), (128, 64, 64, 4, 8), (128, 64, 32, 4, 8), (256, 128, 32, 8, 8),
      (128, 128, 32, 8, 8), (128, 64, 32, 4, 4), (256, 64, 32, 4, 16),
      # batch 3 (after screen 5): staging on a subset of the threads lost on every tile (0.56 to 0.95x), the
      # balanced K = 64 tile won (1.12x). These keep every thread staging whole A blocks and whole texel slots
      # and put 64 or 128 K into one chunk (one barrier and one fetch round per chunk).
      (128, 128, 64, 8, 4), (64, 128, 64, 8, 2), (64, 64, 128, 4, 2), (64, 64, 64, 4, 2)]
xy = "# SARC development zone, Intel Xe2: 8da4w zpg, texel-wise weight staging (generated by sarc-1.5-xe2-prefill-refine/tools/gen_xe2.py). Not shipped.\n\nsarc_dev_linear_dq8ca_coopmat_zpg_xe2bt:\n" + ydef + "  shader_variants:\n"
for m, n, k, sx, sy in XB:
    wg = sx * sy * SG; blocks = (m // 4) * (k // 4)
    assert geometry_ok(m, n, sx, sy) and k % 32 == 0 and (m // sy) // MMA_M <= 4 and (n // sx) == 16, (m, n, k, sx, sy)   # at most 4 x 1 MMA tiles per subgroup
    assert dq_lds(m, n, k, sy) <= LDS_MAX, (m, n, k, dq_lds(m, n, k, sy))
    full = blocks % wg == 0; ab = max(blocks // wg, 1); assert full or blocks < wg
    kb = f"sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt_{tok(m, n, k, sx, sy)}"
    lin_rows += f'    {{"", nullptr, Op::kDq8caLinear,\n     "{kb}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {SG}, {MMA_M}, false}}, kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified}},\n'
    for io in ("texture3d", "buffer"):
        xy += (f"    - NAME: {kb}_{io}_texture2d_half\n      IO_STORAGE: {io}\n      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n"
               f"      SUBGROUP_SIZE: {SG}\n      MMA_M: {MMA_M}\n      MMA_K: 32\n      A_MAP_FULL: {'true' if full else 'false'}\n      A_MULTI_BLOCK: true\n      A_BLOCKS: {ab}\n")
(g / "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt.yaml").write_text(xy)

# ---- 4w, batch 2 (after screen 4): texel-wise weight staging (family sarc_dev_linear_q4gsw_coopmat_xe2bx).
# Screen 4: every other tile shape lost (0.14 to 0.90x), so the shipped geometry stays. In the release body a
# thread stages one K row x 8 N per pass and fetches the packed-weight texel (4 K rows x 8 N) for it, so every
# texel is fetched 4 times per chunk; fetch is 22 to 26 % of a wave on this device. Here a slot is a texel: the
# thread fetches it once and writes its 4 rows. With the shipped tile 64 of the 256 threads stage B. Same LDS
# layout and the same dequantized values as the release kernel. (The 780M bx body keeps one row per thread
# when B_PASSES = 1, which is the Xe2 case, so it does not reduce the fetches here.)
qb = (g.parent / "sarc/sarc_linear_q4gsw_coopmat_body.glslh").read_text()
qb = qb.replace("SARC_LINEAR_Q4GSW_COOPMAT_BODY_GLSLH", "SARC_DEV_LINEAR_Q4GSW_COOPMAT_XE2BX_BODY_GLSLH")
qb = sub(qb, "const uint B_PASSES = WG_TILE_K / B_ROWS_PER_PASS;\n", """const uint B_PASSES = WG_TILE_K / B_ROWS_PER_PASS;
// xe2bx (generated by sarc-1.5-xe2-prefill-refine/tools/gen_xe2.py from the release body): texel-wise B
// staging. A slot is one packed-weight texel (4 K rows x 8 N); slot gl_LocalInvocationID.x + si * WG_SIZE =
// k4 * INVS_PER_ROW_B + n8, so a thread's slots share its N column (b_col) and step B_ROWS_PER_PASS texel rows.
#ifdef CSH_POOL
#error "xe2bx is not combined with CSH_POOL"
#endif
const uint XB_K4 = WG_TILE_K / 4u;
const uint XB_SPT = (XB_K4 + B_ROWS_PER_PASS - 1u) / B_ROWS_PER_PASS;
#define XB_K4_OF(si) (b_row_offset + (si) * B_ROWS_PER_PASS)
#define XB_ACTIVE(si) (XB_K4_OF(si) < XB_K4)
""")
qb = sub(qb, "#else\n  ivec4 temp_B[B_PASSES];\n#endif\n", "#else\n  ivec4 temp_B[XB_SPT];\n#endif\n")
qb = sub(qb, """    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
      const uint k_row = p * B_ROWS_PER_PASS + b_row_offset;
      ivec4 wblock;
#ifdef WEIGHT_BUFFER
      wblock = t_packed_weight[n8_blk * K4 + (k_row >> 2u)];
#else
      wblock = WFETCH(ivec2(k_row >> 2u, n8_blk));
#endif
      temp_B[p] = wblock;
    }
""", """    [[unroll]] for (uint si = 0; si < XB_SPT; ++si) if (XB_ACTIVE(si)) {
#ifdef WEIGHT_BUFFER
      temp_B[si] = t_packed_weight[n8_blk * K4 + XB_K4_OF(si)];
#else
      temp_B[si] = WFETCH(ivec2(XB_K4_OF(si), n8_blk));
#endif
    }
""")
qb = sub(qb, """    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
#ifdef CSH_POOL
      BSH_STORE(BSH_BASE + B_SH_IDX(p * B_ROWS_PER_PASS + b_row_offset, b_col),
          dequant_block(temp_B[p], col_lo, col_hi, sc0, sc1));
#else
      BSH_STORE(B_SH_IDX(p * B_ROWS_PER_PASS + b_row_offset, b_col),
          dequant_block(temp_B[p], col_lo, col_hi, sc0, sc1));
#endif
    }
""", """    [[unroll]] for (uint si = 0; si < XB_SPT; ++si) if (XB_ACTIVE(si)) {
      [[unroll]] for (uint r = 0; r < 4u; ++r) {
        BSH_STORE(B_SH_IDX(XB_K4_OF(si) * 4u + r, b_col),
            dequant_block(temp_B[si], 2u * r, 2u * r + 1u, sc0, sc1));
      }
    }
""")
qb = sub(qb, """      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
        const uint k_row = chunkK_nxt + p * B_ROWS_PER_PASS + b_row_offset;
#ifdef WEIGHT_BUFFER
        temp_B[p] = t_packed_weight[n8_blk * K4 + (k_row >> 2u)];
#else
        temp_B[p] = WFETCH(ivec2(k_row >> 2u, n8_blk));
#endif
      }
""", """      [[unroll]] for (uint si = 0; si < XB_SPT; ++si) if (XB_ACTIVE(si)) {
#ifdef WEIGHT_BUFFER
        temp_B[si] = t_packed_weight[n8_blk * K4 + chunkK_nxt / 4u + XB_K4_OF(si)];
#else
        temp_B[si] = WFETCH(ivec2(chunkK_nxt / 4u + XB_K4_OF(si), n8_blk));
#endif
      }
""")
qb = sub(qb, """      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
        BSH_STORE(nxt_base_B + B_SH_IDX(p * B_ROWS_PER_PASS + b_row_offset, b_col),
            dequant_block(temp_B[p], col_lo, col_hi, sc0, sc1));
      }
""", """      [[unroll]] for (uint si = 0; si < XB_SPT; ++si) if (XB_ACTIVE(si)) {
        [[unroll]] for (uint r = 0; r < 4u; ++r) {
          BSH_STORE(nxt_base_B + B_SH_IDX(XB_K4_OF(si) * 4u + r, b_col),
              dequant_block(temp_B[si], 2u * r, 2u * r + 1u, sc0, sc1));
        }
      }
""")
assert "col_lo, col_hi, sc0, sc1" not in qb
(g / "sarc_dev_linear_q4gsw_coopmat_xe2bx_body.glslh").write_text(qb)
qw = (g / "sarc_linear_q4gsw_coopmat_sweep.glsl").read_text(); qw = qw[qw.index("#version 450 core"):]
qw = sub(qw, '#include "sarc_linear_q4gsw_coopmat_body.glslh"', '#include "sarc_dev_linear_q4gsw_coopmat_xe2bx_body.glslh"')
(g / "sarc_dev_linear_q4gsw_coopmat_xe2bx.glsl").write_text(HDR % ("sarc_linear_q4gsw_coopmat_sweep", "sarc_linear_q4gsw_coopmat",
    "4w with texel-wise weight staging: one fetch per packed-weight texel per chunk.") + qw)
ys = (g / "sarc_linear_q4gsw_coopmat_sweep.yaml").read_text()
ydef = ys[ys.index("  parameter_names_with_default_values:"):ys.index("  shader_variants:")]
qy = "# SARC development zone, Intel Xe2: 4w with texel-wise weight staging (generated by sarc-1.5-xe2-prefill-refine/tools/gen_xe2.py). Not shipped.\n\nsarc_dev_linear_q4gsw_coopmat_xe2bx:\n" + ydef + "  shader_variants:\n"
# (M, N, K, sgx, sgy, IMG_W)
for m, n, k, sx, sy, imgw, band in [(128, 128, 16, 4, 4, False, False), (128, 128, 16, 4, 4, True, False), (128, 128, 32, 4, 4, False, False),
                                    (256, 128, 16, 4, 8, False, False), (128, 128, 16, 4, 4, False, True), (256, 128, 16, 4, 8, False, True)]:
    wg = sx * sy * SG
    assert geometry_ok(m, n, sx, sy) and wg % (k // 8) == 0 and m % (wg // (k // 8)) == 0 and wg % (n // 8) == 0, (m, n, k, sx, sy)   # A passes integral, a thread owns one N column
    assert 2 * 2 * (m * k + k * n) + sy * MMA_M * n * 2 <= LDS_MAX, (m, n, k)
    kb = f"sarc_dev_linear_q4gsw_coopmat_xe2bx_{tok(m, n, k, sx, sy)}fli{'w' if imgw else ''}{'b' if band else ''}"
    lin_rows += f'    {{"", nullptr, Op::kQ4gswLinear,\n     "{kb}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {SG}, {MMA_M}, false}}, kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified}},\n'
    for io in ("texture3d", "buffer"):
        qy += (f"    - NAME: {kb}_{io}_texture2d_half\n      IO_STORAGE: {io}\n      WEIGHT_STORAGE: texture2d\n      MMA_M: {MMA_M}\n      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n"
               f"      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {SG}\n      FRAG_LAYOUT: true\n      IMG_A: {'true' if io == 'texture3d' else 'false'}\n" + ("      IMG_W: true\n" if imgw else "") + ("      CSH_BAND: true\n" if band else ""))
(g / "sarc_dev_linear_q4gsw_coopmat_xe2bx.yaml").write_text(qy)

# ---- 4w, batch 3: split staging roles (family sarc_dev_linear_q4gsw_coopmat_xe2s). What the screens showed:
# the shipped tile already gives every thread one A unit and one B unit per 8 MMA tiles, more accumulators per
# subgroup or more serial staging per thread loses, and the shared-memory footprint does not matter. The only
# way left to stage less per MMA is more reuse: a 256 x 256 tile uses every staged A and B element twice as
# often. With 1024 threads (64 subgroups of the shipped 32 x 32 subgroup tile) it has 512 A units and 512 B
# units per chunk, i.e. ONE unit per thread instead of two; the release body cannot express that (its pass
# counts become 0). Here pass counts round up, the first threads stage A and the last ones B. Needs CSH_BAND (the
# banded drain of the full tile would not fit). Same LDS layout, values and MMA order as the release kernel.
sb = (g.parent / "sarc/sarc_linear_q4gsw_coopmat_body.glslh").read_text()
sb = sb.replace("SARC_LINEAR_Q4GSW_COOPMAT_BODY_GLSLH", "SARC_DEV_LINEAR_Q4GSW_COOPMAT_XE2S_BODY_GLSLH")
sb = sub(sb, "const uint A_PASSES = WG_TILE_M / A_ROWS_PER_PASS;\n", """// xe2s (generated by sarc-1.5-xe2-prefill-refine/tools/gen_xe2.py from the release body): the pass counts
// round up; a tile with fewer A (B) units than threads lets the first (last) threads stage A (B), in ascending
// order within a subgroup as in the release body. A tile that is full on a side keeps the release code for it
// (the conditions fold to true). Screen 9 measured the first version of this body, which mapped B in reverse
// thread order and tested every row: 15 % slower than the release kernel on the shipped geometry.
const uint A_PASSES = (WG_TILE_M + A_ROWS_PER_PASS - 1u) / A_ROWS_PER_PASS;
const bool XS_A_FULL = A_ROWS_PER_PASS <= WG_TILE_M;
const bool XS_B_FULL = WG_SIZE <= WG_TILE_K * (WG_TILE_N / FP16_PER_VEC4);
const uint XS_B_SHIFT = XS_B_FULL ? 0u : WG_SIZE - WG_TILE_K * (WG_TILE_N / FP16_PER_VEC4);
#define XS_A_ON(p) (XS_A_FULL || (p) * A_ROWS_PER_PASS + a_row_offset < WG_TILE_M)
#define XS_B_ON(p) (XS_B_FULL || gl_LocalInvocationID.x >= XS_B_SHIFT)
#ifdef B_COLMAJOR
#error "xe2s is not combined with B_COLMAJOR"
#endif
""")
sb = sub(sb, "const uint B_PASSES = WG_TILE_K / B_ROWS_PER_PASS;\n", "const uint B_PASSES = (WG_TILE_K + B_ROWS_PER_PASS - 1u) / B_ROWS_PER_PASS;\n")
sb = sub(sb, """  const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
  const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;
""", """  const uint xs_b_tid = gl_LocalInvocationID.x - XS_B_SHIFT;
  const uint b_col = xs_b_tid % INVS_PER_ROW_B;
  const uint b_row_offset = xs_b_tid / INVS_PER_ROW_B;
""")
la = "[[unroll]] for (uint p = 0; p < A_PASSES; ++p) {"; lb = "[[unroll]] for (uint p = 0; p < B_PASSES; ++p) {"
assert sb.count(la) == 4 and sb.count(lb) == 4, (sb.count(la), sb.count(lb))
sb = sb.replace(la, "[[unroll]] for (uint p = 0; p < A_PASSES; ++p) if (XS_A_ON(p)) {").replace(lb, "[[unroll]] for (uint p = 0; p < B_PASSES; ++p) if (XS_B_ON(p)) {")
(g / "sarc_dev_linear_q4gsw_coopmat_xe2s_body.glslh").write_text(sb)
sw = (g / "sarc_linear_q4gsw_coopmat_sweep.glsl").read_text(); sw = sw[sw.index("#version 450 core"):]
sw = sub(sw, '#include "sarc_linear_q4gsw_coopmat_body.glslh"', '#include "sarc_dev_linear_q4gsw_coopmat_xe2s_body.glslh"')
(g / "sarc_dev_linear_q4gsw_coopmat_xe2s.glsl").write_text(HDR % ("sarc_linear_q4gsw_coopmat_sweep", "sarc_linear_q4gsw_coopmat",
    "4w with split staging roles for tiles that have fewer staging units than twice the workgroup size.") + sw)
ys = (g / "sarc_linear_q4gsw_coopmat_sweep.yaml").read_text()
ydef = ys[ys.index("  parameter_names_with_default_values:"):ys.index("  shader_variants:")]
sy_ = "# SARC development zone, Intel Xe2: 4w with split staging roles (generated by sarc-1.5-xe2-prefill-refine/tools/gen_xe2.py). Not shipped.\n\nsarc_dev_linear_q4gsw_coopmat_xe2s:\n" + ydef + "  shader_variants:\n"
# (M, N, K, sgx, sgy); all with the band drain. Subgroup tile 32 x 32 (4 x 2 MMA tiles, as shipped).
for m, n, k, sx, sy in [(256, 256, 16, 8, 8), (256, 128, 16, 4, 8), (128, 256, 16, 8, 4), (128, 128, 16, 4, 4)]:
    wg = sx * sy * SG
    assert geometry_ok(m, n, sx, sy) and (m // sy, n // sx) == (32, 32) and wg % (k // 8) == 0 and wg % (n // 8) == 0, (m, n, k, sx, sy)
    assert (wg // (n // 8)) % 4 == 0                                        # B rows per pass keep the nibble phase
    assert 2 * 2 * (m * k + k * n) + MMA_M * n * 2 <= LDS_MAX, (m, n, k)     # A + B double-buffered + one drain band
    kb = f"sarc_dev_linear_q4gsw_coopmat_xe2s_{tok(m, n, k, sx, sy)}flib"
    lin_rows += f'    {{"", nullptr, Op::kQ4gswLinear,\n     "{kb}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {SG}, {MMA_M}, false}}, kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified}},\n'
    for io in ("texture3d", "buffer"):
        sy_ += (f"    - NAME: {kb}_{io}_texture2d_half\n      IO_STORAGE: {io}\n      WEIGHT_STORAGE: texture2d\n      MMA_M: {MMA_M}\n      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n"
                f"      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {SG}\n      FRAG_LAYOUT: true\n      IMG_A: {'true' if io == 'texture3d' else 'false'}\n      CSH_BAND: true\n")
(g / "sarc_dev_linear_q4gsw_coopmat_xe2s.yaml").write_text(sy_)

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
# The shipped tile with the drain staged one band at a time (CSH_BAND): 18.4 KiB of shared memory instead of
# 24.6. The buffer-IO twin of the shipped tile (no drain staging, 16.4 KiB) is 13 to 17 % faster per kernel than
# the texture3d one although drain + write are 1 % of a wave; this separates shared-memory footprint from the
# activation fetch as the cause.
kb = f"sarc_linear_q4gsw_coopmat_sweep_{tok(128, 128, 16, 4, 4)}flib"
lin_rows += f'    {{"", nullptr, Op::kQ4gswLinear,\n     "{kb}",\n     {{128, 128, 16, 4, 4, {SG}, {MMA_M}, false}}, kTex3dTex2d, nullptr, Status::kUnverified}},\n'
ybody["sarc_linear_q4gsw_coopmat_sweep"] += (
    f"    - NAME: {kb}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n      WEIGHT_STORAGE: texture2d\n      MMA_M: {MMA_M}\n"
    f"      WG_TILE_K: 16\n      SG_GRID_X: 4\n      SG_GRID_Y: 4\n      SUBGROUP_SIZE: {SG}\n      FRAG_LAYOUT: true\n      IMG_A: true\n      CSH_BAND: true\n")
# The two 4w tiles the sampled parameter search confirmed (candidate 5, profile xe2-refine5): the release body
# with exactly the flags of search ids 150312 and 170243, so the SPIR-V is that of the measured sweep variants.
Q4F = ["FRAG_LAYOUT", "B_COLMAJOR", "SH_F16V4", "IMG_A", "IMG_W", "CSH_BAND", "CSH_FULL", "CSH_POOL", "CSH_IN_ASH", "ACC_FP32", "ACC_GROUP_FP32"]
for m, n, k, sx, sy, suffix, on in [(128, 128, 16, 8, 2, "flib", {"FRAG_LAYOUT", "IMG_A", "CSH_BAND"}), (128, 128, 32, 8, 4, "flw", {"FRAG_LAYOUT", "IMG_W"})]:
    wg = sx * sy * SG
    assert geometry_ok(m, n, sx, sy) and wg % (k // 8) == 0 and m % (wg // (k // 8)) == 0 and wg % (n // 8) == 0 and k % (wg // (n // 8)) == 0, (m, n, k, sx, sy)
    kb = f"sarc_linear_q4gsw_coopmat_sweep_{tok(m, n, k, sx, sy)}{suffix}"
    lin_rows += f'    {{"", nullptr, Op::kQ4gswLinear,\n     "{kb}",\n     {{{m}, {n}, {k}, {sx}, {sy}, {SG}, {MMA_M}, false}}, kTex3dTex2d, nullptr, Status::kUnverified}},\n'
    ybody["sarc_linear_q4gsw_coopmat_sweep"] += (
        f"    - NAME: {kb}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n      WEIGHT_STORAGE: texture2d\n"
        f"      WG_TILE_M: {m}\n      WG_TILE_N: {n}\n      WG_TILE_K: {k}\n      SG_GRID_X: {sx}\n      SG_GRID_Y: {sy}\n      SUBGROUP_SIZE: {SG}\n      MMA_M: {MMA_M}\n"
        + "".join(f"      {f}: {'true' if f in on else 'false'}\n" for f in Q4F))
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
# SDPA correctness: report the error against the fp32 CPU reference per case (owner decision 2026-10-04, second:
# a candidate that changes kernel arithmetic is judged by its rms and maximum error against the reference
# beside the parent's). Printing only; the pass criterion of the test is unchanged.
EB = "  // xe2 begin: error against the fp32 CPU reference (openspec/changes/sarc-1.5-xe2-prefill-refine)\n"; EE = "  // xe2 end\n"
t = c.read_text()
if EB not in t:
    anchor = "  const bool numeric_ok = mismatches == 0;\n  const bool fired_ok = qk_fired && av_fired && pairing_ok;\n"
    assert t.count(anchor) == 1
    t = t.replace(anchor, EB + """  {
    double se = 0.0, sr = 0.0, emax = 0.0;
    for (int64_t i = 0; i < q_numel; ++i) {
      const double d = static_cast<double>(outf[i]) - static_cast<double>(ref[i]);
      se += d * d;
      sr += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
      emax = std::max(emax, std::fabs(d));
    }
    std::cout << "[sdpa-error] " << c.name << " elements=" << q_numel
              << std::scientific << std::setprecision(4)
              << " rms_err=" << std::sqrt(se / static_cast<double>(q_numel))
              << " max_abs_err=" << emax
              << " ref_rms=" << std::sqrt(sr / static_cast<double>(q_numel))
              << std::defaultfloat << std::setprecision(6) << "\\n";
  }
""" + EE + anchor)
    c.write_text(t)
print(f"xe2: {len(QK_SWEEP)} qk sweep, {len(QK_PK)} qk pk, {len(QK_XE2)} qk xe2, {len(AV_SWEEP)} av sweep, {len(AV_ML)} av ml, {len(AV_XE2)} av xe2 variants; {len(REFINE)} refine profiles")
