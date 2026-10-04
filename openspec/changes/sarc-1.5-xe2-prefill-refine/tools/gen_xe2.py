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

Static pruning: a variant is emitted only if its shared memory fits the device limit (49152 bytes on both cards)
with margin, its workgroup has at most 1024 invocations, the subgroup tile is a whole number of 8x16 MMA tiles
and (single-pass attn*V) WG_SIZE == 4 * WG_TILE_M == WG_TILE_K * WG_TILE_N / 8.
"""
import pathlib, re, sys
root = pathlib.Path(sys.argv[1]) / "backends/vulkan"
g = root / "runtime/graph/ops/glsl/sarc_dev"
impl = root / "runtime/graph/ops/impl/sarc_dev"
LDS_MAX = 46000      # bytes; maxComputeSharedMemorySize is 49152 on BMG G21 and G31
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
(g / "sarc_sdpa_av_coopmat_xe2.yaml").write_text(family_yaml("sarc_sdpa_av_coopmat_xe2", "V_CACHE_STORAGE", "", AV_XE2))

FAMILIES = [  # (yaml file, kernel prefix, op, profile prefix, token prefix, variants)
    ("sarc_sdpa_qk_coopmat_sweep", "sarc_sdpa_qk_coopmat_sweep", "kSdpaQk", "qk", "sweep", QK_SWEEP),
    ("sarc_sdpa_qk_coopmat_pk", "sarc_sdpa_qk_coopmat_pk", "kSdpaQk", "qkpk", "pk", QK_PK),
    ("sarc_sdpa_av_coopmat_sweep", "sarc_sdpa_av_coopmat_sweep", "kSdpaAv", "av", "sweep", AV_SWEEP),
    ("sarc_sdpa_av_coopmat_ml", "sarc_sdpa_av_coopmat_ml", "kSdpaAv", "avml", "ml", AV_ML),
    (None, "sarc_sdpa_qk_coopmat_xe2", "kSdpaQk", "qkfr", "xe2", QK_XE2),
    (None, "sarc_sdpa_av_coopmat_xe2", "kSdpaAv", "avfr", "xe2", AV_XE2),
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
    # name: [(op, token with family prefix)]
    "xe2-sdpa0": [],   # the base rows alone
}
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
        prefs += f"const Preference {ident}[] = {{\n" + "".join(f'    {{Op::{op}, "{t}", nullptr}},\n' for op, t in items) + "};\n"
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

o = impl / "Overrides.cpp"; t = o.read_text()
CB = "// xe2 begin: Intel Xe2 profiles (openspec/changes/sarc-1.5-xe2-prefill-refine, tools/gen_xe2.py)\n"
CE = "// xe2 end\n"
t = block(t, CB, CE, "// Single-kernel screening profiles and the xe2-refineN candidates. They take effect on a device whose\n"
          "// SDPA base rows are active (impl/sarc_dev/Xe2Sdpa.cpp, ET_VK_SARC_UNVERIFIED=1).\n" + prefs, "struct Profile {\n")
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
