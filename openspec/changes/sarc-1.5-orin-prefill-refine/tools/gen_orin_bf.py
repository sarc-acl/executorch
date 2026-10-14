#!/usr/bin/env python3
"""gen_orin_bf.py: 8da4w zpgtr with whole-texel weight staging for the Jetson Orin (dev-zone twin of the release
body; the release files are read, never written).

  glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpgtr_orin_bf{.glsl,.yaml,_body.glslh}
  impl/sarc_dev/Overrides.cpp   `// >>> orin bf-dq8ca-rows`

Why. The shipped kernel (t128x128k64g44s32mk32ra, B_PAIR) stages the weights with 2 texelFetch per thread and
chunk and keeps ONE 32-bit word of each fetched texel (4 words, 32 nibbles): every packed-weight texel is
fetched four times per chunk, by four threads. On this device a texture2d read is the slowest path there is
(fresh roofs: 20 GB/s from DRAM and 45 GB/s from cache for texture2d, against 62 GB/s for a buffer), and int8
has twice the fp16 matrix rate, so the weight fetch, not the MMA, bounds the kernel.
Here a staging slot is a whole texel: one fetch, eight shared-memory words (4 components x 2 nibble parities).
The values, the shared-memory layout the MMA reads and the MMA order are those of the release kernel, so the
output is meant to be bit-identical; the gate checks it (production-diff, and bit comparison of the dumps).

Two forms:
  bf   two ping-pong slices as shipped. With 512 threads and K = 64 only every second thread has a texel.
  bf1  ONE_SLICE: a single staging slice and a second barrier per chunk (between the MMA and the store of the
       next chunk), which halves the shared memory and so allows K = 128 per chunk on a 128 x 128 tile: one
       texel per thread and chunk for 512 threads, and half as many chunks.
Token prefixes `orin_bf_`, `orin_bf1_`; MMA 16x16x32, A_RAW and B_PAIR as shipped; texture3d IO only (the Orin
rows are texture3d only)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from devzone import write_new, put_block
ET = pathlib.Path(__file__).resolve().parents[4]
ops = ET / "backends/vulkan/runtime/graph/ops"; root = ops / "glsl"
LDS_MAX, WG_MAX, SG = 49152, 1024, 32
def sub(text, old, new, count=1):
    n = text.count(old); assert n == count, f"anchor occurs {n} times, expected {count}: {old[:70]!r}"
    return text.replace(old, new)
F = "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf"
b = (root / "sarc/sarc_linear_dq8ca_coopmat_zpgtr_body.glslh").read_text()
G = "SARC_LINEAR_DQ8CA_COOPMAT_ZPGTR_BODY_GLSLH"; b = sub(b, G, "SARC_LINEAR_DQ8CA_COOPMAT_ZPGTR_ORIN_BF_BODY_GLSLH", b.count(G))
# staging slices: 2 (ping-pong) or 1 (ONE_SLICE)
b = sub(b, "// Double-buffered MMA operand staging.\n", "// orin bf: MMA operand staging in two ping-pong slices, or in one (ONE_SLICE).\n#ifdef ONE_SLICE\nconst uint SLICES = 1u;\n#else\nconst uint SLICES = 2u;\n#endif\n")
b = sub(b, "shared uvec4 Ash_int8[2u * ASH_SLICE_U32 / 4u];", "shared uvec4 Ash_int8[SLICES * ASH_SLICE_U32 / 4u];")
b = sub(b, "shared uint Ash_int8[2u * ASH_SLICE_U32];", "shared uint Ash_int8[SLICES * ASH_SLICE_U32];")
b = sub(b, "shared uint Bsh_int8[2u * BSH_SLICE_U32];", "shared uint Bsh_int8[SLICES * BSH_SLICE_U32];")
b = sub(b, "const int kCshFitsInAsh[(CSH_ROWS * WG_TILE_N / 2u <= 2u * ASH_SLICE_U32) ? 1 : -1] = int[](0);",
        "const int kCshFitsInAsh[(CSH_ROWS * WG_TILE_N / 2u <= SLICES * ASH_SLICE_U32) ? 1 : -1] = int[](0);")
# one slot = one whole texel (k4, n8) producing eight shared-memory words
b = sub(b, "  const uint B_SLOTS_PER_THREAD = B_TOTAL_SLOTS / (2u * WG_SIZE);",
        "  // orin bf: a slot is a whole texel (k4, n8) producing eight LDS uints. Threads beyond the texel count\n"
        "  // of a chunk have no slot (b_on).\n"
        "  const uint B_TEXELS = B_TOTAL_SLOTS / 8u;\n"
        "  const uint B_SLOTS_PER_THREAD = (B_TEXELS + WG_SIZE - 1u) / WG_SIZE;")
b = sub(b, "  uint b_k4off[B_SLOTS_PER_THREAD];    // k4 offset of this slot within a chunk\n#ifdef B_PAIR\n",
        "  uint b_k4off[B_SLOTS_PER_THREAD];    // k4 offset of this slot within a chunk\n#ifdef B_PAIR\n  bool b_on[B_SLOTS_PER_THREAD];       // orin bf: this thread owns a texel in this slot\n")
b = sub(b, """    const uint comp        = (q / K4_PER_SLAB) & 3u;
    const uint slab_idx    = (q / (K4_PER_SLAB * 4u)) % NUM_K_SLABS;
    const uint n8_in_tile  = q / (K4_PER_SLAB * 4u * NUM_K_SLABS);
    const uint k4_in_chunk = slab_idx * K4_PER_SLAB + k4_in_slab;
    const uint n_col       = n8_in_tile * 8u + comp;  // parity 1 is n_col + 4
    b_lds_off[si] = slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab;
    b_comp[si]    = comp;
""", """    // orin bf: lanes run over k4 within a slab fastest, then slab, then the texel column block.
    const uint slab_idx    = (q / K4_PER_SLAB) % NUM_K_SLABS;
    const uint n8_in_tile  = (q / (K4_PER_SLAB * NUM_K_SLABS)) % N8_PER_TILE;
    const uint k4_in_chunk = slab_idx * K4_PER_SLAB + k4_in_slab;
    const uint n_col       = n8_in_tile * 8u;  // component c is column n_col + c, parity 1 is + 4
    b_lds_off[si] = slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab;
    b_comp[si]    = 0u;
    b_on[si]      = q < B_TEXELS;
""")
# N8_PER_TILE is declared after the slot count in the release body, before the map: fine.
FETCH0 = "    temp_B[si] = texelFetch(t_packed_weight, ivec2(b_k4off[si], b_n8blk[si]), 0);\n"
b = sub(b, FETCH0, "    if (b_on[si]) {\n  " + FETCH0 + "    }\n")
FETCH1 = "          temp_B[si] = texelFetch(t_packed_weight, ivec2(k4_blk, b_n8blk[si]), 0);\n"
b = sub(b, FETCH1, "          if (b_on[si]) {\n  " + FETCH1 + "          }\n")
def stores(ind, base):
    s = f"{ind}if (b_on[si]) {{\n"
    for c in range(4):
        s += f"{ind}  Bsh_int8[{base} + {c}u * B_STRIDE_U32] = widen_nibbles(uint(temp_B[si][{c}]), 0u);\n"
        s += f"{ind}  Bsh_int8[{base} + {c + 4}u * B_STRIDE_U32] = widen_nibbles(uint(temp_B[si][{c}]), 1u);\n"
    return s + f"{ind}}}\n"
b = sub(b, """#ifdef B_SEL_EARLY_N
      const uint w = B_WORD(si);
#else
      const uint w = uint(temp_B[si][b_comp[si]]);
#endif
      Bsh_int8[b_lds_off[si]] = widen_nibbles(w, 0u);
      Bsh_int8[b_lds_off[si] + 4u * B_STRIDE_U32] = widen_nibbles(w, 1u);
""", stores("      ", "b_lds_off[si]"))
b = sub(b, """#ifdef B_SEL_EARLY_N
          const uint w = B_WORD(si);
#else
          const uint w = uint(temp_B[si][b_comp[si]]);
#endif
          Bsh_int8[nxt_b + b_lds_off[si]] = widen_nibbles(w, 0u);
          Bsh_int8[nxt_b + b_lds_off[si] + 4u * B_STRIDE_U32] = widen_nibbles(w, 1u);
""", stores("          ", "nxt_b + b_lds_off[si]"))
# ONE_SLICE: every chunk uses slice 0; the store of chunk+1 waits for the MMA readers of the same slice.
b = sub(b, """      const uint cur_a = (chunk % 2u) * ASH_SLICE_U32;
      const uint cur_b = (chunk % 2u) * BSH_SLICE_U32;
      const uint nxt_a = ((chunk + 1u) % 2u) * ASH_SLICE_U32;
      const uint nxt_b = ((chunk + 1u) % 2u) * BSH_SLICE_U32;
""", """#ifdef ONE_SLICE
      const uint cur_a = 0u, cur_b = 0u, nxt_a = 0u, nxt_b = 0u;
#else
      const uint cur_a = (chunk % 2u) * ASH_SLICE_U32;
      const uint cur_b = (chunk % 2u) * BSH_SLICE_U32;
      const uint nxt_a = ((chunk + 1u) % 2u) * ASH_SLICE_U32;
      const uint nxt_b = ((chunk + 1u) % 2u) * BSH_SLICE_U32;
#endif
""")
b = sub(b, "      // --- 4. store temp (chunk+1) -> nxt slice ---\n", """#ifdef ONE_SLICE
      // orin bf1: the slice the MMA above read is the one written next.
      memoryBarrierShared();
      barrier();
#endif
      // --- 4. store temp (chunk+1) -> nxt slice ---
""")
b = "// GENERATED by gen_orin_bf.py from sarc_linear_dq8ca_coopmat_zpgtr_body.glslh (whole-texel weight staging;\n// valid with INT4 texture2d weights, A_RAW and B_PAIR, without B_SEL_EARLY_N only).\n#if !defined(B_PAIR) || !defined(A_RAW) || defined(B_SEL_EARLY_N) || defined(WEIGHT_BUFFER) || !defined(WEIGHT_INT4)\n#error orin bf needs INT4 texture2d weights, A_RAW and B_PAIR, without B_SEL_EARLY_N\n#endif\n" + b
write_new(root / f"sarc_dev/{F}_body.glslh", b)
w = (root / "sarc_dev/sarc_linear_dq8ca_coopmat_zpgtr_sweep.glsl").read_text()
w = sub(w[w.index("#version 450 core"):], '#include "sarc_linear_dq8ca_coopmat_zpgtr_body.glslh"', f'#include "{F}_body.glslh"')
w = sub(w, "$if B_PAIR:\n  #define B_PAIR\n", "$if B_PAIR:\n  #define B_PAIR\n// ONE_SLICE (orin bf1): one staging slice, a second barrier per chunk.\n$if ONE_SLICE:\n  #define ONE_SLICE\n")
HDR = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * SARC development zone, Jetson Orin (generated by gen_orin_bf.py;
 * openspec/changes/sarc-1.5-orin-prefill-refine): zpgtr with whole-texel weight staging.
 */

"""
write_new(root / f"sarc_dev/{F}.glsl", HDR + w)
def Z(m, n, k, sx, sy, one):
    wg = sx * sy * SG; sl = 1 if one else 2
    assert wg <= WG_MAX and 2048 % m == 0 and 128 % k == 0 and k % 32 == 0 and m % (16 * sy) == 0 and n % (16 * sx) == 0
    assert (m * k // 16) % wg == 0 and m * k // 16 >= wg, ("A vectors", m, n, k, sx, sy)
    assert sl * (m * k + k * n) + 8 * m + 12 * n <= LDS_MAX, ("shared memory", m, n, k, sx, sy)
    assert sy * 16 * n * 2 <= sl * m * k, ("drain band in Ash", m, n, k, sx, sy)
    return dict(tok=f"orin_bf{'1' if one else ''}_t{m}x{n}k{k}g{sx}{sy}s{SG}mk32ra", m=m, n=n, k=k, sx=sx, sy=sy, one=one,
                texels_per_thread=(k // 4) * (n // 8) / wg)
VS = [Z(128, 128, 64, 4, 4, False), Z(128, 128, 64, 4, 2, False), Z(128, 128, 64, 2, 4, False), Z(256, 128, 32, 4, 4, False),
      Z(128, 128, 128, 4, 4, True), Z(128, 128, 128, 4, 2, True), Z(128, 128, 128, 2, 4, True), Z(256, 128, 64, 4, 4, True), Z(128, 128, 64, 4, 2, True),
      # second batch (after 8da4w screen 2: one whole texel per thread on a 256-thread tile is the gain; a second
      # barrier per chunk costs more than K = 128 gives). Fewer loads per thread and smaller subgroup tiles:
      Z(64, 128, 64, 4, 2, False), Z(64, 128, 64, 2, 2, False), Z(64, 64, 128, 2, 4, False),
      Z(64, 64, 128, 4, 2, False), Z(64, 64, 64, 2, 2, False), Z(128, 128, 64, 2, 2, False)]
ry = (root / "sarc/sarc_linear_dq8ca_coopmat_zpgtr.yaml").read_text()
params = ry[ry.index("  parameter_names_with_default_values:"):ry.index("  shader_variants:")]
y = "# SARC development zone, Jetson Orin: zpgtr with whole-texel weight staging (generated by gen_orin_bf.py). Not shipped.\n\n" + f"{F}:\n" + params + "    ONE_SLICE: false\n  shader_variants:\n"
rows = "    // Jetson Orin zpgtr with whole-texel weight staging (glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpgtr_orin_bf.yaml).\n"
for v in VS:
    name = "sarc_linear_dq8ca_coopmat_zpgtr_" + v["tok"]
    y += (f"    - NAME: {name}_texture3d_texture2d_half\n      IO_STORAGE: texture3d\n      CSH_IN_ASH: true\n      WG_TILE_M: {v['m']}\n      WG_TILE_N: {v['n']}\n"
          f"      WG_TILE_K: {v['k']}\n      SG_GRID_X: {v['sx']}\n      SG_GRID_Y: {v['sy']}\n      SUBGROUP_SIZE: {SG}\n      MMA_K: 32\n      A_RAW: true\n      B_PAIR: true\n"
          + ("      ONE_SLICE: true\n" if v["one"] else ""))
    rows += f'    {{"", nullptr, Op::kDq8caLinear,\n     "{name}", {{{v["m"]}, {v["n"]}, {v["k"]}, {v["sx"]}, {v["sy"]}, {SG}, 16, true}},\n     kTex3dTex2d, nullptr, Status::kUnverified,\n     /*rowmajor_a=*/true}},\n'
write_new(root / f"sarc_dev/{F}.yaml", y)
put_block(ops / "impl/sarc_dev/Overrides.cpp", "    // >>> 4070ti prof-dq8ca-rows\n", "bf-dq8ca-rows", rows, "    ")
for v in VS: print(v["tok"], "texels per thread and chunk:", v["texels_per_thread"])
