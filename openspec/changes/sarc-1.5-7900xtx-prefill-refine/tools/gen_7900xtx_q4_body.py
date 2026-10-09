#!/usr/bin/env python3
# gen_7900xtx_q4_body.py <release 4w body> <out body>: release 4w body (glsl/sarc/sarc_linear_q4gsw_coopmat_body.glslh) ->
# glsl/sarc_dev/sarc_dev_7900xtx_q4_body.glslh (round 2, candidate 2). Every edit asserts that it matched exactly once.
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:70])
    t = t.replace(old, new)

rep("// Body of the SARC q4gsw coopmat linear. Included by a template header", """// SARC development zone, 7900 XTX campaign round 2 (openspec/changes/sarc-1.5-7900xtx-prefill-refine): a copy of
// glsl/sarc/sarc_linear_q4gsw_coopmat_body.glslh (the release body is untouched) with the options below, all default off (a variant
// with none of them computes exactly what the release body computes):
//   BT_PAD_H (constant from the wrapper; default 8): padding in fp16 of one row (one N column) of the column-major B staging
//     (B_COLMAJOR), so that the row pitch is WG_TILE_K + BT_PAD_H fp16 (80 bytes for the default; 72 or 88 spread the 16 lanes of a
//     ds_read_b64 fragment load over the LDS banks).
//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY (wrong results by design).
// Body of the SARC q4gsw coopmat linear. Included by a template header""")
rep("SARC_LINEAR_Q4GSW_COOPMAT_BODY_GLSLH", "SARC_DEV_X7900XTX_Q4GSW_BODY_GLSLH", 3)
rep("const uint BT_STRIDE_H = WG_TILE_K + 8u;", "const uint BT_STRIDE_H = WG_TILE_K + BT_PAD_H;")

# ---- second pass: RX_A_V2, the A staging array typed uvec2 so that its row pitch can be 8-byte granular (WG_TILE_K + A_PAD_H fp16) ----
rep("//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY (wrong results by design).", """//   RX_A_V2 (with A_PAD_H, constant from the wrapper): the A staging array is uvec2 (not uvec4) with a row pitch of WG_TILE_K + A_PAD_H fp16;
//     A_PAD_H = 4 or 12 gives 72 or 88 bytes. Needs B_COLMAJOR and CSH_IN_ASH and none of SH_F16V4, CSH_POOL, CSH_FULL, CSH_BAND, FRAG_LAYOUT.
//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY (wrong results by design).""")
rep("""const uint A_STRIDE_VEC4 = (WG_TILE_K + FP16_PER_VEC4) / FP16_PER_VEC4;
const uint B_STRIDE_VEC4 = (WG_TILE_N + FP16_PER_VEC4) / FP16_PER_VEC4;
""", """#ifdef RX_A_V2
#if !defined(B_COLMAJOR) || !defined(CSH_IN_ASH) || defined(SH_F16V4) || defined(CSH_POOL) || defined(CSH_FULL) || defined(CSH_BAND)
#error "RX_A_V2 needs B_COLMAJOR and CSH_IN_ASH, and excludes SH_F16V4, CSH_POOL, CSH_FULL and CSH_BAND"
#endif
// uvec2 elements: the A row pitch in elements of the staging array (uvec2)
const uint A_STRIDE_VEC4 = (WG_TILE_K + A_PAD_H) / 4u;
#else
const uint A_STRIDE_VEC4 = (WG_TILE_K + FP16_PER_VEC4) / FP16_PER_VEC4;
#endif
const uint B_STRIDE_VEC4 = (WG_TILE_N + FP16_PER_VEC4) / FP16_PER_VEC4;
""")
rep("""#define A_SH_IDX(m, c) (m) * A_STRIDE_VEC4 + (c)
""", """#ifdef RX_A_V2
#define A_SH_IDX(m, c) (m) * A_STRIDE_VEC4 + 2u * (c)
#else
#define A_SH_IDX(m, c) (m) * A_STRIDE_VEC4 + (c)
#endif
""")
rep("""#else
shared uvec4 Ash[2 * ASH_SLICE];
#endif""", """#else
#ifdef RX_A_V2
shared uvec2 Ash[2 * ASH_SLICE];
#else
shared uvec4 Ash[2 * ASH_SLICE];
#endif
#endif""")
rep("""#define ASH_STORE(idx, v) Ash[idx] = v
""", """#ifdef RX_A_V2
#define ASH_STORE(idx, v) { const uint ash_i = (idx); Ash[ash_i] = (v).xy; Ash[ash_i + 1u] = (v).zw; }
#else
#define ASH_STORE(idx, v) Ash[idx] = v
#endif
""")
rep("""#ifdef CSH_IN_ASH
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]), Ash,
          (warpInTile.y * MMA_M * WG_TILE_N +
              MMA_N * (MMAS_PER_SG_N * warpInTile.x + j)) / FP16_PER_VEC4,
          WG_TILE_N / FP16_PER_VEC4,
          gl_CooperativeMatrixLayoutRowMajor);""", """#ifdef CSH_IN_ASH
#ifdef RX_A_V2
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]), Ash,
          (warpInTile.y * MMA_M * WG_TILE_N +
              MMA_N * (MMAS_PER_SG_N * warpInTile.x + j)) / 4u,
          WG_TILE_N / 4u,
          gl_CooperativeMatrixLayoutRowMajor);
#else
      coopMatStore(
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]), Ash,
          (warpInTile.y * MMA_M * WG_TILE_N +
              MMA_N * (MMAS_PER_SG_N * warpInTile.x + j)) / FP16_PER_VEC4,
          WG_TILE_N / FP16_PER_VEC4,
          gl_CooperativeMatrixLayoutRowMajor);
#endif""")
rep("""#ifdef CSH_IN_ASH
      // base is a multiple of 4 halves: the texel is one half of a uvec4.
      const uvec4 q = Ash[base / FP16_PER_VEC4];
      const bool hi = (base % FP16_PER_VEC4) != 0u;
      imageStore(
          t_output,
          ivec3(tile_n_start / 4u + lc4, m, 0),
          vec4(unpackHalf2x16(hi ? q.z : q.x), unpackHalf2x16(hi ? q.w : q.y)));""", """#ifdef CSH_IN_ASH
#ifdef RX_A_V2
      // base is a multiple of 4 halves: the texel is one uvec2.
      const uvec2 q = Ash[base / 4u];
      imageStore(
          t_output,
          ivec3(tile_n_start / 4u + lc4, m, 0),
          vec4(unpackHalf2x16(q.x), unpackHalf2x16(q.y)));
#else
      // base is a multiple of 4 halves: the texel is one half of a uvec4.
      const uvec4 q = Ash[base / FP16_PER_VEC4];
      const bool hi = (base % FP16_PER_VEC4) != 0u;
      imageStore(
          t_output,
          ivec3(tile_n_start / 4u + lc4, m, 0),
          vec4(unpackHalf2x16(hi ? q.z : q.x), unpackHalf2x16(hi ? q.w : q.y)));
#endif""")

# ---- third pass (second set, item 3): RX_ABL, removed work, MEASUREMENT ONLY (wrong results by design): 4 = the shared-memory stores of the next chunk, 16 = the barrier of the chunk loop ----
rep("#define SARC_DEV_X7900XTX_Q4GSW_BODY_GLSLH\n", """#define SARC_DEV_X7900XTX_Q4GSW_BODY_GLSLH

// RX_ABL (MEASUREMENT ONLY, wrong results by design; set by the wrapper from ABL): bit mask of removed work, 4 = the shared-memory stores of the next
// chunk, 16 = the memory barrier and barrier() at the top of the chunk loop. Default 0: nothing removed.
#ifndef RX_ABL
#define RX_ABL 0
#endif
""")
rep("""    memoryBarrierShared();
    barrier();

    // --- prefetch chunk+1 -> temp ---""", """#if (RX_ABL & 16) == 0
    memoryBarrierShared();
    barrier();
#endif

    // --- prefetch chunk+1 -> temp ---""")
rep("""    // --- store temp (chunk+1) -> nxt slice, dequantizing B ---
    {""", """    // --- store temp (chunk+1) -> nxt slice, dequantizing B ---
#if (RX_ABL & 4) == 0
    {""")
rep("""#endif
    }
  }

  // --- epilogue: barrier, then MMA on the last chunk (loop peeled) ---""", """#endif
    }
#endif // RX_ABL & 4
  }

  // --- epilogue: barrier, then MMA on the last chunk (loop peeled) ---""")

open(dst, "w").write(t)
print("ok", len(t.splitlines()), "lines")
