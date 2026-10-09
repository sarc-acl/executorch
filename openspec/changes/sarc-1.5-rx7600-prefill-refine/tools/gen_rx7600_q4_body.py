#!/usr/bin/env python3
# gen_rx7600_q4_body.py <release 4w body> <out body>: release 4w body (glsl/sarc/sarc_linear_q4gsw_coopmat_body.glslh) ->
# glsl/sarc_dev/sarc_dev_rx7600_q4_body.glslh (round 2, candidate 2). Every edit asserts that it matched exactly once.
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:70])
    t = t.replace(old, new)

rep("// Body of the SARC q4gsw coopmat linear. Included by a template header", """// SARC development zone, RX 7600 campaign round 2 (openspec/changes/sarc-1.5-rx7600-prefill-refine): a copy of
// glsl/sarc/sarc_linear_q4gsw_coopmat_body.glslh (the release body is untouched) with the options below, all default off (a variant
// with none of them computes exactly what the release body computes):
//   BT_PAD_H (constant from the wrapper; default 8): padding in fp16 of one row (one N column) of the column-major B staging
//     (B_COLMAJOR), so that the row pitch is WG_TILE_K + BT_PAD_H fp16 (80 bytes for the default; 72 or 88 spread the 16 lanes of a
//     ds_read_b64 fragment load over the LDS banks).
//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY (wrong results by design). Added after candidate 2's session (2026-10-09); texture3d banded drain only.
// Body of the SARC q4gsw coopmat linear. Included by a template header""")
rep("SARC_LINEAR_Q4GSW_COOPMAT_BODY_GLSLH", "SARC_DEV_RX7600_Q4GSW_BODY_GLSLH", 3)
rep("const uint BT_STRIDE_H = WG_TILE_K + 8u;", "const uint BT_STRIDE_H = WG_TILE_K + BT_PAD_H;")

# ---- second pass: RX_A_V2, the A staging array typed uvec2 so that its row pitch can be 8-byte granular (WG_TILE_K + A_PAD_H fp16) ----
rep("//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY (wrong results by design). Added after candidate 2's session (2026-10-09); texture3d banded drain only.", """//   RX_A_V2 (with A_PAD_H, constant from the wrapper): the A staging array is uvec2 (not uvec4) with a row pitch of WG_TILE_K + A_PAD_H fp16;
//     A_PAD_H = 4 or 12 gives 72 or 88 bytes. Needs B_COLMAJOR and CSH_IN_ASH and none of SH_F16V4, CSH_POOL, CSH_FULL, CSH_BAND, FRAG_LAYOUT.
//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY (wrong results by design). Added after candidate 2's session (2026-10-09); texture3d banded drain only.""")
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


# ---- third pass (round 2, added 2026-10-09 after candidate 2's session): RX_PROF phase timing, MEASUREMENT ONLY. Texture3d output, banded drain
# (CSH_IN_ASH, the path of the picked variants) only; the counters overwrite the first two texels of each workgroup's output tile. ----
rep("""void main() {
  const uvec2 tileID""", """#ifdef RX_PROF
#define PTICK(acc) { const uint p_n = clock2x32ARB().x; acc += (p_n - p_t) & 0xFFFFFu; p_t = p_n; }
#else
#define PTICK(acc)
#endif

void main() {
#ifdef RX_PROF
#if !defined(IO_TEXTURE) || defined(CSH_FULL) || defined(CSH_BAND) || defined(CSH_POOL)
#error "RX_PROF needs the texture3d banded drain"
#endif
  uint p_pro = 0u, p_bar = 0u, p_pre = 0u, p_mma = 0u, p_st = 0u, p_ep = 0u, p_dr = 0u, p_wr = 0u;
  uint p_t = clock2x32ARB().x;
#endif
  const uvec2 tileID""")
rep("  uint chunk;\n  for (chunk = 0; chunk + 1u < num_chunks; ++chunk) {", "  PTICK(p_pro)\n  uint chunk;\n  for (chunk = 0; chunk + 1u < num_chunks; ++chunk) {")
rep("    memoryBarrierShared();\n    barrier();\n\n    // --- prefetch chunk+1 -> temp ---", "    memoryBarrierShared();\n    barrier();\n    PTICK(p_bar)\n\n    // --- prefetch chunk+1 -> temp ---")
rep("    // --- MMA math on the cur slice ---", "    PTICK(p_pre)\n    // --- MMA math on the cur slice ---")
rep("    // --- store temp (chunk+1) -> nxt slice, dequantizing B ---", "    PTICK(p_mma)\n    // --- store temp (chunk+1) -> nxt slice, dequantizing B ---")
rep("    }\n  }\n\n  // --- epilogue: barrier, then MMA on the last chunk (loop peeled) ---", "    }\n    PTICK(p_st)\n  }\n\n  // --- epilogue: barrier, then MMA on the last chunk (loop peeled) ---")
rep("  const uint CSH_TEXELS_PER_ROW = WG_TILE_N / 4u;", "  PTICK(p_ep)\n  const uint CSH_TEXELS_PER_ROW = WG_TILE_N / 4u;")
rep("    memoryBarrierShared();\n    barrier();\n\n    for (uint t = gl_LocalInvocationID.x; t < CSH_TEXELS; t += WG_SIZE) {\n      const uint lr = t / CSH_TEXELS_PER_ROW;\n      const uint lc4 = t % CSH_TEXELS_PER_ROW;\n      const uint m =",
    "    memoryBarrierShared();\n    barrier();\n    PTICK(p_dr)\n\n    for (uint t = gl_LocalInvocationID.x; t < CSH_TEXELS; t += WG_SIZE) {\n      const uint lr = t / CSH_TEXELS_PER_ROW;\n      const uint lc4 = t % CSH_TEXELS_PER_ROW;\n      const uint m =")
rep("#endif\n    }\n  }\n#endif // CSH_FULL / CSH_BAND", "#endif\n    }\n    PTICK(p_wr)\n  }\n#endif // CSH_FULL / CSH_BAND")
rep("#endif // IO_TEXTURE\n}", """#endif // IO_TEXTURE
#ifdef RX_PROF
  // PROF output: subgroup 0 writes the phase means over the first two texels of its tile (same layout as the 8da4w twins).
  PTICK(p_wr)
  {
    const float it = float(max(num_chunks, 1u));
    const vec4 v0 = vec4(float(p_bar), float(p_pre), float(p_mma), float(p_st)) / it;
    const vec4 v1 = vec4(float(p_pro), float(p_ep), float(p_dr), float(p_wr)) / 64.0;
    memoryBarrierImage();
    barrier();
    if (gl_SubgroupID == 0u && subgroupElect()) {
      imageStore(t_output, ivec3(tile_n_start / 4u, tile_m_start, 0), v0);
      imageStore(t_output, ivec3(tile_n_start / 4u + 1u, tile_m_start, 0), v1);
    }
  }
#endif
}""")

open(dst, "w").write(t)
print("ok", len(t.splitlines()), "lines")
