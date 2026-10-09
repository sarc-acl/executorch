# gen_rx7600_dq_body.py <release body> <out body>: release zpg body (glsl/sarc/sarc_linear_dq8ca_coopmat_zpg_body.glslh) ->
# glsl/sarc_dev/sarc_dev_rx7600_dq8ca_zpg_body.glslh (round 2, candidate 1). Every edit asserts that it matched exactly once, so a change
# of the release body stops the generator instead of silently diverging.
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:70])
    t = t.replace(old, new)

rep("""// Body of sarc_linear_dq8ca_coopmat_zpg; included by its template header (glsl/sarc/sarc_linear_dq8ca_coopmat_zpg.glsl
// or the sweep twin), which defines bindings, spec constants and tile
// constants. Must not contain template syntax.
""", """// SARC development zone, RX 7600 campaign round 2 (openspec/changes/sarc-1.5-rx7600-prefill-refine). Body of
// sarc_dev_rx7600_dq8ca_zpg: a copy of glsl/sarc/sarc_linear_dq8ca_coopmat_zpg_body.glslh (the release body is untouched) with
// the options below, all default off (a variant with none of them computes exactly what the release body computes):
//   A_PAD_U32 (constant from the wrapper): padding in uint between the K slabs of the A staging buffer in shared memory, to
//     spread the A staging stores over the LDS banks.
//   RX_CSH_IN_ASH: the texture drain stages its fp16 tile in the A staging buffer (no separate Csh_out).
//   RX_BF: branch-free chunk loop: the prefetch and the stores of the next chunk run on every iteration (the last one re-reads the
//     last chunk and stores it into the unused buffer), so that no branch separates the MMAs from the stores.
//   RX_ST_A / RX_ST_B (need RX_BF): issue the shared-memory stores of the next chunk's A / B staging right after K slab RX_ST_A /
//     RX_ST_B of the current chunk's MMAs instead of after the last slab, so that the stores of one wave overlap the MMAs of others.
//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY (wrong results by design: the counters overwrite the output tile).
//   RX_ABL: bit mask of removed work, MEASUREMENT ONLY (wrong results by design): 1 A global fetch, 2 B global fetch,
//     4 LDS stores of the next chunk, 8 MMA, 16 the barrier of the chunk loop.
// Included by sarc_dev_rx7600_dq8ca_zpg.glsl, which defines bindings, spec constants and tile constants.
// Must not contain template syntax.
""")
rep("SARC_LINEAR_DQ8CA_COOPMAT_ZPG_BODY_GLSLH", "SARC_DEV_RX7600_DQ8CA_ZPG_BODY_GLSLH", 3)

# A staging pitch
rep("""const uint A_SLAB_U32      = A_SLAB_INT8 / 4u;
const uint A_STRIDE_U32    = MMA_K / 4u;
""", """const uint A_SLAB_U32      = A_SLAB_INT8 / 4u;
const uint A_STRIDE_U32    = MMA_K / 4u;
// Distance between two K slabs of A in shared memory (A_SLAB_U32 plus the option's padding).
const uint A_SLAB_PITCH_U32 = A_SLAB_U32 + A_PAD_U32;
""")
rep("const uint ASH_SLICE_U32 = NUM_K_SLABS * A_SLAB_U32;", "const uint ASH_SLICE_U32 = NUM_K_SLABS * A_SLAB_PITCH_U32;")
rep("""a_lds_off[ai] = (kb / (MMA_K >> 2u)) * A_SLAB_U32""", """a_lds_off[ai] = (kb / (MMA_K >> 2u)) * A_SLAB_PITCH_U32""")
rep("""      (a_k_block / (MMA_K >> 2u)) * A_SLAB_U32""", """      (a_k_block / (MMA_K >> 2u)) * A_SLAB_PITCH_U32""")
rep("const uint slab_a_base_u32 = cur_a + k * A_SLAB_U32;", "const uint slab_a_base_u32 = cur_a + k * A_SLAB_PITCH_U32;")

# Csh in Ash
rep("""#ifdef IO_TEXTURE
// Result staging for the imageStore epilogue""", """#if defined(IO_TEXTURE) && !defined(RX_CSH_IN_ASH)
// Result staging for the imageStore epilogue""")
rep("""const uint CSH_ROWS = SG_GRID_Y * MMA_M;
shared float16_t Csh_out[CSH_ROWS * WG_TILE_N];
#endif""", """const uint CSH_ROWS = SG_GRID_Y * MMA_M;
shared float16_t Csh_out[CSH_ROWS * WG_TILE_N];
#endif
#if defined(IO_TEXTURE) && defined(RX_CSH_IN_ASH)
// The fp16 drain tile (SG_GRID_Y bands of MMA_M rows, WG_TILE_N wide, row-major) lives in Ash_int8, as uint pairs of fp16.
// It is first written after the barrier that ends the last MMA reads of Ash (the first barrier of the drain loop below).
const uint CSH_ROWS = SG_GRID_Y * MMA_M;
const uint CSH_ROW_U32 = WG_TILE_N / 2u;
#endif""")

# prof macros
rep("""void main() {
  const uvec2 tileID""", """#ifdef RX_PROF
#define PTICK(acc) { const uint p_n = clock2x32ARB().x; acc += (p_n - p_t) & 0xFFFFFu; p_t = p_n; }
#else
#define PTICK(acc)
#endif

#ifndef RX_ABL
#define RX_ABL 0
#endif

void main() {
#ifdef RX_PROF
  uint p_pro = 0u, p_bar = 0u, p_pre = 0u, p_mma = 0u, p_st = 0u, p_ep = 0u, p_dr = 0u, p_wr = 0u;
  uint p_t = clock2x32ARB().x;
#endif
  const uvec2 tileID""")
rep("""  uint chunk = 0;
  for (uint group_i = 0; group_i < num_groups; ++group_i) {""", """  PTICK(p_pro)
  uint chunk = 0;
  for (uint group_i = 0; group_i < num_groups; ++group_i) {""")
rep("""      memoryBarrierShared();
      barrier();

      // --- 2. prefetch chunk+1 -> temp ---""", """#if (RX_ABL & 16) == 0
      memoryBarrierShared();
      barrier();
#endif
      PTICK(p_bar)

      // --- 2. prefetch chunk+1 -> temp ---""")
# ablation of fetches inside the loop: A and B prefetch
rep("""        const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;
#ifndef A_ALWAYS_ACTIVE
        if (a_active)
#endif
        {
#ifdef A_MULTI_BLOCK
          [[unroll]] for (uint ai = 0; ai < A_BLOCKS; ++ai) {
            temp_A[ai] = t_packed_int8_input[a_glb[ai] + (chunkK_nxt >> 2u)];
          }""", """        const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;
#if (RX_ABL & 1) == 0
#ifndef A_ALWAYS_ACTIVE
        if (a_active)
#endif
        {
#ifdef A_MULTI_BLOCK
          [[unroll]] for (uint ai = 0; ai < A_BLOCKS; ++ai) {
            temp_A[ai] = t_packed_int8_input[a_glb[ai] + (chunkK_nxt >> 2u)];
          }""")
rep("""          temp_A = t_packed_int8_input[a_glb_row + (chunkK_nxt >> 2u) + a_k_block];
#endif
        }
#ifdef WEIGHT_INT4
        [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
          const uint k4_blk = (chunkK_nxt >> 2u) + b_k4off[si];
#ifdef WEIGHT_BUFFER
          temp_B[si] = t_packed_weight[(b_n8blk[si] * nblocks_x_A) + k4_blk];
#else
          temp_B[si] = texelFetch(t_packed_weight, ivec2(k4_blk, b_n8blk[si]), 0);
#endif
        }""", """          temp_A = t_packed_int8_input[a_glb_row + (chunkK_nxt >> 2u) + a_k_block];
#endif
        }
#endif // RX_ABL & 1
#ifdef WEIGHT_INT4
#if (RX_ABL & 2) == 0
        [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
          const uint k4_blk = (chunkK_nxt >> 2u) + b_k4off[si];
#ifdef WEIGHT_BUFFER
          temp_B[si] = t_packed_weight[(b_n8blk[si] * nblocks_x_A) + k4_blk];
#else
          temp_B[si] = texelFetch(t_packed_weight, ivec2(k4_blk, b_n8blk[si]), 0);
#endif
        }
#endif // RX_ABL & 2""")
rep("""      }

      // --- 3. int8 MMA on the cur slice ---
      [[unroll]] for (uint k = 0; k < NUM_K_SLABS; ++k) {""", """      }
      PTICK(p_pre)

      // --- 3. int8 MMA on the cur slice ---
#if (RX_ABL & 8) == 0
      [[unroll]] for (uint k = 0; k < NUM_K_SLABS; ++k) {""")
rep("""            accum_int32[i][j] = coopMatMulAdd(matA[i], matB, accum_int32[i][j]);
          }
        }
      }

      // --- 4. store temp (chunk+1) -> nxt slice ---
      if (has_next) {""", """            accum_int32[i][j] = coopMatMulAdd(matA[i], matB, accum_int32[i][j]);
          }
        }
      }
#endif // RX_ABL & 8
      PTICK(p_mma)

      // --- 4. store temp (chunk+1) -> nxt slice ---
#if (RX_ABL & 4) == 0
      if (has_next) {""")
rep("""            Bsh_int8[nxt_b + slab_idx * B_SLAB_U32 + (n_col_base + n_in_blk) * B_STRIDE_U32 + k4_in_slab] =
                uint(temp_B[n_in_blk]);
          }
        }
#endif
      }
    }  // chunks""", """            Bsh_int8[nxt_b + slab_idx * B_SLAB_U32 + (n_col_base + n_in_blk) * B_STRIDE_U32 + k4_in_slab] =
                uint(temp_B[n_in_blk]);
          }
        }
#endif
      }
#endif // RX_ABL & 4
      PTICK(p_st)
    }  // chunks""")
rep("""          accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
        }
      }
    }
  }  // groups""", """          accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
        }
      }
    }
    PTICK(p_ep)
  }  // groups""")
rep("""  // --- Bias (optional) ---
#ifdef HAS_BIAS
  if (apply_bias > 0) {
    for (uint t""", """  PTICK(p_ep)
  // --- Bias (optional) ---
#ifdef HAS_BIAS
  if (apply_bias > 0) {
    for (uint t""")
# drain with Csh in Ash
rep("""      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> out_tile =
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]);
      coopMatStore(
          out_tile, Csh_out,
          warpInTile.y * MMA_M * WG_TILE_N +
              MMA_N * (MMAS_PER_SG_N * warpInTile.x + j),
          WG_TILE_N,
          gl_CooperativeMatrixLayoutRowMajor);
    }
    memoryBarrierShared();
    barrier();

    for (uint t = gl_LocalInvocationID.x; t < CSH_TEXELS; t += WG_SIZE) {""", """      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> out_tile =
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]);
#ifdef RX_CSH_IN_ASH
      coopMatStore(
          out_tile, Ash_int8,
          warpInTile.y * MMA_M * CSH_ROW_U32 +
              (MMA_N / 2u) * (MMAS_PER_SG_N * warpInTile.x + j),
          CSH_ROW_U32,
          gl_CooperativeMatrixLayoutRowMajor);
#else
      coopMatStore(
          out_tile, Csh_out,
          warpInTile.y * MMA_M * WG_TILE_N +
              MMA_N * (MMAS_PER_SG_N * warpInTile.x + j),
          WG_TILE_N,
          gl_CooperativeMatrixLayoutRowMajor);
#endif
    }
    memoryBarrierShared();
    barrier();
    PTICK(p_dr)

    for (uint t = gl_LocalInvocationID.x; t < CSH_TEXELS; t += WG_SIZE) {""")
rep("""      const uint base = lr * WG_TILE_N + lc4 * 4u;
      imageStore(
          t_output,
          ivec3(tile_n_start / 4u + lc4, m, 0),
          vec4(
              float(Csh_out[base]),
              float(Csh_out[base + 1u]),
              float(Csh_out[base + 2u]),
              float(Csh_out[base + 3u])));
    }
  }
#else""", """#ifdef RX_CSH_IN_ASH
      const uint base = lr * CSH_ROW_U32 + lc4 * 2u;
      imageStore(
          t_output,
          ivec3(tile_n_start / 4u + lc4, m, 0),
          vec4(unpackHalf2x16(Ash_int8[base]), unpackHalf2x16(Ash_int8[base + 1u])));
#else
      const uint base = lr * WG_TILE_N + lc4 * 4u;
      imageStore(
          t_output,
          ivec3(tile_n_start / 4u + lc4, m, 0),
          vec4(
              float(Csh_out[base]),
              float(Csh_out[base + 1u]),
              float(Csh_out[base + 2u]),
              float(Csh_out[base + 3u])));
#endif
    }
    PTICK(p_wr)
  }
#else""")
rep("""#endif // IO_TEXTURE
}
""", """#endif // IO_TEXTURE

#ifdef RX_PROF
  // PROF output: subgroup 0 writes the phase means into the first 8 elements of its tile.
  PTICK(p_wr)
  {
    const float it = float(max(num_chunks, 1u));
    const vec4 v0 = vec4(float(p_bar), float(p_pre), float(p_mma), float(p_st)) / it;
    const vec4 v1 = vec4(float(p_pro), float(p_ep), float(p_dr), float(p_wr)) / 64.0;
#ifdef IO_TEXTURE
    memoryBarrierImage();
    barrier();
    if (gl_SubgroupID == 0u && subgroupElect()) {
      imageStore(t_output, ivec3(tile_n_start / 4u, tile_m_start, 0), v0);
      imageStore(t_output, ivec3(tile_n_start / 4u + 1u, tile_m_start, 0), v1);
    }
#else
    memoryBarrierBuffer();
    barrier();
    if (gl_SubgroupID == 0u && subgroupElect()) {
      const uint pb = tile_m_start * uint(out_N_arg) + tile_n_start;
      [[unroll]] for (uint c = 0; c < 4u; ++c) {
        t_output[pb + c] = float16_t(v0[c]);
        t_output[pb + 4u + c] = float16_t(v1[c]);
      }
    }
#endif
  }
#endif // RX_PROF
}
""")

# ---- second pass (round 2, interleaved stores; the anchors are the text produced by the edits above) ----
A_SNIP = """#ifdef A_MULTI_BLOCK
          [[unroll]] for (uint ai = 0; ai < A_BLOCKS; ++ai) {
            [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
              Ash_int8[nxt_a + a_lds_off[ai] + m4i * A_STRIDE_U32] = uint(temp_A[ai][m4i]);
            }
          }
#else
          [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
            Ash_int8[nxt_a + a_lds_off0 + m4i * A_STRIDE_U32] = uint(temp_A[m4i]);
          }
#endif
"""
B_SNIP = """          [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
            Bsh_int8[nxt_b + b_lds_off[si]] =
                widen_nibbles(uint(temp_B[si][b_comp[si]]), b_par[si]);
          }
"""
# R1: branch-free prefetch
rep("""      // --- 2. prefetch chunk+1 -> temp ---
      if (has_next) {
        const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;
""", """      // --- 2. prefetch chunk+1 -> temp ---
#ifdef RX_BF
      {
        const uint chunkK_nxt = min(chunk + 1u, num_chunks - 1u) * WG_TILE_K;
#else
      if (has_next) {
        const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;
#endif
""")
# R2: stores inside the slab loop
rep("""            accum_int32[i][j] = coopMatMulAdd(matA[i], matB, accum_int32[i][j]);
          }
        }
      }
#endif // RX_ABL & 8""", """            accum_int32[i][j] = coopMatMulAdd(matA[i], matB, accum_int32[i][j]);
          }
        }
#if (RX_ABL & 4) == 0
#if defined(RX_ST_A) || defined(RX_ST_B)
#if !defined(RX_BF) || !defined(WEIGHT_INT4) || !defined(A_ALWAYS_ACTIVE)
#error "RX_ST_A / RX_ST_B need RX_BF, WEIGHT_INT4 and A_ALWAYS_ACTIVE"
#endif
#endif
#ifdef RX_ST_A
        if (k == RX_ST_A) {
""" + A_SNIP + """        }
#endif
#ifdef RX_ST_B
        if (k == RX_ST_B) {
""" + B_SNIP + """        }
#endif
#endif // RX_ABL & 4
      }
#endif // RX_ABL & 8""")
# R3: the post-loop stores: branch-free option; the parts moved into the slab loop are not repeated
rep("""#if (RX_ABL & 4) == 0
      if (has_next) {
#ifndef A_ALWAYS_ACTIVE
        if (a_active)
#endif
        {
""" + A_SNIP.replace("          [[unroll]]", "          [[unroll]]") + """        }
#ifdef WEIGHT_INT4
        [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
          Bsh_int8[nxt_b + b_lds_off[si]] =
              widen_nibbles(uint(temp_B[si][b_comp[si]]), b_par[si]);
        }
""", """#if (RX_ABL & 4) == 0
#ifdef RX_BF
      {
#else
      if (has_next) {
#endif
#ifndef RX_ST_A
#ifndef A_ALWAYS_ACTIVE
        if (a_active)
#endif
        {
""" + A_SNIP + """        }
#endif // RX_ST_A
#ifdef WEIGHT_INT4
#ifndef RX_ST_B
        [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
          Bsh_int8[nxt_b + b_lds_off[si]] =
              widen_nibbles(uint(temp_B[si][b_comp[si]]), b_par[si]);
        }
#endif // RX_ST_B
""")

# ---- third pass (round 2): RX_UV4, shared staging arrays typed uvec4 so that every coopMatLoad of A / B is 16-byte aligned ----
import re as _re
def _balanced(s, i):
    """s[i] == '[': index just after the matching ']'."""
    d = 0
    while True:
        if s[i] == "[": d += 1
        elif s[i] == "]":
            d -= 1
            if d == 0: return i + 1
        i += 1
out, i, n_st = [], 0, 0
pat = _re.compile(r"\b(Ash_int8|Bsh_int8)\[")
while True:
    m = pat.search(t, i)
    if not m:
        out.append(t[i:]); break
    j = _balanced(t, m.end() - 1)
    k = j
    while t[k] in " \t\n": k += 1
    if t[k] == "=" and t[k + 1] != "=":
        e = t.index(";", k)
        out.append(t[i:m.start()]); out.append(f"{'ASH' if m.group(1).startswith('A') else 'BSH'}_ST({t[m.end():j - 1]}, {t[k + 1:e].strip()})"); i = e
        n_st += 1
    else:
        out.append(t[i:j]); i = j
t = "".join(out)
assert n_st == 11, n_st
rep("const uint ASH_SLICE_U32 = NUM_K_SLABS * A_SLAB_PITCH_U32;", "const uint ASH_SLICE_U32 = NUM_K_SLABS * A_SLAB_PITCH_U32;")
rep("""// Double-buffered MMA operand staging.
shared uint Ash_int8[2u * ASH_SLICE_U32];
shared uint Bsh_int8[2u * BSH_SLICE_U32];
""", """// Double-buffered MMA operand staging.
#ifdef RX_UV4
// uvec4 arrays: the compiler then knows that every fragment load is 16-byte aligned (one ds_read_b128 instead of two ds_read_b64);
// the 32-bit stores of the staging write one component of an element. coopMatLoad offsets and strides count elements (uvec4).
#ifdef RX_CSH_IN_ASH
#error "RX_UV4 is not combined with RX_CSH_IN_ASH"
#endif
shared uvec4 Ash_int8[2u * ASH_SLICE_U32 / 4u];
shared uvec4 Bsh_int8[2u * BSH_SLICE_U32 / 4u];
#define ASH_ST(idx, v) Ash_int8[(idx) >> 2u][(idx) & 3u] = (v)
#define BSH_ST(idx, v) Bsh_int8[(idx) >> 2u][(idx) & 3u] = (v)
#define RX_LDO(o) ((o) >> 2u)
#define RX_LDS(s) ((s) >> 2u)
#else
shared uint Ash_int8[2u * ASH_SLICE_U32];
shared uint Bsh_int8[2u * BSH_SLICE_U32];
#define ASH_ST(idx, v) Ash_int8[idx] = (v)
#define BSH_ST(idx, v) Bsh_int8[idx] = (v)
#define RX_LDO(o) (o)
#define RX_LDS(s) (s)
#endif
""")
rep("""              slab_a_base_u32 + row_a * A_STRIDE_U32,
              A_STRIDE_U32,""", """              RX_LDO(slab_a_base_u32 + row_a * A_STRIDE_U32),
              RX_LDS(A_STRIDE_U32),""")
rep("""              slab_b_base_u32 + col_b * B_STRIDE_U32,
              B_STRIDE_U32,""", """              RX_LDO(slab_b_base_u32 + col_b * B_STRIDE_U32),
              RX_LDS(B_STRIDE_U32),""")
rep("//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY", """//   RX_UV4: the shared staging arrays are uvec4 (aligned 128-bit fragment loads; component-wise 32-bit stores).
//   RX_PROF: shader-clock phase timing, MEASUREMENT ONLY""")


# ---- fourth pass (round 2): row pitch of the A / B staging in shared memory (A_PITCH_U32 / B_PITCH_U32 from the wrapper, default 16 bytes) ----
rep("""const uint A_SLAB_INT8     = WG_TILE_M * MMA_K;""", """const uint A_SLAB_INT8     = WG_TILE_M * MMA_K;""")
rep("""const uint B_STRIDE_U32    = B_USEFUL_U32;
const uint B_SLAB_U32      = WG_TILE_N * B_STRIDE_U32;""", """// RX row pitch: A_PITCH_U32 / B_PITCH_U32 (uint per LDS row of one K slab, default MMA_K / 4 = 16 bytes). A pitch of 6 (24 bytes) makes
// the two ds_read_b64 per fragment row conflict-free over 16 lanes (a 16-byte pitch puts lanes l and l + 8 on the same banks).
const uint B_STRIDE_U32    = B_PITCH_U32;
const uint B_SLAB_U32      = WG_TILE_N * B_STRIDE_U32;
const uint B_DENSE_SLAB    = WG_TILE_N * B_USEFUL_U32;""")
rep("""const uint A_SLAB_U32      = A_SLAB_INT8 / 4u;
const uint A_STRIDE_U32    = MMA_K / 4u;""", """const uint A_STRIDE_U32    = A_PITCH_U32;
const uint A_SLAB_U32      = WG_TILE_M * A_STRIDE_U32;""")
rep("""    const uint a           = gl_LocalInvocationID.x + si * WG_SIZE;
    const uint slab_idx    = a / B_SLAB_U32;
    const uint local_a     = a % B_SLAB_U32;
    const uint n_col       = local_a / B_STRIDE_U32;
    const uint k4_in_slab  = local_a % B_STRIDE_U32;""", """    const uint a           = gl_LocalInvocationID.x + si * WG_SIZE;
#ifdef RX_PITCH
    // dense thread index a -> (slab, column, k4); the LDS address uses the padded pitch
    const uint slab_idx    = a / B_DENSE_SLAB;
    const uint local_a     = a % B_DENSE_SLAB;
    const uint n_col       = local_a / B_USEFUL_U32;
    const uint k4_in_slab  = local_a % B_USEFUL_U32;
#else
    const uint slab_idx    = a / B_SLAB_U32;
    const uint local_a     = a % B_SLAB_U32;
    const uint n_col       = local_a / B_STRIDE_U32;
    const uint k4_in_slab  = local_a % B_STRIDE_U32;
#endif""")
rep("""    b_lds_off[si] = a;""", """#ifdef RX_PITCH
    b_lds_off[si] = slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab;
#else
    b_lds_off[si] = a;
#endif""")
rep("//   RX_UV4: the shared staging arrays", """//   A_PITCH_U32 / B_PITCH_U32 (constants from the wrapper) and RX_PITCH (set when either is not 4): row pitch of the A / B staging in shared
//     memory in uint (default 4 = 16 bytes); 6 = 24 bytes.
//   RX_UV4: the shared staging arrays""")

open(dst, "w").write(t)
print("ok", len(t.splitlines()), "lines")
