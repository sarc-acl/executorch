/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * TILE/SUBGROUP-SWEEP variant of linear_q4gsw_coopmat's dbuf4 ("store-first",
 * prefetch-peeled) loop structure (specs/041-dbuf4-tile-sweep). Forked from
 * linear_q4gsw_coopmat_tsweep.glsl (which carries dbuf1's loop, the
 * production winner) -- everything except the PROLOGUE/MAIN LOOP block is
 * identical: same bindings, spec-constants, tile-geometry templating,
 * dequant_block, fp16 accumulator, bias/store epilogue. Only the loop
 * structure is swapped to dbuf4, recovered from the pre-retune reference
 * (git commit 8d0f23ee78's sibling; see specs/041/reference/) that predates
 * the dbuf1-winner fp16-accumulate/flattened-loop retune. dbuf4:
 *   prologue: prefetch chunk 0 -> temp, store to slice 0 (no barrier)
 *   per iter: barrier -> prefetch(next) -> MMA(cur) -> store(next)
 *   epilogue: barrier -> MMA(last)        [loop peeled, 1 barrier/iter]
 * vs. dbuf1's single flattened loop with a trailing conditional barrier
 * (skipped on the last chunk) -- dbuf4 issues one barrier per chunk
 * (num_chunks total) instead of dbuf1's (num_chunks - 1).
 *
 * Selected at dispatch via
 * ET_VK_Q4GSW_COOPMAT_VARIANT=tsweep_dbuf4_t<M>x<N>k<K>g<SGX><SGY>s<sub>
 * (QuantizedLinear.cpp), additive to the existing tsweep_t... (dbuf1)
 * namespace.
 *
 * Hard preconditions (no shape/alignment checks inside the shader):
 *   M % WG_TILE_M == 0
 *   N % WG_TILE_N == 0
 *   K % WG_TILE_K == 0
 *   group_size % WG_TILE_K == 0
 * Misaligned shapes silently miscompute / overrun -- gated at dispatch time
 * by can_use_q4gsw_coopmat() using the ACTIVE variant's own tile dims.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if HAS_BIAS:
  #define HAS_BIAS

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

$if IO_STORAGE == "texture3d":
  #define IO_TEXTURE

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output",         "half", IO_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input",          "half", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_weight",  "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales",  "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",           "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

// --- Tile geometry (from yaml; per-variant tile-sweep candidate) ---
const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};

const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};

const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
const uint NUM_SUBGROUPS = SG_GRID_X * SG_GRID_Y;
const uint WG_SIZE = NUM_SUBGROUPS * SUBGROUP_SIZE;

const uint SG_TILE_M = WG_TILE_M / SG_GRID_Y;
const uint SG_TILE_N = WG_TILE_N / SG_GRID_X;
const uint MMAS_PER_SG_M = SG_TILE_M / MMA_M;
const uint MMAS_PER_SG_N = SG_TILE_N / MMA_N;

// fp16: 8 elements per uvec4 (128-bit)
const uint FP16_PER_VEC4 = 8;
const uint A_STRIDE_VEC4 = (WG_TILE_K + FP16_PER_VEC4) / FP16_PER_VEC4;
const uint B_STRIDE_VEC4 = (WG_TILE_N + FP16_PER_VEC4) / FP16_PER_VEC4;

// One ping-pong slice of each shared-memory buffer (in uvec4 units).
const uint ASH_SLICE = WG_TILE_M * A_STRIDE_VEC4;
const uint BSH_SLICE = WG_TILE_K * B_STRIDE_VEC4;

// Double-buffered shared memory.
shared uvec4 Ash[2 * ASH_SLICE];
shared uvec4 Bsh[2 * BSH_SLICE];
#ifdef HAS_BIAS
shared float16_t bias_sh[WG_TILE_N];
#endif

#ifdef IO_TEXTURE
// Result staging for the imageStore epilogue: SG_GRID_Y bands of MMA_M rows,
// each WG_TILE_N wide, row-major. A full WG_TILE_M x WG_TILE_N buffer would
// cost SG_GRID_Y/MMAS_PER_SG_M x more LDS and wreck occupancy. Not aliased
// with Ash/Bsh: GLSL has no shared unions and coopMatStore needs a
// float16_t-typed array.
const uint CSH_ROWS = SG_GRID_Y * MMA_M;
shared float16_t Csh[CSH_ROWS * WG_TILE_N];
#endif

// Staging thread maps: each thread covers one uvec4 (8 fp16) per pass.
const uint INVS_PER_ROW_A = WG_TILE_K / FP16_PER_VEC4;
const uint A_ROWS_PER_PASS = WG_SIZE / INVS_PER_ROW_A;
const uint A_PASSES = WG_TILE_M / A_ROWS_PER_PASS;
const uint INVS_PER_ROW_B = WG_TILE_N / FP16_PER_VEC4;
const uint B_ROWS_PER_PASS = WG_SIZE / INVS_PER_ROW_B;
const uint B_PASSES = WG_TILE_K / B_ROWS_PER_PASS;

// FP16 accumulator coopmats (MMAS_PER_SG_M x MMAS_PER_SG_N per thread).
coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    result[MMAS_PER_SG_M][MMAS_PER_SG_N];

// Dequant one packed INT4 block column-pair into 8 scaled fp16 weights
// (one Bsh uvec4). col_lo/col_hi select the K row within the block.
uvec4 dequant_block(
    const ivec4 wb,
    const uint col_lo,
    const uint col_hi,
    const f16vec4 s0,
    const f16vec4 s1) {
  const f16vec4 v0 = f16vec4(((wb >> int(4u * col_lo)) & 0xF) - 8) * s0;
  const f16vec4 v1 = f16vec4(((wb >> int(4u * col_hi)) & 0xF) - 8) * s1;
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
}

// Fetch 8 consecutive fp16 activations of `row` starting at half-vec4 index
// `k_hv4`, packed into one uvec4. Both spellings address the same bytes: a
// width-packed texture3d holds elements [4x, 4x+3] of row m at texel (x, m, 0),
// so buffer index (row * K4 + x) and texel coord (x, row, 0) are one address in
// two notations.
uvec4 load_a_vec4(const uint row, const uint k_hv4, const uint K4) {
#ifdef IO_TEXTURE
  // Narrow to f16vec4 AT the fetch: a half sampler is typed to return vec4
  // (fp32), and consuming the fp32 costs an image_load into 4 VGPRs plus a
  // v_cvt_pk_rtz per pair. Taking it as f16vec4 immediately lets the compiler
  // fold the narrowing into `image_load ... d16` (2 VGPRs, no conversion).
  // Lossless -- the source is rgba16f.
  const f16vec4 v0 = f16vec4(texelFetch(t_input, ivec3(k_hv4, row, 0), 0));
  const f16vec4 v1 = f16vec4(texelFetch(t_input, ivec3(k_hv4 + 1u, row, 0), 0));
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
#else
  const f16vec4 v0 = t_input[row * K4 + k_hv4];
  const f16vec4 v1 = t_input[row * K4 + k_hv4 + 1u];
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
#endif
}

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % SG_GRID_X,
      gl_SubgroupID / SG_GRID_X);

  const uint K = uint(input_sizes.x);
  const uint K4 = (K + 3u) / 4u;
  const uint N4 = (uint(output_sizes.x) + 3u) / 4u;

  const uint CHUNKS_PER_GROUP = uint(K4_per_group) * 4u / WG_TILE_K;
  const uint num_chunks = uint(num_groups_arg) * CHUNKS_PER_GROUP;

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
    }
  }

  const uint a_col = gl_LocalInvocationID.x % INVS_PER_ROW_A;
  const uint a_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_A;
  const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
  const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;

  const uint n8_blk = (tile_n_start + b_col * 8u) >> 3u;
  const uint col_lo = 2u * (b_row_offset & 3u);
  const uint col_hi = col_lo + 1u;

  const uint sc_n4 = (tile_n_start + b_col * 8u) >> 2u;
  uint cached_group = 0xFFFFFFFFu;
  f16vec4 sc0;
  f16vec4 sc1;

  uvec4 temp_A[A_PASSES];
  ivec4 temp_B[B_PASSES];

  // =========================================================
  // PROLOGUE (dbuf4): prefetch chunk 0 into temp registers, then store to
  // slice 0. No barrier here -- the main loop's first iteration barriers
  // before reading slice 0.
  // =========================================================
  {
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
      const uint row = tile_m_start + p * A_ROWS_PER_PASS + a_row_offset;
      const uint k_hv4 = (a_col * FP16_PER_VEC4) / 4u;
      temp_A[p] = load_a_vec4(row, k_hv4, K4);
    }
    cached_group = 0u;
    sc0 = t_weight_scales[sc_n4];
    sc1 = t_weight_scales[sc_n4 + 1u];
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
      const uint k_row = p * B_ROWS_PER_PASS + b_row_offset;
      ivec4 wblock;
#ifdef WEIGHT_BUFFER
      wblock = t_packed_weight[n8_blk * K4 + (k_row >> 2u)];
#else
      wblock = texelFetch(t_packed_weight, ivec2(k_row >> 2u, n8_blk), 0);
#endif
      temp_B[p] = wblock;
    }
  }
  {
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
      Ash[(p * A_ROWS_PER_PASS + a_row_offset) * A_STRIDE_VEC4 + a_col] = temp_A[p];
    }
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
      Bsh[(p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
          dequant_block(temp_B[p], col_lo, col_hi, sc0, sc1);
    }
  }

  // =========================================================
  // MAIN LOOP (dbuf4) — one barrier per iteration, unconditional body, loop
  // peeled (bound excludes the last chunk). Iteration `chunk` does:
  //   1. barrier      — slice (chunk%2) fully written
  //   2. prefetch     — chunk+1 from global into temp (in flight during math)
  //   3. MMA math     — on slice (chunk%2)
  //   4. store        — temp (chunk+1, dequantized) into slice ((chunk+1)%2)
  // =========================================================
  uint chunk;
  for (chunk = 0; chunk + 1u < num_chunks; ++chunk) {
    const uint cur_base_A = (chunk % 2u) * ASH_SLICE;
    const uint cur_base_B = (chunk % 2u) * BSH_SLICE;
    const uint nxt_base_A = ((chunk + 1u) % 2u) * ASH_SLICE;
    const uint nxt_base_B = ((chunk + 1u) % 2u) * BSH_SLICE;

    barrier();

    // --- prefetch chunk+1 -> temp ---
    {
      const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;

      [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
        const uint row = tile_m_start + p * A_ROWS_PER_PASS + a_row_offset;
        const uint k_hv4 = (chunkK_nxt + a_col * FP16_PER_VEC4) / 4u;
        temp_A[p] = load_a_vec4(row, k_hv4, K4);
      }
      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
        const uint k_row = chunkK_nxt + p * B_ROWS_PER_PASS + b_row_offset;
#ifdef WEIGHT_BUFFER
        temp_B[p] = t_packed_weight[n8_blk * K4 + (k_row >> 2u)];
#else
        temp_B[p] = texelFetch(t_packed_weight, ivec2(k_row >> 2u, n8_blk), 0);
#endif
      }
      const uint group_nxt = (chunk + 1u) / CHUNKS_PER_GROUP;
      if (group_nxt != cached_group) {
        cached_group = group_nxt;
        sc0 = t_weight_scales[group_nxt * N4 + sc_n4];
        sc1 = t_weight_scales[group_nxt * N4 + sc_n4 + 1u];
      }
    }

    // --- MMA math on the cur slice ---
    [[unroll]] for (uint k = 0; k < WG_TILE_K / MMA_K; ++k) {
      const uint k_start = MMA_K * k;

      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
      [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
        coopMatLoad(
            matA[i], Ash,
            cur_base_A + row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
            A_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);
      }

      coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
      [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
        const uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j) / FP16_PER_VEC4;
        coopMatLoad(
            matB, Bsh,
            cur_base_B + k_start * B_STRIDE_VEC4 + col_b,
            B_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);

        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
        }
      }
    }

    // --- store temp (chunk+1) -> nxt slice, dequantizing B ---
    {
      [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
        Ash[nxt_base_A + (p * A_ROWS_PER_PASS + a_row_offset) * A_STRIDE_VEC4 + a_col] =
            temp_A[p];
      }
      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
        Bsh[nxt_base_B + (p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
            dequant_block(temp_B[p], col_lo, col_hi, sc0, sc1);
      }
    }
  }

  // --- epilogue: barrier, then MMA on the last chunk (loop peeled) ---
  {
    const uint cur_base_A = (chunk % 2u) * ASH_SLICE;
    const uint cur_base_B = (chunk % 2u) * BSH_SLICE;

    barrier();

    [[unroll]] for (uint k = 0; k < WG_TILE_K / MMA_K; ++k) {
      const uint k_start = MMA_K * k;

      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
      [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
        coopMatLoad(
            matA[i], Ash,
            cur_base_A + row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
            A_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);
      }

      coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
      [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
        const uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j) / FP16_PER_VEC4;
        coopMatLoad(
            matB, Bsh,
            cur_base_B + k_start * B_STRIDE_VEC4 + col_b,
            B_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);

        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
        }
      }
    }
  }

  // --- Bias staging (if any) ---
#ifdef HAS_BIAS
  if (apply_bias > 0) {
    for (uint t = gl_LocalInvocationID.x; t < WG_TILE_N; t += WG_SIZE) {
      bias_sh[t] = float16_t(t_bias[tile_n_start + t]);
    }
    memoryBarrierShared();
    barrier();
  }
#endif

  // --- Store result tile ---
#ifdef IO_TEXTURE
  // Epilogue iteration i drains accumulator row-block i from EVERY subgroup
  // into Csh at once, so the SG_GRID_Y bands it holds are disjoint global row
  // ranges; the whole workgroup then imageStores them. The band-to-global-row
  // map below reproduces the buffer path's gi exactly:
  //   buffer: gi = tile_m_start + MMA_M * (MMAS_PER_SG_M * warpInTile.y + i)
  //              = tile_m_start + warpInTile.y * SG_TILE_M + i * MMA_M
  // and lr / MMA_M is the writing subgroup's warpInTile.y.
  //
  // PORTABILITY NOTE: [[unroll]] is only a hint and glslc does not honor it
  // here -- the barrier() in the loop body keeps the loop rolled, so
  // result[i][j] IS dynamically indexed. Coopmat arrays are opaque per-lane
  // storage and dynamic indexing is exactly the construct the Xclipse/AMD-PAL
  // compiler has broken before, so check this first if the texture variants
  // miscompile on M51. Fully unrolling would need the drain hand-expanded so
  // each i gets its own barrier.
  const uint CSH_TEXELS_PER_ROW = WG_TILE_N / 4u;
  const uint CSH_TEXELS = CSH_ROWS * CSH_TEXELS_PER_ROW;
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    // Guards Csh against the previous iteration's readers. Inert on i == 0 but
    // must stay unconditional to remain workgroup-uniform.
    barrier();
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
#ifdef HAS_BIAS
      if (apply_bias > 0) {
        const uint local_n = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
        coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> bias_tile;
        coopMatLoad(
            bias_tile, bias_sh,
            local_n, /*stride=*/0u,
            gl_CooperativeMatrixLayoutRowMajor);
        result[i][j] += bias_tile;
      }
#endif
      coopMatStore(
          result[i][j], Csh,
          warpInTile.y * MMA_M * WG_TILE_N +
              MMA_N * (MMAS_PER_SG_N * warpInTile.x + j),
          WG_TILE_N,
          gl_CooperativeMatrixLayoutRowMajor);
    }
    memoryBarrierShared();
    barrier();

    for (uint t = gl_LocalInvocationID.x; t < CSH_TEXELS; t += WG_SIZE) {
      const uint lr = t / CSH_TEXELS_PER_ROW;
      const uint lc4 = t % CSH_TEXELS_PER_ROW;
      const uint m =
          tile_m_start + (lr / MMA_M) * SG_TILE_M + i * MMA_M + (lr % MMA_M);
      const uint base = lr * WG_TILE_N + lc4 * 4u;
      imageStore(
          t_output,
          ivec3(tile_n_start / 4u + lc4, m, 0),
          vec4(
              float(Csh[base]),
              float(Csh[base + 1u]),
              float(Csh[base + 2u]),
              float(Csh[base + 3u])));
    }
  }
#else
  const uint N_out = uint(out_N_arg);
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      const uint gi = tile_m_start + MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
      const uint gj = tile_n_start + MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

#ifdef HAS_BIAS
      if (apply_bias > 0) {
        const uint local_n = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
        coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> bias_tile;
        coopMatLoad(
            bias_tile, bias_sh,
            local_n, /*stride=*/0u,
            gl_CooperativeMatrixLayoutRowMajor);
        result[i][j] += bias_tile;
      }
#endif

      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> out_tile =
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]);
      coopMatStore(
          out_tile, t_output,
          gi * N_out + gj, N_out,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
#endif // IO_TEXTURE
}
