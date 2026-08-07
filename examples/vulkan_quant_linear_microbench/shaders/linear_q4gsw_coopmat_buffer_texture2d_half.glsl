// REQUIRED_SUBGROUP_SIZE = 32
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix variant of the weight-only int4 quantized linear
 * tiled shader (WEIGHT_NBITS=4 in the yaml):
 *   4  ->  linear_q4gsw_coopmat   INT4 group-symmetric weight
 *          (group_size = 4 * K4_per_group)
 *
 * Performs: out[M,N] = activation[M,K] * weight^T[N,K] (+ bias)
 *
 * Inner-loop math is pure fp16 -> fp32 MMA via coopMatMulAdd for both
 * formats. The weight scale is applied during the B-tile store to shared
 * memory: each int weight is unpacked (nibble - 8 for INT4; bitfieldExtract
 * for INT8), cast to fp16, and multiplied by its scale before it lands in
 * Bsh, keeping the K-loop a clean fp16 MMA.
 *
 * Loop structure follows the NVIDIA double-buffered GEMM reference
 * (shmem_double_buf.comp, "prefetch-first" variant, aka dbuf1 — the winner
 * of the dbuf1..dbuf4 loop-structure sweep on M5 EVT1, see
 * report-for-human/dbuf-sweep-q4gsw-m2048.md; measured 1.87x faster than
 * the previous single-buffered skeleton at fp16 on Xclipse 970):
 *   - PROLOGUE: load tile 0 from global memory DIRECTLY into shared-memory
 *     slice 0 (no temp registers), then barrier.
 *   - Single flattened loop over all chunks, conditioned on `last`:
 *     prefetch the NEXT tile into temp (skipped on the last chunk) -> MMA
 *     math on the CURRENT slice -> store temp into the OTHER slice + barrier
 *     (both skipped on the last chunk). `last` is workgroup-uniform (the
 *     loop trip count is spec-const-derived), so the conditional barrier is
 *     uniformly executed.
 *   - Ping-pong shared-memory slices make the overlap safe.
 *
 * Loop trip count and coopMatStore width N come from spec constants
 * (num_groups_arg / out_N_arg), not the sizes UBO. The driver correctness
 * bugs that originally forced this are fixed, but spec consts let the
 * compiler resolve the coopmat K-loop bound at compile time (unroll) — a perf
 * win this branch is measuring (UBO method regressed 1B e2e ~0.97x vs tiled).
 *
 * Each thread keeps its 8 weight scales (2 f16vec4) in registers. For INT4
 * they are reloaded from global only when the prefetched chunk crosses a
 * group boundary (a workgroup-uniform branch); for INT8 (per-channel = a
 * single group spanning all of K) they are loaded once in the prologue.
 * There is no scales staging in shared memory and no extra barrier.
 *
 * Tile hierarchy (yaml; tile-sweep optimum for M5 EVT1, ~+25% over the
 * prior 128x128/4x2 layout — see report-for-human TODO "Update dbuf1 to
 * optimal tile size"):
 *   MMA_*         per-MMA-instruction shape (16x16x16 fp16)
 *   WG_TILE_*     output tile per workgroup (128x64)
 *   SG_GRID_*     subgroup grid inside workgroup (2x2 = 4 subgroups)
 *   SUBGROUP_SIZE 32, forced at pipeline creation via the
 *                 REQUIRED_SUBGROUP_SIZE annotation below
 *
 * Storage: activation/output forced to buffer; INT weight = texture2d or
 * buffer (yaml variant). DTYPE = half only.
 *
 * Hard preconditions (no shape/alignment checks inside the shader):
 *   M % WG_TILE_M == 0
 *   N % WG_TILE_N == 0
 *   K % WG_TILE_K == 0
 *   INT4: group_size % WG_TILE_K == 0  (each group = whole number of chunks)
 * Misaligned shapes silently miscompute / overrun — gate at dispatch time.
 */

// REQUIRED_SUBGROUP_SIZE = 32

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION highp

#define WEIGHT_INT4



layout(std430) buffer;

#include "common.glslh"

// Bindings — match the order used by add_linear_qw_node so the dispatch
// site can reuse the same arg layout.

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly t_outputBuffer {
    float16_t t_output[];
};


layout(set = 0, binding = 1) buffer PRECISION restrict readonly t_inputBuffer {
    f16vec4 t_input[];
};

layout(set = 0, binding = 2) uniform PRECISION isampler2D t_packed_weight;

layout(set = 0, binding = 3) buffer PRECISION restrict readonly t_weight_scalesBuffer {
    f16vec4 t_weight_scales[];
};


layout(set = 0, binding = 4) buffer PRECISION restrict readonly t_biasBuffer {
    float16_t t_bias[];
};



layout(set = 0, binding = 5) uniform PRECISION restrict readonly output_sizes_UBO {
  ivec4 output_sizes;
};

layout(set = 0, binding = 6) uniform PRECISION restrict readonly input_sizes_UBO {
  ivec4 input_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int apply_bias = 0;
// INT4 only; inert (0) for INT8 so the dispatcher's spec list lines up.
layout(constant_id = 4) const int K4_per_group = 0;
// PERF-ABLATION (2026-06-30): loop trip count + coopMatStore width N passed as
// spec constants again (not the sizes UBO). The driver correctness bugs that
// originally forced this are fixed, but the spec-const form lets the compiler
// resolve the coopmat K-loop bound at compile time (unroll) — testing whether
// the UBO method was the e2e perf regression. INT4: num quant groups.
layout(constant_id = 5) const int num_groups_arg = 0;
layout(constant_id = 6) const int out_N_arg = 0;

// --- Tile geometry (from yaml; defaults match coopmat_mm_ref) ---
const uint MMA_M = 16;
const uint MMA_N = 16;
const uint MMA_K = 16;

const uint WG_TILE_M = 128;
const uint WG_TILE_N = 128;
const uint WG_TILE_K = 16;

const uint SG_GRID_X = 2;
const uint SG_GRID_Y = 2;
const uint SUBGROUP_SIZE = 32;
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

// Staging thread maps: each thread covers one uvec4 (8 fp16) per pass.
const uint INVS_PER_ROW_A = WG_TILE_K / FP16_PER_VEC4;
const uint A_ROWS_PER_PASS = WG_SIZE / INVS_PER_ROW_A;
const uint A_PASSES = WG_TILE_M / A_ROWS_PER_PASS;
const uint INVS_PER_ROW_B = WG_TILE_N / FP16_PER_VEC4;
const uint B_ROWS_PER_PASS = WG_SIZE / INVS_PER_ROW_B;
const uint B_PASSES = WG_TILE_K / B_ROWS_PER_PASS;

// FP16 accumulator coopmats (MMAS_PER_SG_M x MMAS_PER_SG_N per thread).
// EXPERIMENT (2026-06-30): fp16 accumulate instead of fp32 — Xclipse 970
// exposes coopmat config #1 (f16 x f16 -> f16 accum), ~2x matrix throughput
// vs the f32-accum config #0. Precision risk over K=2048..4096; gated on the
// microbench correctness pass.
coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    result[MMAS_PER_SG_M][MMAS_PER_SG_N];

#ifdef WEIGHT_INT4

// Dequant one packed INT4 block column-pair into 8 scaled fp16 weights
// (one Bsh uvec4). col_lo/col_hi select the K row within the block.
//
// All 4 packed ints (wb.xyzw) share the SAME nibble bit-position for a given
// K row (col_lo / col_hi), so the extract is a pure vec4 op: one ivec4
// shift + mask + (-8) zero-point per K row instead of 8 scalar extracts. The
// int->fp16 cast and the per-column scale fold into a single f16vec4 multiply.
// (Idea 2 / upstream-style zero-ALU nibble split, no weight repack.)
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

#else // INT8

// Dequant 8 int8 weights (two ivec4 blocks, one K-row selected by shift)
// into 8 scaled fp16 weights (one Bsh uvec4).
uvec4 dequant_block(
    const ivec4 wa,
    const ivec4 wb,
    const int shift,
    const f16vec4 s0,
    const f16vec4 s1) {
  f16vec4 v0;
  v0.x = float16_t(bitfieldExtract(wa.x, shift, 8)) * s0.x;
  v0.y = float16_t(bitfieldExtract(wa.y, shift, 8)) * s0.y;
  v0.z = float16_t(bitfieldExtract(wa.z, shift, 8)) * s0.z;
  v0.w = float16_t(bitfieldExtract(wa.w, shift, 8)) * s0.w;
  f16vec4 v1;
  v1.x = float16_t(bitfieldExtract(wb.x, shift, 8)) * s1.x;
  v1.y = float16_t(bitfieldExtract(wb.y, shift, 8)) * s1.y;
  v1.z = float16_t(bitfieldExtract(wb.z, shift, 8)) * s1.z;
  v1.w = float16_t(bitfieldExtract(wb.w, shift, 8)) * s1.w;
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
}

#endif // WEIGHT_INT4

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % SG_GRID_X,
      gl_SubgroupID / SG_GRID_X);

  const uint K = uint(input_sizes.x);
  const uint K4 = (K + 3u) / 4u;
  const uint N4 = (uint(output_sizes.x) + 3u) / 4u;

#ifdef WEIGHT_INT4
  const uint CHUNKS_PER_GROUP = uint(K4_per_group) * 4u / WG_TILE_K;
  const uint num_chunks = uint(num_groups_arg) * CHUNKS_PER_GROUP;
#else
  const uint num_chunks = uint(num_groups_arg);
#endif

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  // Initialize fp32 accumulators to zero.
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
    }
  }

  const uint a_col = gl_LocalInvocationID.x % INVS_PER_ROW_A;
  const uint a_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_A;
  const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
  const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;

#ifdef WEIGHT_INT4
  // INT4 weight block grid (see pack_q4_linear_weight.glsl): block (k4, n8)
  // covers K=[k4*4, k4*4+3] x N=[n8*8, n8*8+7]; buffer pitch = K4 blocks per
  // n8 row, texture coord = ivec2(x=k4, y=n8). This thread's 8 N-values at
  // any K-row live in column n8_blk of the block grid:
  const uint n8_blk = (tile_n_start + b_col * 8u) >> 3u;

  // The K row within a block depends only on (b_row_offset & 3): chunkK and
  // the pass offset are both multiples of 4.
  const uint col_lo = 2u * (b_row_offset & 3u);
  const uint col_hi = col_lo + 1u;

  // Per-thread per-group weight scales (8 consecutive N), kept in registers
  // and reloaded only when the prefetched chunk crosses a group boundary.
  const uint sc_n4 = (tile_n_start + b_col * 8u) >> 2u;
  uint cached_group = 0xFFFFFFFFu;
  f16vec4 sc0;
  f16vec4 sc1;

  // Temp registers holding the prefetched (next) tile.
  uvec4 temp_A[A_PASSES];
  ivec4 temp_B[B_PASSES]; // raw packed INT4 blocks; dequant at the store stage
#else
  // INT8 weight block layout: t_packed_weight[k4 * N4 + n4] = ivec4 whose
  // component n_in_blk packs 4 K-bytes (K of block k4) for N-col
  // (n4*4 + n_in_blk). This thread's 8 N-values span two adjacent n4 blocks:
  const uint n4_a = (tile_n_start + b_col * 8u) >> 2u; // n_start mult of 8 -> even

  // The byte within a packed uint depends only on (b_row_offset & 3): chunkK
  // and the pass offset are both multiples of 4.
  const int b_shift = int(8u * (b_row_offset & 3u));

  // Per-thread per-channel weight scales (8 consecutive N), cached ONCE.
  f16vec4 sc0 = t_weight_scales[n4_a];
  f16vec4 sc1 = t_weight_scales[n4_a + 1u];

  // Temp registers holding the prefetched (next) tile.
  uvec4 temp_A[A_PASSES];
  ivec4 temp_Ba[B_PASSES]; // raw packed INT8 blocks; dequant at the store stage
  ivec4 temp_Bb[B_PASSES];
#endif

  // =========================================================
  // PROLOGUE: load chunk 0 from global memory DIRECTLY into slice 0,
  // then barrier (the first loop iteration reads slice 0).
  // =========================================================
  {
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
      const uint row = tile_m_start + p * A_ROWS_PER_PASS + a_row_offset;
      const uint k_hv4 = (a_col * FP16_PER_VEC4) / 4u;
      f16vec4 v0 = t_input[row * K4 + k_hv4];
      f16vec4 v1 = t_input[row * K4 + k_hv4 + 1u];
      Ash[(p * A_ROWS_PER_PASS + a_row_offset) * A_STRIDE_VEC4 + a_col] = uvec4(
          packFloat2x16(v0.xy), packFloat2x16(v0.zw),
          packFloat2x16(v1.xy), packFloat2x16(v1.zw));
    }
#ifdef WEIGHT_INT4
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
      Bsh[(p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
          dequant_block(wblock, col_lo, col_hi, sc0, sc1);
    }
#else
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
      const uint k4 = (p * B_ROWS_PER_PASS + b_row_offset) >> 2u;
      ivec4 wa;
      ivec4 wb;
#ifdef WEIGHT_BUFFER
      wa = t_packed_weight[k4 * N4 + n4_a];
      wb = t_packed_weight[k4 * N4 + n4_a + 1u];
#else
      wa = texelFetch(t_packed_weight, ivec2(n4_a, k4), 0);
      wb = texelFetch(t_packed_weight, ivec2(n4_a + 1u, k4), 0);
#endif
      Bsh[(p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
          dequant_block(wa, wb, b_shift, sc0, sc1);
    }
#endif
    barrier();
  }

  // =========================================================
  // MAIN LOOP — flattened, conditionals on `last`. Iteration `chunk` does:
  //   1. prefetch     — chunk+1 from global into temp (skipped when last)
  //   2. MMA math     — on slice (chunk%2)
  //   3. store        — temp into slice ((chunk+1)%2), then barrier
  //                     (both skipped when last)
  // =========================================================
  for (uint chunk = 0; chunk < num_chunks; ++chunk) {
    const bool last = (chunk + 1u >= num_chunks);
    const uint cur_base_A = (chunk % 2u) * ASH_SLICE;
    const uint cur_base_B = (chunk % 2u) * BSH_SLICE;
    const uint nxt_base_A = ((chunk + 1u) % 2u) * ASH_SLICE;
    const uint nxt_base_B = ((chunk + 1u) % 2u) * BSH_SLICE;

    // --- prefetch chunk+1 -> temp ---
    if (!last) {
      const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;

      [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
        const uint row = tile_m_start + p * A_ROWS_PER_PASS + a_row_offset;
        const uint k_hv4 = (chunkK_nxt + a_col * FP16_PER_VEC4) / 4u;
        f16vec4 v0 = t_input[row * K4 + k_hv4];
        f16vec4 v1 = t_input[row * K4 + k_hv4 + 1u];
        temp_A[p] = uvec4(
            packFloat2x16(v0.xy), packFloat2x16(v0.zw),
            packFloat2x16(v1.xy), packFloat2x16(v1.zw));
      }
#ifdef WEIGHT_INT4
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
#else
      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
        const uint k4 = (chunkK_nxt + p * B_ROWS_PER_PASS + b_row_offset) >> 2u;
#ifdef WEIGHT_BUFFER
        temp_Ba[p] = t_packed_weight[k4 * N4 + n4_a];
        temp_Bb[p] = t_packed_weight[k4 * N4 + n4_a + 1u];
#else
        temp_Ba[p] = texelFetch(t_packed_weight, ivec2(n4_a, k4), 0);
        temp_Bb[p] = texelFetch(t_packed_weight, ivec2(n4_a + 1u, k4), 0);
#endif
      }
#endif
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

    // --- store temp (chunk+1) -> nxt slice, dequantizing B, then barrier ---
    if (!last) {
      [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
        Ash[nxt_base_A + (p * A_ROWS_PER_PASS + a_row_offset) * A_STRIDE_VEC4 + a_col] =
            temp_A[p];
      }
      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
#ifdef WEIGHT_INT4
        Bsh[nxt_base_B + (p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
            dequant_block(temp_B[p], col_lo, col_hi, sc0, sc1);
#else
        Bsh[nxt_base_B + (p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
            dequant_block(temp_Ba[p], temp_Bb[p], b_shift, sc0, sc1);
#endif
      }
      barrier();
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
}
