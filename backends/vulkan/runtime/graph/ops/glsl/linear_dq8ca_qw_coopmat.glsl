/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * specs/025/026/027: loop structure updated from "dbuf4" (store-first,
 * single-buffered-until-prefetch) to "dbuf2" (store-first, prefetch-first
 * prologue) per specs/025 User Story 1's re-confirmed loop-structure winner
 * for this shader; tile geometry updated from the prior 128x64/K32/2x2/s64
 * to 64x32/K32/1x2/s64 per specs/027's e2e-ranked sweep winner
 * (tsweep_t64x32k32g12s64) -- confirmed +9.32% real end-to-end prefill
 * throughput on M5 EVT1 (Llama 3.1 8B, 2048-token prefill), not just
 * isolated-kernel GFLOP/s. See specs/027-e2e-tile-sweep/results/sweep-report.md.
 *
 * KHR Cooperative Matrix variant of the dynamically-quantized-activation
 * linear tiled shader (WEIGHT_NBITS=4):
 *   4  ->  linear_dq8ca_q4gsw_coopmat   INT4 group-symmetric weight
 *
 * Performs: out[M,N] = dequant(int8_act) * dequant(int_w) (+ bias)
 * via coopmat<int8> x coopmat<int8> -> coopmat<int32> on the matrix unit.
 *
 * Math (per group; per-channel INT8 is the num_groups == 1 special case
 * where the single "group" spans all of K):
 *   accum_int32 = sum_k(int8_in_k * int_w_signed_k)       // coopMatMulAdd
 *   result_fp  += float(accum_int32) * weight_scale[g, n] // group epilog
 * with everything row- or K-invariant hoisted OUT of the group loop
 * (specs/036 epilog restructure; algebraically identical to the previous
 * per-group form):
 *   out[m,n] = input_scale[m] * result_fp[m,n]
 *            - input_zp[m] * input_scale[m] * C[n]        (+ bias)
 *   C[n]     = sum_g(weight_sums[g,n] * weight_scale[g,n])
 * The zp*wsum correction is bilinear rank-1 over the whole K reduction, so
 * it needs no per-group matrix work: C[n] is accumulated as one scalar FMA
 * per group on the same thread that already prefetches that group's
 * wsum/wsc, and applied once in the output epilog together with the s_in
 * row scale. The group epilog is thereby reduced from {izp/wsum broadcasts,
 * int32 subtract, s_in*s_w outer product, convert, FMA} to {convert, one
 * column-broadcast FMA}, and no broadcast coopmats stay live across the
 * main loop (register-pressure relief for the double-accumulator layout).
 *
 * Because INT4 weights are sign-extended to int8 in the B-stage, the
 * "8 * input_sum" term of the tiled correction (which compensates for
 * unsigned int4 nibbles in dotPacked4x8) cancels out and is not needed.
 *
 * Loop structure ("dbuf2", specs/023-8da4w-int8-dbuf-sweep naming): prologue
 * prefetches chunk 0 into temp registers only (no shared-memory write, no
 * barrier); each loop iteration then does store(temp -> cur slice)
 * -> barrier() [UNCONDITIONAL, every iteration] -> MMA(cur) -> prefetch(next
 * chunk -> temp) [skipped on the last chunk]. Iteration `chunk` stores the
 * data FOR ITSELF (already prefetched by the previous iteration, or by the
 * prologue for chunk 0), immediately before using it. The same inversion
 * applies to the group wsum/wsc ping-pong: this variant stores the CURRENT
 * group's values (prefetched by the previous group's last chunk) at the head
 * of the group's first chunk. Group 0's wsum/wsc are unaffected -- set up
 * directly in the prologue. The nested groups x chunks loop and
 * unconditional group epilog are kept exactly as before -- flattening them
 * with a conditional coopmat epilog crashes the Xclipse PAL compiler at
 * large spec-resolved trip counts (specs/023 finding).
 *
 * Per-(group, N) weight scales live in a SECOND ping-pong pair indexed by
 * group parity: the next group's values are prefetched into registers and
 * stored to the other wsc slice during the iteration that crosses the group
 * boundary, and the regular per-iteration barrier makes them visible before
 * that group's epilog runs. weight_sums ride the same prefetch but never
 * touch LDS: they only feed the per-thread scalar C[n] accumulation (see
 * the hoisted math above). Per-row activation zp/scale land in LDS once in
 * the prologue and are only broadcast in the output epilog.
 *
 * LDS layout for the MMA operands: K-slab split + ColumnMajor B + per-col
 * skew padding: the int8 WMMA matB lane layout wants 4 K-contiguous bytes
 * per lane, so a RowMajor B in LDS forces per-byte ds_load + v_perm repack
 * chains. ColumnMajor with a +1-uint skew per column gives one ds_load_b32
 * per lane with a bank-conflict-free col stride. Each uint holds 4 packed
 * int8.
 *
 * Tile hierarchy (yaml): MMA 16x16x16 int8, WG_TILE 128x64, WG_TILE_K = 32,
 * 8 subgroups x 32 threads (4x2 grid, WG_SIZE 256) -- specs/038's g128 sweep
 * winner on 780M/RADV. The 4x2 grid puts SG_TILE at 64x16 (MMAS_PER_SG 4x1):
 * one matB load reused across four matA, and only one N-tile of the double
 * accumulator per subgroup, which halves VGPR vs a 2x2 grid (256 -> 128) and
 * doubles occupancy (4 -> 8 waves/SIMD). SUBGROUP_SIZE 32 gates 44/44 correct
 * at this shape on RADV/ACO: specs/026's subgroup=32 ban was Xclipse-specific
 * (that compiler miscompiled sg32 tile-shape-dependently); it does not apply
 * to this device. See specs/038-dq8ca-g128-tile-sweep for the sweep guardrails.
 *
 * Hard preconditions:
 *   M % WG_TILE_M == 0, N % WG_TILE_N == 0, K % WG_TILE_K == 0,
 *   INT4: group_size % WG_TILE_K == 0,
 *   device exposes coopmat<int8>x<int8>-><int32> at 16x16x16.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if WEIGHT_NBITS == 4:
  #define WEIGHT_INT4

$if HAS_BIAS:
  #define HAS_BIAS

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

$if IO_STORAGE == "texture3d":
  #define IO_TEXTURE

layout(std430) buffer;

#include "common.glslh"

// Bindings — match add_linear_dqa_qw_node arg order:
//   output(0), fp_input(1), packed_int8_input(2), int_input_sums(3 - unused),
//   input_scales(4), input_zps(5), packed_weight(6), weight_sums(7),
//   weight_scales(8), bias(9).
${layout_declare_tensor(B, "w", "t_output",              "half", IO_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input",               "half", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input",   "int",  "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_sums",     "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_int8_input_scales",   "half", "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps",      "int8", "texture3d")}
${layout_declare_tensor(B, "r", "t_packed_weight",       "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums",         "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight_scales",       "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",                "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
// INT4 only; inert (0) for INT8 so the dispatcher's spec list lines up.
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
// Trip-count source for the coopmat K loop, passed as a spec constant (not
// derived from the runtime sizes UBO): the Xclipse/AMD-PAL shader compiler
// crashes (null deref in vkCreateComputePipelines) when a loop containing
// coopMatMulAdd has a UBO-derived trip count. INT4: number of quant groups;
// INT8: number of K-chunks.
//
// Unlike linear_qw_coopmat, this spec-const workaround is INTENTIONALLY kept
// here: on 2026-06-30 the UBO-direct method (sizes UBO feeding num_chunks/N
// directly) was A/B'd on this shader and produced wrong results for the
// coopmat (buffer) path at M>=128, while this spec-const version validated
// clean — see add_linear_dqa_qw_node in QuantizedLinear.cpp.
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
// Output width N for coopMatStore: the Xclipse compiler MISCOMPILES
// coopMatStore whose offset/stride derive from a UBO value (only the first
// store per subgroup lands correctly; standalone repro cm_acc2).
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

// Tile geometry
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

const uint A_SLAB_INT8     = WG_TILE_M * MMA_K;
const uint B_USEFUL_U32    = MMA_K / 4u;
const uint B_STRIDE_U32    = B_USEFUL_U32 + 1u; // +1 skew
const uint B_SLAB_U32      = WG_TILE_N * B_STRIDE_U32;
const uint NUM_K_SLABS     = WG_TILE_K / MMA_K;

const uint A_SLAB_U32      = A_SLAB_INT8 / 4u;
const uint A_STRIDE_U32    = MMA_K / 4u;

// One ping-pong slice covers all K-slabs of one chunk.
const uint ASH_SLICE_U32 = NUM_K_SLABS * A_SLAB_U32;
const uint BSH_SLICE_U32 = NUM_K_SLABS * B_SLAB_U32;

// Double-buffered MMA operand staging.
shared uint Ash_int8[2u * ASH_SLICE_U32];
shared uint Bsh_int8[2u * BSH_SLICE_U32];

// Per-WG-tile-row activation params (loaded ONCE at WG start; only
// broadcast in the output epilog).
shared float ifs_sh[WG_TILE_M];   // input_scale[m]
shared float zpfs_sh[WG_TILE_M];  // input_zp[m] * input_scale[m]

// Per-(group, output-channel) weight scales, ping-ponged by
// group % (2 * GROUPS_PER_CHUNK) -- the same double-buffer discipline as
// Ash/Bsh, so a chunk's step-1 stores never race the previous chunk's
// epilog reads. Sized for the worst case (one group per K-slab); at
// CHUNKS_PER_GROUP > 1 or INT8 only the first two slices are used.
shared float wsc_sh[2u * NUM_K_SLABS * WG_TILE_N];

// Rank-1 zp-correction constant C[n] (see header math), written by each
// column's owning thread after the group loop.
shared float C_sh[WG_TILE_N];

#ifdef HAS_BIAS
shared float bias_sh[WG_TILE_N];
#endif

#ifdef IO_TEXTURE
// Result staging for the imageStore epilogue, mirroring linear_qw_coopmat:
// SG_GRID_Y bands of MMA_M rows, WG_TILE_N wide, row-major. Drained one
// accumulator row-block at a time so this costs MMA_M*SG_GRID_Y rows of LDS
// rather than the full WG_TILE_M x WG_TILE_N tile. Distinct from C_sh, which
// holds the per-output-channel weight sums.
const uint CSH_ROWS = SG_GRID_Y * MMA_M;
shared float16_t Csh_out[CSH_ROWS * WG_TILE_N];
#endif

// Running accumulator across groups, held in fp16: RADV shader stats showed
// the fp32 version at 256 VGPRs with 51-65 spilled (the real cost of the
// double-accumulator layout -- scratch traffic in the MMA loop); halving
// this array's footprint targets that directly. Precision: each element
// accumulates one already-scaled (small-magnitude) partial per group, not
// per-K-element -- the same regime the 4w shader's fp16 accumulator already
// validates at production K. The per-group product itself is computed in
// fp32 (int32 -> float -> *wsc) before narrowing.
coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    result[MMAS_PER_SG_M][MMAS_PER_SG_N];

// Per-group int32 MMA accumulator.
coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    accum_int32[MMAS_PER_SG_M][MMAS_PER_SG_N];

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % SG_GRID_X,
      gl_SubgroupID / SG_GRID_X);

  const uint K = uint(input_sizes.x);
  const uint N = uint(output_sizes.x);
  const uint N4 = (N + 3u) / 4u;
  const uint nblocks_x_A = (K + 3u) >> 2u;

#ifdef WEIGHT_INT4
  const uint num_groups = uint(num_groups_arg);
  // A chunk may cover SEVERAL whole quant groups (WG_TILE_K >= group K-span,
  // the fixed-cost-amortizing case: one barrier serves GROUPS_PER_CHUNK
  // groups' worth of MMA), or a group may span several chunks (the previous
  // structure). The dispatch gate guarantees one of the two divides.
  const uint GROUP_K = uint(K4_per_group) * 4u;
  const uint GROUPS_PER_CHUNK =
      WG_TILE_K >= GROUP_K ? WG_TILE_K / GROUP_K : 1u;
  const uint CHUNKS_PER_GROUP =
      WG_TILE_K >= GROUP_K ? 1u : GROUP_K / WG_TILE_K;
#else
  // Per-channel: a single quant "group" spanning all of K; the epilog runs
  // exactly once, on the last chunk.
  const uint num_groups = 1u;
  const uint GROUPS_PER_CHUNK = 1u;
  const uint CHUNKS_PER_GROUP = uint(num_groups_arg);
#endif
  const uint num_chunks = num_groups * CHUNKS_PER_GROUP / GROUPS_PER_CHUNK;
  const uint SLABS_PER_SEG = NUM_K_SLABS / GROUPS_PER_CHUNK;
  const uint WSC_SLOTS = 2u * GROUPS_PER_CHUNK;

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
      accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
    }
  }

  // --- A staging thread map: (m4, k4) ivec4 blocks, striped over threads
  // (multiple blocks per thread once WG_TILE_K grows past the one-block-
  // per-thread point). Each block covers 4 M-rows x 4 K-positions and
  // expands to 4 slab-major LDS uints. ---
  const uint K_BLOCKS_PER_CHUNK = WG_TILE_K >> 2u;
  const uint A_TOTAL_BLOCKS = (WG_TILE_M >> 2u) * K_BLOCKS_PER_CHUNK;
  const uint A_BLOCKS_PER_THREAD = (A_TOTAL_BLOCKS + WG_SIZE - 1u) / WG_SIZE;

#ifdef WEIGHT_INT4
  // --- B staging thread map: (block, col) slots; each slot extracts one
  //     ColumnMajor LDS uint (4 K-contiguous sign-extended int8) ---
  // INT4 weight block grid (see pack_q4_linear_weight.glsl): block (k4, n8)
  // covers K=[k4*4, k4*4+3] x N=[n8*8, n8*8+7]. Within a block, int32[r]
  // nibble col c maps to N = n8*8 + r + (c&1 ? 4 : 0), K = k4*4 + c/2 — one
  // (component, parity) pair yields exactly the 4 K-contiguous bytes of one
  // N column = one ColumnMajor LDS uint.
  const uint B_TOTAL_SLOTS = K_BLOCKS_PER_CHUNK * WG_TILE_N;
  const uint B_SLOTS_PER_THREAD = B_TOTAL_SLOTS / WG_SIZE;
  const uint N8_PER_TILE = WG_TILE_N >> 3u;
#else
  // --- B staging thread map: one (k4, n4) ivec4 block per active thread ---
  // INT8 weight block layout: wblk[n_in_blk] packs 4 K-contiguous bytes for
  // N-col (n4*4 + n_in_blk) — exactly one ColumnMajor LDS uint, written
  // as-is (no byte repack).
  const uint B_FETCH_SLOTS = K_BLOCKS_PER_CHUNK * (WG_TILE_N >> 2u);
  const uint N4_PER_TILE = WG_TILE_N >> 2u;
  const uint b_k4_in_chunk = gl_LocalInvocationID.x / N4_PER_TILE;
  const uint b_n_uint_col = gl_LocalInvocationID.x % N4_PER_TILE;
  const bool b_active = gl_LocalInvocationID.x < B_FETCH_SLOTS;
#endif

  // Prefetch temp registers.
  ivec4 temp_A[A_BLOCKS_PER_THREAD];
#ifdef WEIGHT_INT4
  ivec4 temp_B[B_SLOTS_PER_THREAD];
  int   temp_wsum[NUM_K_SLABS]; // [GROUPS_PER_CHUNK] used
  float temp_wsc[NUM_K_SLABS];
#else
  ivec4 temp_B;
#endif

  // =========================================================
  // PROLOGUE
  // =========================================================
  // One-time: per-row input zp + scale (texture3d, one m4-block of 4 rows per
  // texel) — constant across K groups, only used in the output epilog.
  if (gl_LocalInvocationID.x < (WG_TILE_M >> 2u)) {
    const uint m4 = (tile_m_start >> 2u) + gl_LocalInvocationID.x;
    const vec4  sc = vec4(texelFetch(t_int8_input_scales, ivec3(m4, 0, 0), 0));
    const ivec4 zp = texelFetch(t_int8_input_zps,         ivec3(m4, 0, 0), 0);
    const uint base = gl_LocalInvocationID.x * 4u;
    ifs_sh[base + 0u] = sc.x;  ifs_sh[base + 1u] = sc.y;
    ifs_sh[base + 2u] = sc.z;  ifs_sh[base + 3u] = sc.w;
    zpfs_sh[base + 0u] = float(zp.x) * sc.x;
    zpfs_sh[base + 1u] = float(zp.y) * sc.y;
    zpfs_sh[base + 2u] = float(zp.z) * sc.z;
    zpfs_sh[base + 3u] = float(zp.w) * sc.w;
  }
  // Chunk 0's group(s): weight scale(s) -> their wsc slots; their C[n]
  // contributions -> C_reg.
  float C_reg = 0.0;
  if (gl_LocalInvocationID.x < WG_TILE_N) {
    const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
    for (uint g = 0; g < GROUPS_PER_CHUNK; ++g) {
      f16vec4 sv = t_weight_scales[g * N4 + (n_idx >> 2u)];
      const float w = float(sv[n_idx & 3u]);
      wsc_sh[g * WG_TILE_N + gl_LocalInvocationID.x] = w;
      C_reg += float(t_weight_sums[g * N + n_idx]) * w;
    }
  }
  memoryBarrierShared();
  barrier();

  // dbuf2: prefetch chunk 0 into temp registers only -- no shared-memory
  // write, no barrier here. The main loop's first iteration stores temp
  // into slice 0 and barriers as normal (uniform code path for every chunk,
  // including chunk 0).
  [[unroll]] for (uint ai = 0; ai < A_BLOCKS_PER_THREAD; ++ai) {
    const uint blk = gl_LocalInvocationID.x + ai * WG_SIZE;
    if (blk < A_TOTAL_BLOCKS) {
      const uint m4_global = (tile_m_start >> 2u) + blk / K_BLOCKS_PER_CHUNK;
      temp_A[ai] =
          t_packed_int8_input[m4_global * nblocks_x_A + blk % K_BLOCKS_PER_CHUNK];
    }
  }
#ifdef WEIGHT_INT4
  [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
    const uint slot = gl_LocalInvocationID.x + si * WG_SIZE;
    const uint block_in_chunk = slot >> 3u;
    const uint k4_blk = block_in_chunk / N8_PER_TILE;
    const uint n8_blk = (tile_n_start >> 3u) + (block_in_chunk % N8_PER_TILE);
#ifdef WEIGHT_BUFFER
    temp_B[si] = t_packed_weight[(n8_blk * nblocks_x_A) + k4_blk];
#else
    temp_B[si] = texelFetch(t_packed_weight, ivec2(k4_blk, n8_blk), 0);
#endif
  }
#else
  if (b_active) {
    const uint block_x_w = (tile_n_start >> 2u) + b_n_uint_col;
#ifdef WEIGHT_BUFFER
    temp_B = t_packed_weight[(b_k4_in_chunk * N4) + block_x_w];
#else
    temp_B = texelFetch(t_packed_weight, ivec2(block_x_w, b_k4_in_chunk), 0);
#endif
  }
#endif

  // =========================================================
  // MAIN LOOP — FLAT over chunks (dbuf2 "store-first" ordering, one
  // UNCONDITIONAL barrier per chunk). When WG_TILE_K spans several quant
  // groups (GROUPS_PER_CHUNK > 1), one barrier amortizes over all of them:
  // step 3 runs the MMA one group-segment at a time with the group epilog
  // at each segment tail. specs/023's nested groups x chunks structure
  // (and its ban on a conditional coopmat epilog) worked around an Xclipse
  // PAL compiler crash; this 780M-only branch compiles with RADV/ACO,
  // where the flat form is fine and the nesting only cost registers.
  //   1. store     — temp (prologue for chunk 0, else the previous
  //                  iteration's step 4) -> A/B slice (chunk%2), unpacking
  //                  the weight; on each group-opening chunk also that
  //                  chunk's group wsc slot(s) + C[n] scalar FMA.
  //   2. barrier   — cur slices fully written; never skipped.
  //   3. int8 MMA  — per group-segment, epilog at owning segment's tail.
  //   4. prefetch  — chunk+1 A/B blocks (+ its groups' wsum/wsc when it
  //                  opens them). Skipped entirely on the final chunk.
  // =========================================================
  for (uint chunk = 0; chunk < num_chunks; ++chunk) {
    const uint first_group = (chunk * GROUPS_PER_CHUNK) / CHUNKS_PER_GROUP;
    {
      const bool has_next = chunk + 1u < num_chunks;
      const uint cur_a = (chunk % 2u) * ASH_SLICE_U32;
      const uint cur_b = (chunk % 2u) * BSH_SLICE_U32;

      // --- 1. store temp (this chunk) -> cur slice ---
      [[unroll]] for (uint ai = 0; ai < A_BLOCKS_PER_THREAD; ++ai) {
        const uint blk = gl_LocalInvocationID.x + ai * WG_SIZE;
        if (blk < A_TOTAL_BLOCKS) {
          const uint a_k_block = blk % K_BLOCKS_PER_CHUNK;
          const uint slab_idx       = a_k_block / (MMA_K >> 2u);
          const uint k_uint_in_slab = a_k_block % (MMA_K >> 2u);
          const uint base_row = (blk / K_BLOCKS_PER_CHUNK) * 4u;
          [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
            Ash_int8[cur_a + slab_idx * A_SLAB_U32 + (base_row + m4i) * A_STRIDE_U32 + k_uint_in_slab] =
                uint(temp_A[ai][m4i]);
          }
        }
      }
#ifdef WEIGHT_INT4
      [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
        const uint slot = gl_LocalInvocationID.x + si * WG_SIZE;
        const uint block_in_chunk = slot >> 3u;
        const uint col_in_block   = slot & 7u;
        const uint k4_in_chunk    = block_in_chunk / N8_PER_TILE;
        const uint n8_in_tile     = block_in_chunk % N8_PER_TILE;
        const uint r      = col_in_block & 3u;
        const uint parity = col_in_block >> 2u;
        const int  w      = temp_B[si][r];
        const int  base   = int(4u * parity);
        const int v0 = (((w >> (base + 0))  & 0xF) - 8) & 0xFF;
        const int v1 = (((w >> (base + 8))  & 0xF) - 8) & 0xFF;
        const int v2 = (((w >> (base + 16)) & 0xF) - 8) & 0xFF;
        const int v3 = (((w >> (base + 24)) & 0xF) - 8) & 0xFF;
        const uint n_col      = n8_in_tile * 8u + r + parity * 4u;
        const uint slab_idx   = k4_in_chunk / (MMA_K >> 2u);
        const uint k4_in_slab = k4_in_chunk % (MMA_K >> 2u);
        Bsh_int8[cur_b + slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab] =
            uint(v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
      }
      // Group-opening chunk (beyond prologue-covered chunk 0): store the
      // prefetched wsc slot(s) and fold wsum into the per-thread C[n]
      // scalar (never touches LDS). Slots cycle group % WSC_SLOTS, so this
      // never overwrites a slot the previous chunk's epilog still reads.
      if (chunk > 0u && (chunk % CHUNKS_PER_GROUP) == 0u &&
          gl_LocalInvocationID.x < WG_TILE_N) {
        [[unroll]] for (uint g = 0; g < GROUPS_PER_CHUNK; ++g) {
          wsc_sh
              [((first_group + g) % WSC_SLOTS) * WG_TILE_N +
               gl_LocalInvocationID.x] = temp_wsc[g];
          C_reg += float(temp_wsum[g]) * temp_wsc[g];
        }
      }
#else
      if (b_active) {
        const uint slab_idx   = b_k4_in_chunk / (MMA_K >> 2u);
        const uint k4_in_slab = b_k4_in_chunk % (MMA_K >> 2u);
        const uint n_col_base = b_n_uint_col * 4u;
        [[unroll]] for (uint n_in_blk = 0u; n_in_blk < 4u; ++n_in_blk) {
          Bsh_int8[cur_b + slab_idx * B_SLAB_U32 + (n_col_base + n_in_blk) * B_STRIDE_U32 + k4_in_slab] =
              uint(temp_B[n_in_blk]);
        }
      }
#endif

      // --- 2. barrier — cur slice(s) fully written ---
      memoryBarrierShared();
      barrier();

      // --- 3. int8 MMA, one group-segment at a time; the group epilog
      // (result += float(accum) * s_w, reset accum -- zp/wsum and s_in are
      // hoisted to the output epilog, see header math) runs at a segment
      // tail whenever that segment closes its group. ---
      [[unroll]] for (uint seg = 0; seg < GROUPS_PER_CHUNK; ++seg) {
        [[unroll]] for (uint k = seg * SLABS_PER_SEG;
                        k < (seg + 1u) * SLABS_PER_SEG;
                        ++k) {
          const uint slab_a_base_u32 = cur_a + k * A_SLAB_U32;
          const uint slab_b_base_u32 = cur_b + k * B_SLAB_U32;

          coopmat<int8_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
          [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
            const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
            coopMatLoad(
                matA[i], Ash_int8,
                slab_a_base_u32 + row_a * A_STRIDE_U32,
                A_STRIDE_U32,
                gl_CooperativeMatrixLayoutRowMajor);
          }

          coopmat<int8_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
          [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
            const uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
            coopMatLoad(
                matB, Bsh_int8,
                slab_b_base_u32 + col_b * B_STRIDE_U32,
                B_STRIDE_U32,
                gl_CooperativeMatrixLayoutColumnMajor);
            [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
              accum_int32[i][j] = coopMatMulAdd(matA[i], matB, accum_int32[i][j]);
            }
          }
        }

        if ((chunk + 1u) % CHUNKS_PER_GROUP == 0u) {
          const uint wbase = ((first_group + seg) % WSC_SLOTS) * WG_TILE_N;
          [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
            const uint local_n_base = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

            coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsc_bcast;
            coopMatLoad(
                wsc_bcast, wsc_sh,
                wbase + local_n_base, /*stride=*/0u,
                gl_CooperativeMatrixLayoutRowMajor);

            [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
              result[i][j] +=
                  coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(
                      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(
                          accum_int32[i][j]) *
                      wsc_bcast);
              accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
            }
          }
        }
      }

      // --- 4. prefetch chunk+1 -> temp ---
      if (has_next) {
        const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;
        [[unroll]] for (uint ai = 0; ai < A_BLOCKS_PER_THREAD; ++ai) {
          const uint blk = gl_LocalInvocationID.x + ai * WG_SIZE;
          if (blk < A_TOTAL_BLOCKS) {
            const uint m4_global =
                (tile_m_start >> 2u) + blk / K_BLOCKS_PER_CHUNK;
            const uint k4_global =
                (chunkK_nxt >> 2u) + blk % K_BLOCKS_PER_CHUNK;
            temp_A[ai] = t_packed_int8_input[m4_global * nblocks_x_A + k4_global];
          }
        }
#ifdef WEIGHT_INT4
        [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
          const uint slot = gl_LocalInvocationID.x + si * WG_SIZE;
          const uint block_in_chunk = slot >> 3u;
          const uint k4_blk = (chunkK_nxt >> 2u) + block_in_chunk / N8_PER_TILE;
          const uint n8_blk = (tile_n_start >> 3u) + (block_in_chunk % N8_PER_TILE);
#ifdef WEIGHT_BUFFER
          temp_B[si] = t_packed_weight[(n8_blk * nblocks_x_A) + k4_blk];
#else
          temp_B[si] = texelFetch(t_packed_weight, ivec2(k4_blk, n8_blk), 0);
#endif
        }
        // chunk+1 opens new group(s): prefetch their wsc/wsum.
        if ((chunk + 1u) % CHUNKS_PER_GROUP == 0u &&
            gl_LocalInvocationID.x < WG_TILE_N) {
          const uint next_fg =
              ((chunk + 1u) * GROUPS_PER_CHUNK) / CHUNKS_PER_GROUP;
          const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
          [[unroll]] for (uint g = 0; g < GROUPS_PER_CHUNK; ++g) {
            f16vec4 sv = t_weight_scales[(next_fg + g) * N4 + (n_idx >> 2u)];
            temp_wsc[g] = float(sv[n_idx & 3u]);
            temp_wsum[g] = t_weight_sums[(next_fg + g) * N + n_idx];
          }
        }
#else
        if (b_active) {
          const uint block_y_w = (chunkK_nxt >> 2u) + b_k4_in_chunk;
          const uint block_x_w = (tile_n_start >> 2u) + b_n_uint_col;
#ifdef WEIGHT_BUFFER
          temp_B = t_packed_weight[(block_y_w * N4) + block_x_w];
#else
          temp_B = texelFetch(t_packed_weight, ivec2(block_x_w, block_y_w), 0);
#endif
        }
#endif
      }
    }
  }  // chunks

  // --- C[n] + optional bias -> LDS for the output-epilog broadcasts ---
  if (gl_LocalInvocationID.x < WG_TILE_N) {
    C_sh[gl_LocalInvocationID.x] = C_reg;
  }
#ifdef HAS_BIAS
  if (apply_bias > 0) {
    for (uint t = gl_LocalInvocationID.x; t < WG_TILE_N; t += WG_SIZE) {
      bias_sh[t] = float(t_bias[tile_n_start + t]);
    }
  }
#endif
  memoryBarrierShared();
  barrier();

  // --- Output epilog + store: out = s_in[m]*result - zp[m]*s_in[m]*C[n]
  // (+ bias). N for the store address math MUST come from the spec
  // constant, not the sizes UBO (see out_N_arg above). ---
#ifdef IO_TEXTURE
  // Epilogue iteration i drains accumulator row-block i from EVERY subgroup
  // into Csh_out at once, so the SG_GRID_Y bands it holds are disjoint global
  // row ranges; the whole workgroup then imageStores them.
  //
  // PORTABILITY NOTE (carried from linear_qw_coopmat): [[unroll]] is only a
  // hint and glslc does NOT honor it once the loop body contains a barrier(),
  // so result[i][j] IS dynamically indexed here. RADV/ACO accepts it; coopmat
  // arrays are opaque per-lane storage and dynamic indexing is the kind of
  // construct the Xclipse/AMD-PAL compiler has broken before -- check this
  // first if the texture variants ever miscompile on another driver.
  const uint CSH_TEXELS_PER_ROW = WG_TILE_N / 4u;
  const uint CSH_TEXELS = CSH_ROWS * CSH_TEXELS_PER_ROW;
#else
  const uint N_out = uint(out_N_arg);
#endif
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
#ifdef IO_TEXTURE
    // Guards Csh_out against the previous iteration's readers. Inert on
    // i == 0, but must stay unconditional to remain workgroup-uniform.
    barrier();
#endif
    const uint local_m_base = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
    coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> ifs_bcast;
    coopMatLoad(
        ifs_bcast, ifs_sh,
        local_m_base, /*stride=*/0u,
        gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> zpfs_bcast;
    coopMatLoad(
        zpfs_bcast, zpfs_sh,
        local_m_base, /*stride=*/0u,
        gl_CooperativeMatrixLayoutColumnMajor);

    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      const uint local_n = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> C_bcast;
      coopMatLoad(C_bcast, C_sh, local_n, 0u, gl_CooperativeMatrixLayoutRowMajor);
      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> acc =
          ifs_bcast *
              coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(
                  result[i][j]) -
          zpfs_bcast * C_bcast;

#ifdef HAS_BIAS
      if (apply_bias > 0) {
        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> bias_tile;
        coopMatLoad(bias_tile, bias_sh, local_n, 0u, gl_CooperativeMatrixLayoutRowMajor);
        acc += bias_tile;
      }
#endif

      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> out_tile =
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(acc);
#ifdef IO_TEXTURE
      coopMatStore(
          out_tile, Csh_out,
          warpInTile.y * MMA_M * WG_TILE_N + local_n,
          WG_TILE_N,
          gl_CooperativeMatrixLayoutRowMajor);
#else
      coopMatStore(
          out_tile, t_output,
          (tile_m_start + local_m_base) * N_out + (tile_n_start + local_n),
          N_out,
          gl_CooperativeMatrixLayoutRowMajor);
#endif
    }
#ifdef IO_TEXTURE
    memoryBarrierShared();
    barrier();

    for (uint t = gl_LocalInvocationID.x; t < CSH_TEXELS; t += WG_SIZE) {
      const uint lr = t / CSH_TEXELS_PER_ROW;
      const uint lc4 = t % CSH_TEXELS_PER_ROW;
      // lr / MMA_M is the band = the writing subgroup's warpInTile.y, so the
      // global row matches the buffer path's tile_m_start + local_m_base.
      const uint m =
          tile_m_start + (lr / MMA_M) * SG_TILE_M + i * MMA_M + (lr % MMA_M);
      const uint base = lr * WG_TILE_N + lc4 * 4u;
      imageStore(
          t_output,
          ivec3(tile_n_start / 4u + lc4, m, 0),
          vec4(
              float(Csh_out[base]),
              float(Csh_out[base + 1u]),
              float(Csh_out[base + 2u]),
              float(Csh_out[base + 3u])));
    }
#endif
  }
}
