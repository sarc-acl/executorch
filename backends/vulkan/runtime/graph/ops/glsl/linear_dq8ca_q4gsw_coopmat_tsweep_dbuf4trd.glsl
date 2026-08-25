/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * DIAGNOSTIC BISECT variant #2 of linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4tr.
 *
 * A does not go through LDS at all: each subgroup coopMatLoads the A tiles it
 * needs straight from the row-major global buffer inside the MMA loop. This
 * isolates coopMatLoad-from-StorageBuffer-int[] from
 * coopMatStore-into-Workgroup-uint[]: dbuf4trm (manual staging) already proved
 * the packer and layout correct, so if this variant passes the load is fine
 * and the store is the culprit, and if it fails the load is.
 *
 * Ash_int8 is left allocated but unused (keeps the diff minimal; costs LDS).
 *
 * Original dbuf4tr header follows.
 *
 * "-tr" (coopmat-staged A) variant of linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4.
 *
 * Ported from shmem_double_buf4-tr.comp on vk_cooperative_matrix_perf's
 * gemm-ubm branch. That reference file's delta over shmem_double_buf4.comp is
 * that the global -> LDS staging goes through COOPERATIVE MATRIX REGISTERS
 * (coopMatLoad from global -> coopmat<> array -> coopMatStore into shared)
 * instead of a hand-rolled per-thread uvec4 copy, and that B lands in LDS
 * column-major.
 *
 * Only the A half of that idea is portable to this kernel:
 *
 *   - B is ALREADY column-major in LDS here (Bsh_int8 is K-contiguous per
 *     output column, read back with gl_CooperativeMatrixLayoutColumnMajor),
 *     so the reference's "tr" property is not a delta for B at all.
 *   - B CANNOT be coopmat-staged: the weights are int4, each ivec4 holding
 *     8 columns x 4 K-values that need the nibble-extract / -8 / sign-pack
 *     below. coopMatLoad cannot unpack nibbles, and a coopmat's per-lane
 *     layout is opaque so one cannot be built from unpacked registers.
 *     B staging is therefore left byte-identical to dbuf4.
 *   - A CAN be coopmat-staged, but only against a ROW-MAJOR packed int8
 *     activation buffer. The stock 4h4w layout (kPackedInt8_4H4W, produced by
 *     quantize_and_pack_4h4w_with_group_sums.glsl) is NOT row-major: element
 *     [m4 * K4 + k4] is an ivec4 whose COMPONENT selects one of 4 rows, so as
 *     a uint array the index is m4*(4*K4) + k4*4 + r, which is not affine in
 *     the row index and cannot be addressed by any RowMajor/ColumnMajor
 *     coopMatLoad. (ColumnMajor is out on contiguity too: a uint packs 4
 *     K-values, not 4 M-values.)
 *
 * So this shader binds t_packed_int8_input as a SCALAR int array in the
 * kPackedInt8_4W layout -- plain row-major int8, 4 K-values per int32,
 * row stride K4 -- produced by quantize_and_pack_4w_with_group_sums.glsl.
 * QuantizedLinear.cpp allocates that layout (and dispatches that packer)
 * only when the active dq8ca variant is a "tsweep_dbuf4tr_t..." token AND
 * the coopmat gate passes, so the tiled fallback never sees the wrong
 * layout. Everything downstream of A staging -- LDS layout, int8 WMMA thread
 * maps, group epilog, bias/store epilogue -- is unchanged from dbuf4.
 *
 * A staging (the actual -tr port):
 *   dbuf4:   per-thread (m4, k4) ivec4 fetch; only A_ACTIVE_THREADS =
 *            (WG_TILE_M/4) * (WG_TILE_K/4) invocations participate, each
 *            scattering 4 rows into Ash_int8 with 4 scalar stores.
 *   dbuf4tr: per-SUBGROUP MMA_M x MMA_K tile fetch via coopMatLoad straight
 *            from global, then coopMatStore into the same Ash_int8 slot.
 *            The (WG_TILE_M/MMA_M) * (WG_TILE_K/MMA_K) tiles of a chunk are
 *            dealt round-robin across the NUM_SUBGROUPS subgroups.
 *
 * The loop structure is dbuf4's, unchanged:
 *   prologue: prefetch chunk 0 -> temp, store to slice 0 (no barrier)
 *   per iter: barrier -> prefetch(next) -> MMA(cur) -> store(next)
 * and the nested `groups x chunks` loop with an unconditional group epilog is
 * kept as-is (flattening it crashes the Xclipse PAL compiler at large
 * spec-resolved trip counts -- see dbuf2's header).
 *
 * Selected at dispatch via
 * ET_VK_DQ8CA_COOPMAT_VARIANT=tsweep_dbuf4tr_t<M>x<N>k<K>g<SGX><SGY>s<32|64>
 * (QuantizedLinear.cpp), additive to the tsweep_dbuf4_t... and tsweep_t...
 * namespaces. NOT the default -- unvalidated until it passes repeated
 * test_llama_microbench --correctness-only runs (see dq8ca_coopmat_variant()'s
 * comment on why a single pass is not proof).
 *
 * Performs: out[M,N] = dequant(int8_act) * dequant(int_w) (+ bias)
 * via coopmat<int8> x coopmat<int8> -> coopmat<int32> on the matrix unit.
 *
 * Hard preconditions (in addition to dbuf4's):
 *   M % WG_TILE_M == 0, N % WG_TILE_N == 0, K % WG_TILE_K == 0,
 *   group_size % WG_TILE_K == 0, K % 4 == 0,
 *   WG_TILE_M % MMA_M == 0, WG_TILE_K % MMA_K == 0,
 *   t_packed_int8_input in kPackedInt8_4W (row-major) layout,
 *   device exposes coopmat<int8>x<int8>-><int32> at 16x16x16.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
// 8-bit SSBO access: A is bound as a scalar int8_t array so that the
// coopMatLoad below has a MATCHING component type. Loading a
// coopmat<int8_t> from a 32-bit int[] SSBO is what broke the first
// attempt (see header).
#extension GL_EXT_shader_8bit_storage : require
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
// t_input is unread here -- the activations arrive already quantized in
// t_packed_int8_input -- but stays declared so the binding layout matches the
// dispatch site. It tracks IO_STORAGE so the two IO tensors stay consistent.
${layout_declare_tensor(B, "r", "t_input",               "half", IO_STORAGE, is_scalar_array=False)}
// ROW-MAJOR (kPackedInt8_4W) packed activations, bound as a scalar int8_t
// array (row stride = K int8). Two things differ from dbuf4, which takes the
// 4h4w ivec4 block layout:
//   1. row-major, so a coopMatLoad can address it at all;
//   2. element type int8_t, MATCHING the coopmat component type.
// (2) is not cosmetic. Binding the same memory as int[] and loading a
// coopmat<int8_t> from it -- a type mismatch that demonstrably works for the
// Workgroup storage class, which is how the MMA loop reads Ash_int8 below --
// silently produces wrong results from a StorageBuffer on this driver.
// The reference shmem_double_buf4-tr.comp sidesteps it the same way: its
// buffer_reference is declared `A_TYPE x[]`, i.e. int8_t for the int8 config.
// All A offsets/strides here are therefore in INT8 elements, not int.
${layout_declare_tensor(B, "r", "t_packed_int8_input",   "int8", "buffer", is_scalar_array=True)}
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
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
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

// Per-WG-tile-row activation params (loaded ONCE at WG start; constant
// across groups).
shared int   izp_sh[WG_TILE_M];   // int32 (cast from int8 source) for broadcast
shared float ifs_sh[WG_TILE_M];   // float32 (cast from fp16 source) for broadcast

// Per-(group, output-channel) weight params, ping-ponged by group parity.
// (For per-channel INT8 only slice 0 is ever used.)
shared int   wsum_sh[2u * WG_TILE_N];
shared float wsc_sh[2u * WG_TILE_N];

#ifdef HAS_BIAS
shared float bias_sh[WG_TILE_N];
#endif

#ifdef IO_TEXTURE
// Result staging for the imageStore epilogue, mirroring the fp16 kernel:
// SG_GRID_Y bands of MMA_M rows, each WG_TILE_N wide, row-major. A full
// WG_TILE_M x WG_TILE_N buffer would cost SG_GRID_Y/MMAS_PER_SG_M x more LDS
// and wreck occupancy. float16_t-typed because coopMatStore needs it.
const uint CSH_ROWS = SG_GRID_Y * MMA_M;
shared float16_t Csh_out[CSH_ROWS * WG_TILE_N];
#endif

// Running fp32 accumulator (across all groups).
coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
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
  const uint CHUNKS_PER_GROUP = uint(K4_per_group) * 4u / WG_TILE_K;
#else
  // Per-channel: a single quant "group" spanning all of K. The nested
  // groups x chunks loop below collapses to a flat chunk loop, the wsum/wsc
  // ping-pong never crosses a boundary, and the epilog runs exactly once.
  const uint num_groups = 1u;
  const uint CHUNKS_PER_GROUP = uint(num_groups_arg);
#endif
  const uint num_chunks = num_groups * CHUNKS_PER_GROUP;

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
      accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
    }
  }

  const uint K_BLOCKS_PER_CHUNK = WG_TILE_K >> 2u;

  // --- A staging tile map: one MMA_M x MMA_K coopmat tile per subgroup per
  //     slot. A chunk holds A_TILES_M x A_TILES_K such tiles; they are dealt
  //     round-robin across the NUM_SUBGROUPS subgroups, so every subgroup
  //     participates (dbuf4's per-thread map leaves WG_SIZE -
  //     A_ACTIVE_THREADS invocations idle whenever the tile is small).
  //     A_TILES_PER_SG rounds up, so the last slot may be partially used --
  //     the `t < NUM_A_TILES` guard below is subgroup-uniform (t depends only
  //     on gl_SubgroupID), which is what coopmat ops require.
  const uint A_TILES_M      = WG_TILE_M / MMA_M;
  const uint A_TILES_K      = WG_TILE_K / MMA_K;  // == NUM_K_SLABS
  const uint NUM_A_TILES    = A_TILES_M * A_TILES_K;
  const uint A_TILES_PER_SG = (NUM_A_TILES + NUM_SUBGROUPS - 1u) / NUM_SUBGROUPS;

#ifdef WEIGHT_INT4
  // --- B staging thread map: (block, col) slots; each slot extracts one
  //     ColumnMajor LDS uint (4 K-contiguous sign-extended int8) ---
  const uint B_TOTAL_SLOTS = K_BLOCKS_PER_CHUNK * WG_TILE_N;
  const uint B_SLOTS_PER_THREAD = B_TOTAL_SLOTS / WG_SIZE;
  const uint N8_PER_TILE = WG_TILE_N >> 3u;
#else
  // --- B staging thread map: one (k4, n4) ivec4 block per active thread ---
  const uint B_FETCH_SLOTS = K_BLOCKS_PER_CHUNK * (WG_TILE_N >> 2u);
  const uint N4_PER_TILE = WG_TILE_N >> 2u;
  const uint b_k4_in_chunk = gl_LocalInvocationID.x / N4_PER_TILE;
  const uint b_n_uint_col = gl_LocalInvocationID.x % N4_PER_TILE;
  const bool b_active = gl_LocalInvocationID.x < B_FETCH_SLOTS;
#endif

  // Prefetch temp registers. temp_A is a coopmat array (the -tr change);
  // indices into it are [[unroll]]-resolved compile-time constants, never
  // dynamic -- dynamic indexing of a coopmat array is exactly the construct
  // the Xclipse/AMD-PAL compiler has miscompiled before.
  // (temp_A removed)
#ifdef WEIGHT_INT4
  ivec4 temp_B[B_SLOTS_PER_THREAD];
  int   temp_wsum;
  float temp_wsc;
#else
  ivec4 temp_B;
#endif

  // =========================================================
  // PROLOGUE
  // =========================================================
  if (gl_LocalInvocationID.x < (WG_TILE_M >> 2u)) {
    const uint m4 = (tile_m_start >> 2u) + gl_LocalInvocationID.x;
    const vec4  sc = vec4(texelFetch(t_int8_input_scales, ivec3(m4, 0, 0), 0));
    const ivec4 zp = texelFetch(t_int8_input_zps,         ivec3(m4, 0, 0), 0);
    const uint base = gl_LocalInvocationID.x * 4u;
    ifs_sh[base + 0u] = sc.x;  ifs_sh[base + 1u] = sc.y;
    ifs_sh[base + 2u] = sc.z;  ifs_sh[base + 3u] = sc.w;
    izp_sh[base + 0u] = zp.x;  izp_sh[base + 1u] = zp.y;
    izp_sh[base + 2u] = zp.z;  izp_sh[base + 3u] = zp.w;
  }
  // Group 0 weight sums/scales -> slice 0.
  if (gl_LocalInvocationID.x < WG_TILE_N) {
    const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
    f16vec4 sv = t_weight_scales[n_idx >> 2u];
    wsc_sh[gl_LocalInvocationID.x] = float(sv[n_idx & 3u]);
    wsum_sh[gl_LocalInvocationID.x] = t_weight_sums[n_idx];
  }
  memoryBarrierShared();
  barrier();

  // izp/ifs are per-row activation params, constant across K groups —
  // broadcast them into coopmats ONCE; the group epilog reuses them every
  // group (they depend only on the row block i, not on the group or j).
  coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
      izp_bcast[MMAS_PER_SG_M];
  coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
      ifs_bcast[MMAS_PER_SG_M];
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    const uint local_m_base = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
    coopMatLoad(
        izp_bcast[i], izp_sh,
        local_m_base, /*stride=*/0u,
        gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(
        ifs_bcast[i], ifs_sh,
        local_m_base, /*stride=*/0u,
        gl_CooperativeMatrixLayoutColumnMajor);
  }

  // dbuf4: prefetch chunk 0 into temp registers, THEN store to slice 0 (no
  // barrier here -- the main loop's first iteration barriers before
  // reading slice 0).
  // (A prefetch removed: loaded directly in the MMA loop)
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
  {
    // store chunk 0 -> slice 0
    // (A LDS store removed)
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
      Bsh_int8[slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab] =
          uint(v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
    }
#else
    if (b_active) {
      const uint slab_idx   = b_k4_in_chunk / (MMA_K >> 2u);
      const uint k4_in_slab = b_k4_in_chunk % (MMA_K >> 2u);
      const uint n_col_base = b_n_uint_col * 4u;
      [[unroll]] for (uint n_in_blk = 0u; n_in_blk < 4u; ++n_in_blk) {
        Bsh_int8[slab_idx * B_SLAB_U32 + (n_col_base + n_in_blk) * B_STRIDE_U32 + k4_in_slab] =
            uint(temp_B[n_in_blk]);
      }
    }
#endif
  }

  // =========================================================
  // MAIN LOOP (dbuf4) — nested groups x chunks (kept nested; flattening it
  // with a conditional coopmat epilog crashes the Xclipse PAL compiler at
  // large spec-resolved trip counts). One barrier per chunk. Chunk
  // iteration (global index `chunk`):
  //   1. barrier   — A/B slice (chunk%2) fully written; on the first chunk
  //                  of group g, wsum/wsc slice (g%2) is too.
  //   2. prefetch  — chunk+1 (A blocks, B blocks) into temp; when chunk+1
  //                  starts a new group, also its wsum/wsc element. Skipped
  //                  entirely on the final chunk.
  //   3. int8 MMA  — on slice (chunk%2) into accum_int32.
  //   4. store     — temp -> A/B slice ((chunk+1)%2), unpacking the weight;
  //                  on a group boundary, wsum/wsc -> slice ((g+1)%2).
  // The group epilog runs unconditionally at the tail of each group.
  // =========================================================
  uint chunk = 0;
  for (uint group_i = 0; group_i < num_groups; ++group_i) {
    for (uint inner = 0; inner < CHUNKS_PER_GROUP; ++inner, ++chunk) {
      const bool has_next = chunk + 1u < num_chunks;
      const bool group_crossing = has_next && (inner + 1u == CHUNKS_PER_GROUP);
      const uint cur_a = (chunk % 2u) * ASH_SLICE_U32;
      const uint cur_b = (chunk % 2u) * BSH_SLICE_U32;
      const uint nxt_a = ((chunk + 1u) % 2u) * ASH_SLICE_U32;
      const uint nxt_b = ((chunk + 1u) % 2u) * BSH_SLICE_U32;

      barrier();

      // --- 2. prefetch chunk+1 -> temp ---
      if (has_next) {
        const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;
        // (A prefetch removed)
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
        if (group_crossing && gl_LocalInvocationID.x < WG_TILE_N) {
          const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
          f16vec4 sv = t_weight_scales[(group_i + 1u) * N4 + (n_idx >> 2u)];
          temp_wsc = float(sv[n_idx & 3u]);
          temp_wsum = t_weight_sums[(group_i + 1u) * N + n_idx];
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

      // --- 3. int8 MMA on the cur slice ---
      [[unroll]] for (uint k = 0; k < NUM_K_SLABS; ++k) {
        const uint slab_a_base_u32 = cur_a + k * A_SLAB_U32;
        const uint slab_b_base_u32 = cur_b + k * B_SLAB_U32;

        coopmat<int8_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
          // Offsets/strides are in int8 elements, matching the int8_t
          // binding. With A bound as int[] instead, this load silently
          // produced wrong results regardless of which unit was used.
          const uint a_row_stride_i8 = nblocks_x_A * 4u;  // int8 elements
          coopMatLoad(
              matA[i], t_packed_int8_input,
              (tile_m_start + row_a) * a_row_stride_i8 +
                  (chunk * WG_TILE_K + k * MMA_K),
              a_row_stride_i8,
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

      // --- 4. store temp (chunk+1) -> nxt slice ---
      if (has_next) {
        // (A LDS store removed)
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
          Bsh_int8[nxt_b + slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab] =
              uint(v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
        }
        if (group_crossing && gl_LocalInvocationID.x < WG_TILE_N) {
          const uint wbase_nxt = ((group_i + 1u) % 2u) * WG_TILE_N;
          wsum_sh[wbase_nxt + gl_LocalInvocationID.x] = temp_wsum;
          wsc_sh[wbase_nxt + gl_LocalInvocationID.x] = temp_wsc;
        }
#else
        if (b_active) {
          const uint slab_idx   = b_k4_in_chunk / (MMA_K >> 2u);
          const uint k4_in_slab = b_k4_in_chunk % (MMA_K >> 2u);
          const uint n_col_base = b_n_uint_col * 4u;
          [[unroll]] for (uint n_in_blk = 0u; n_in_blk < 4u; ++n_in_blk) {
            Bsh_int8[nxt_b + slab_idx * B_SLAB_U32 + (n_col_base + n_in_blk) * B_STRIDE_U32 + k4_in_slab] =
                uint(temp_B[n_in_blk]);
          }
        }
#endif
      }
    }  // chunks

    // --- Group epilog: dequant accum_int32 -> result, reset accum ---
    {
      const uint wbase = (group_i % 2u) * WG_TILE_N;
      [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
        const uint local_n_base = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

        coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsum_bcast;
        coopMatLoad(
            wsum_bcast, wsum_sh,
            wbase + local_n_base, /*stride=*/0u,
            gl_CooperativeMatrixLayoutRowMajor);

        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsc_bcast;
        coopMatLoad(
            wsc_bcast, wsc_sh,
            wbase + local_n_base, /*stride=*/0u,
            gl_CooperativeMatrixLayoutRowMajor);

        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> adjusted =
              accum_int32[i][j] - izp_bcast[i] * wsum_bcast;
          coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> adjusted_fp =
              coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(adjusted);
          coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> scales_outer =
              ifs_bcast[i] * wsc_bcast;
          result[i][j] += adjusted_fp * scales_outer;
          accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
        }
      }
    }
  }  // groups

  // --- Bias (optional) ---
#ifdef HAS_BIAS
  if (apply_bias > 0) {
    for (uint t = gl_LocalInvocationID.x; t < WG_TILE_N; t += WG_SIZE) {
      bias_sh[t] = float(t_bias[tile_n_start + t]);
    }
    memoryBarrierShared();
    barrier();
  }
#endif

  // --- Store result tile ---
  // N for the store address math MUST come from the spec constant, not the
  // sizes UBO (see out_N_arg above).
#ifdef IO_TEXTURE
  // Epilogue iteration i drains accumulator row-block i from EVERY subgroup
  // into Csh_out at once, so the SG_GRID_Y bands it holds are disjoint global
  // row ranges; the whole workgroup then imageStores them. lr / MMA_M is the
  // writing subgroup's warpInTile.y, so the global row reproduces the buffer
  // path's gi exactly.
  //
  // PORTABILITY NOTE: the barrier() in the loop body keeps this loop rolled
  // despite [[unroll]], so result[i][j] IS dynamically indexed. Coopmat arrays
  // are opaque per-lane storage and dynamic indexing is exactly the construct
  // the Xclipse/AMD-PAL compiler has broken before -- check this first if the
  // texture variants miscompile on M51.
  const uint CSH_TEXELS_PER_ROW = WG_TILE_N / 4u;
  const uint CSH_TEXELS = CSH_ROWS * CSH_TEXELS_PER_ROW;
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    // Guards Csh_out against the previous iteration's readers. Inert on i == 0
    // but must stay unconditional to remain workgroup-uniform.
    barrier();
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
#ifdef HAS_BIAS
      if (apply_bias > 0) {
        const uint local_n = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> bias_tile;
        coopMatLoad(bias_tile, bias_sh, local_n, 0u, gl_CooperativeMatrixLayoutRowMajor);
        result[i][j] += bias_tile;
      }
#endif
      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> out_tile =
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
              float(Csh_out[base]),
              float(Csh_out[base + 1u]),
              float(Csh_out[base + 2u]),
              float(Csh_out[base + 3u])));
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
        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> bias_tile;
        coopMatLoad(bias_tile, bias_sh, local_n, 0u, gl_CooperativeMatrixLayoutRowMajor);
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
