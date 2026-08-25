/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * "zp-hoisted" variant: identical to dbuf4 except that the activation
 * zero-point correction and the per-row activation scale are applied ONCE
 * after the group loop instead of once per quantization group.
 *
 * The per-group epilogue term factors exactly:
 *
 *   out[m][n] = ifs[m] * SUM_g wsc[g][n] * ( acc[g] - izp[m]*wsum[g][n] )
 *             = ifs[m] * [ SUM_g wsc[g][n]*acc[g] - izp[m]*SUM_g wsc[g][n]*wsum[g][n] ]
 *                                                    \__ weight-side only __/
 *
 * `ifs` is per-row and group-independent so it factors out entirely, and the
 * zero-point term separates into a per-row scalar times a per-output-channel
 * weight-side sum. That sum depends on no activation data, so it is
 * accumulated once into `wcorr_sh` in the prologue rather than being rebuilt
 * per group.
 *
 * Consequences vs dbuf4, per accumulator tile per group:
 *   - gone: izp*wsum multiply and the subtract     (48 v_sub* in the loop body)
 *   - gone: ifs*wsc multiply                       (part of 48 dequant-fp)
 *   - gone: the wsum_sh shared array and its ping-pong
 *   - gone: izp_bcast / ifs_bcast live across the loop  (register pressure)
 *   - kept: result += float(acc) * wsc
 *
 * Exact in exact arithmetic, but NOT bit-exact in fp32 -- the summation order
 * changes -- so it is gated on the correctness matrix like any other change.
 *
 * No new binding and no export-format change: the weight-side sum is derived
 * in the prologue from t_weight_scales and t_weight_sums, both already bound.
 *
 * Additionally widens int4 -> int8 byte-parallel (see widen_nibbles below),
 * replacing the per-nibble shift/mask/bias-subtract chain. Bit-identical.
 *
 * Additionally templates the B LDS skew (B_SKEW) so a power-of-two stride can
 * be measured against the baseline +1 skew.
 *
 * Selected via
 * ET_VK_DQ8CA_COOPMAT_VARIANT=tsweep_dbuf4zpb_t<M>x<N>k<K>g<SGX><SGY>s<32|64>.
 *
 * Original dbuf4 header follows.
 *
 * TILE/SUBGROUP-SWEEP variant of the int8 dq8ca_q4gsw coopmat shader's dbuf4
 * ("store-first-for-next", the ORIGINAL loop structure before specs/025 User
 * Story 1 picked dbuf2) loop structure (specs/041-dbuf4-tile-sweep). Forked
 * from linear_dq8ca_q4gsw_coopmat_tsweep.glsl (which carries dbuf2's loop,
 * the production winner) -- everything except the PROLOGUE/MAIN LOOP block
 * is identical: bindings, spec-constants, tile-geometry templating, LDS
 * layout (ColumnMajor B + skew), int8 WMMA thread maps, group epilog,
 * bias/store epilogue. Only the loop structure is swapped to dbuf4,
 * recovered from git commit 8d0f23ee78's
 * linear_dq8ca_q4gsw_coopmat_dbuf4.glsl (see specs/041/reference/) -- the
 * byte-identical pre-swap copy of what is now linear_dq8ca_qw_coopmat.glsl.
 *
 * The nested `groups x chunks` loop and unconditional group epilog are kept
 * exactly as in dbuf2 -- flattening them crashes the Xclipse PAL compiler at
 * large spec-resolved trip counts (see dbuf2's own header). Only the
 * store/barrier/prefetch ORDER within each chunk iteration is inverted:
 *
 *   dbuf2 (this file's base): store(temp, already prefetched -> cur slice)
 *     -> barrier -> MMA(cur) -> prefetch(next -> temp)  [store owns the
 *     CURRENT chunk, at the iteration's start]
 *   dbuf4 (this file): barrier -> prefetch(next -> temp) -> MMA(cur) ->
 *     store(temp -> next slice)  [store owns the NEXT chunk, at the
 *     iteration's end -- the mirror image]
 *
 * The group wsum/wsc ping-pong is inverted the same way: dbuf4 stores the
 * next group's values (prefetched during the crossing chunk) at the TAIL of
 * that chunk, instead of dbuf2's HEAD-of-new-group placement.
 *
 * Selected at dispatch via
 * ET_VK_DQ8CA_COOPMAT_VARIANT=tsweep_dbuf4_t<M>x<N>k<K>g<SGX><SGY>s<32|64>
 * (QuantizedLinear.cpp), additive to the existing tsweep_t... (dbuf2)
 * namespace.
 *
 * KHR Cooperative Matrix variant of the dynamically-quantized-activation
 * linear tiled shader (WEIGHT_NBITS=4):
 *   4  ->  linear_dq8ca_q4gsw_coopmat   INT4 group-symmetric weight
 *
 * Performs: out[M,N] = dequant(int8_act) * dequant(int_w) (+ bias)
 * via coopmat<int8> x coopmat<int8> -> coopmat<int32> on the matrix unit.
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
// t_input is unread here -- the activations arrive already quantized in
// t_packed_int8_input -- but stays declared so the binding layout matches the
// dispatch site. It tracks IO_STORAGE so the two IO tensors stay consistent.
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
// Intervention A: the bank-conflict skew is what makes B's LDS strides
// non-power-of-two (B_USEFUL_U32+1 = 5, and B_SLAB_U32 = WG_TILE_N*5), forcing
// integer multiplies in every B address. B_SKEW is templated so skew=4 (round
// up to a power of two) and skew=0 (no skew, like the reference -tr's
// ROW_PAD_SH=0) can both be measured against the baseline skew of 1.
const uint B_STRIDE_U32    = B_USEFUL_U32 + ${B_SKEW}u;
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
shared float wsc_sh[2u * WG_TILE_N];
// SUM_g wsc[g][n]*wsum[g][n] per output channel -- weight-side only, so it is
// accumulated once in the prologue. Replaces dbuf4's ping-ponged wsum_sh
// (which was 2*WG_TILE_N ints), so this is a net LDS saving.
shared float wcorr_sh[WG_TILE_N];

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


// Byte-parallel int4 -> int8 widening.
//
// The four nibbles this shader needs from one packed uint are ALREADY one per
// byte (bits 3:0 of each byte for parity 0, bits 7:4 for parity 1), so all four
// can be widened at once instead of with a per-nibble
// shift/mask/bias-subtract/mask chain.
//
// For v in [0,15] the biased value is v-8. `v ^ 8` is exactly the 4-bit
// two's-complement pattern of v-8, because -8 == +8 (mod 16):
//     v=0  -> 0x8 -> -8      v=7  -> 0xF -> -1
//     v=8  -> 0x0 ->  0      v=15 -> 0x7 -> +7
// so the only remaining work is sign-extending bit 3 into bits 7:4 per byte.
// `sgn * 0x1E` does that with no cross-byte carry: 0x08 * 0x1E == 0xF0 exactly,
// and sgn is at most 0x08080808 so the product is at most 0xF0F0F0F0.
//
// A naive `nib - 0x08080808` would NOT work -- it borrows across byte lanes
// whenever a nibble is < 8. Shifts must be on uint, not int, so the >> is
// logical rather than arithmetic.
//
// ~5 ops per 4 weights vs ~22 for the per-nibble chain; bit-identical output.
uint widen_nibbles(const uint w, const uint parity) {
  const uint nib = (parity == 0u) ? (w & 0x0F0F0F0Fu) : ((w >> 4u) & 0x0F0F0F0Fu);
  const uint p   = nib ^ 0x08080808u;
  const uint sgn = p & 0x08080808u;
  return p | (sgn * 0x1Eu);
}

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

  // --- A staging thread map: one (m4, k4) ivec4 block per active thread ---
  const uint K_BLOCKS_PER_CHUNK = WG_TILE_K >> 2u;
  const uint A_ACTIVE_THREADS = (WG_TILE_M >> 2u) * K_BLOCKS_PER_CHUNK;
  const uint a_m_block = gl_LocalInvocationID.x / K_BLOCKS_PER_CHUNK;
  const uint a_k_block = gl_LocalInvocationID.x % K_BLOCKS_PER_CHUNK;
  const bool a_active = gl_LocalInvocationID.x < A_ACTIVE_THREADS;

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

  // Prefetch temp registers.
  ivec4 temp_A;
#ifdef WEIGHT_INT4
  ivec4 temp_B[B_SLOTS_PER_THREAD];
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
  // Group 0 weight scales -> slice 0, and the hoisted weight-side correction
  // SUM_g wsc[g][n]*wsum[g][n] accumulated across ALL groups. The loop is
  // prologue-only (the prologue is ~1.4% of the dynamic instruction stream),
  // and it replaces per-group wsum work inside the loop body.
  if (gl_LocalInvocationID.x < WG_TILE_N) {
    const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
    f16vec4 sv0 = t_weight_scales[n_idx >> 2u];
    wsc_sh[gl_LocalInvocationID.x] = float(sv0[n_idx & 3u]);

    float corr = 0.0;
    for (uint g = 0; g < num_groups; ++g) {
      f16vec4 sv = t_weight_scales[g * N4 + (n_idx >> 2u)];
      corr += float(sv[n_idx & 3u]) * float(t_weight_sums[g * N + n_idx]);
    }
    wcorr_sh[gl_LocalInvocationID.x] = corr;
  }
  memoryBarrierShared();
  barrier();

  // NOTE: dbuf4 builds izp_bcast/ifs_bcast here and keeps them live across the
  // whole group loop. This variant needs them only AFTER the loop, so they are
  // loaded there instead -- that is the register-pressure saving.

  // dbuf4: prefetch chunk 0 into temp registers, THEN store to slice 0 (no
  // barrier here -- the main loop's first iteration barriers before
  // reading slice 0).
  if (a_active) {
    const uint m4_global = (tile_m_start >> 2u) + a_m_block;
    temp_A = t_packed_int8_input[m4_global * nblocks_x_A + a_k_block];
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
  {
    // store chunk 0 -> slice 0
    if (a_active) {
      const uint slab_idx       = a_k_block / (MMA_K >> 2u);
      const uint k_uint_in_slab = a_k_block % (MMA_K >> 2u);
      const uint base_row = a_m_block * 4u;
      [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
        Ash_int8[slab_idx * A_SLAB_U32 + (base_row + m4i) * A_STRIDE_U32 + k_uint_in_slab] =
            uint(temp_A[m4i]);
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
      const uint n_col      = n8_in_tile * 8u + r + parity * 4u;
      const uint slab_idx   = k4_in_chunk / (MMA_K >> 2u);
      const uint k4_in_slab = k4_in_chunk % (MMA_K >> 2u);
      Bsh_int8[slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab] =
          widen_nibbles(uint(temp_B[si][r]), parity);
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
        if (a_active) {
          const uint m4_global = (tile_m_start >> 2u) + a_m_block;
          const uint k4_global = (chunkK_nxt >> 2u) + a_k_block;
          temp_A = t_packed_int8_input[m4_global * nblocks_x_A + k4_global];
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
        if (group_crossing && gl_LocalInvocationID.x < WG_TILE_N) {
          const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
          f16vec4 sv = t_weight_scales[(group_i + 1u) * N4 + (n_idx >> 2u)];
          temp_wsc = float(sv[n_idx & 3u]);
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

      // --- 4. store temp (chunk+1) -> nxt slice ---
      if (has_next) {
        if (a_active) {
          const uint slab_idx       = a_k_block / (MMA_K >> 2u);
          const uint k_uint_in_slab = a_k_block % (MMA_K >> 2u);
          const uint base_row = a_m_block * 4u;
          [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
            Ash_int8[nxt_a + slab_idx * A_SLAB_U32 + (base_row + m4i) * A_STRIDE_U32 + k_uint_in_slab] =
                uint(temp_A[m4i]);
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
          const uint n_col      = n8_in_tile * 8u + r + parity * 4u;
          const uint slab_idx   = k4_in_chunk / (MMA_K >> 2u);
          const uint k4_in_slab = k4_in_chunk % (MMA_K >> 2u);
          Bsh_int8[nxt_b + slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab] =
              widen_nibbles(uint(temp_B[si][r]), parity);
        }
        if (group_crossing && gl_LocalInvocationID.x < WG_TILE_N) {
          const uint wbase_nxt = ((group_i + 1u) % 2u) * WG_TILE_N;
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

    // --- Group epilog: scale-only accumulate, reset accum ---
    // Just result += float(acc) * wsc. The zero-point subtract and the ifs
    // multiply are hoisted out of the group loop (applied once below).
    {
      const uint wbase = (group_i % 2u) * WG_TILE_N;
      [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
        const uint local_n_base = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsc_bcast;
        coopMatLoad(
            wsc_bcast, wsc_sh,
            wbase + local_n_base, /*stride=*/0u,
            gl_CooperativeMatrixLayoutRowMajor);

        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          result[i][j] +=
              coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(
                  accum_int32[i][j]) * wsc_bcast;
          accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
        }
      }
    }
  }  // groups

  // --- Hoisted correction, applied ONCE: ---------------------------------
  //   result = ifs * ( result - izp * SUM_g wsc_g*wsum_g )
  // izp/ifs are loaded here rather than before the group loop so they are not
  // live across it.
  {
    coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
        izpf_bcast[MMAS_PER_SG_M];
    coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
        ifs_bcast[MMAS_PER_SG_M];
    [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
      const uint local_m_base = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
      coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> izp_i;
      coopMatLoad(
          izp_i, izp_sh, local_m_base, /*stride=*/0u,
          gl_CooperativeMatrixLayoutColumnMajor);
      izpf_bcast[i] =
          coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(izp_i);
      coopMatLoad(
          ifs_bcast[i], ifs_sh, local_m_base, /*stride=*/0u,
          gl_CooperativeMatrixLayoutColumnMajor);
    }
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      const uint local_n_base = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wcorr_bcast;
      coopMatLoad(
          wcorr_bcast, wcorr_sh, local_n_base, /*stride=*/0u,
          gl_CooperativeMatrixLayoutRowMajor);
      [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        result[i][j] = ifs_bcast[i] * (result[i][j] - izpf_bcast[i] * wcorr_bcast);
      }
    }
  }

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
