/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * "zpg" + "-tr" combination: linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4zpg.glsl
 * with its per-thread scalar A-staging replaced by
 * linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4tr.glsl's coopMat-mediated A
 * staging (coopMatLoad(global) -> coopmat<> -> coopMatStore(LDS)). This is
 * an ADDITIVE combination, not a redesign: every non-A-staging block below
 * (B staging: coalesced write, no skew; zp-hoist: izp/ifs applied once after
 * the group loop via wcorr_sh; byte-parallel nibble widening; static
 * A_ALWAYS_ACTIVE branch elision -- N/A here, see below; group epilog;
 * bias/store epilogue) is byte-identical to dbuf4zpg's. Only the A-staging
 * block (prologue load+store, main-loop prefetch+store) is dbuf4tr's,
 * verbatim.
 *
 * dbuf4zpg's per-thread A staging used an `a_active` guard (statically always
 * true when A_ACTIVE_THREADS == WG_SIZE, via the A_MAP_FULL-gated
 * A_ALWAYS_ACTIVE macro). dbuf4tr's per-SUBGROUP tile map has no equivalent
 * concept -- every subgroup participates via a `t < NUM_A_TILES` guard that
 * depends only on gl_SubgroupID, not gl_LocalInvocationID.x -- so
 * A_MAP_FULL/A_ALWAYS_ACTIVE is dropped entirely in this file; it would be
 * dead code for the new A-staging block.
 *
 * Rationale for combining this way (not the reverse) and why this is worth
 * building at all: see this change's design.md D0-D3. In short -- the only
 * existing measurement of dbuf4tr's A-staging technique (28.72-30.51%,
 * dq8ca-arch-redesign) was taken against dbuf4tr's own pre-zpg baseline
 * (old B skew, no byte-parallel widening, no branch elision) -- a materially
 * weaker shader than the 46.49-46.50% dbuf4zpg this file now combines it
 * with. This file exists to answer whether that combination performs
 * differently now that register pressure is already reduced.
 *
 * A staging (the actual delta from dbuf4zpg):
 *   dbuf4zpg: per-thread (m4, k4) ivec4 fetch, hoisted a_lds_off0/a_glb_row;
 *             only A_ACTIVE_THREADS invocations participate, each scattering
 *             4 rows into Ash_int8 with 4 scalar stores.
 *   this file: per-SUBGROUP MMA_M x MMA_K tile fetch via coopMatLoad straight
 *             from a ROW-MAJOR (kPackedInt8_4W) int8 activation buffer, then
 *             coopMatStore into the same Ash_int8 slot -- dbuf4tr's mapping,
 *             unmodified (not re-derived; see design.md D3).
 *
 * t_packed_int8_input is therefore bound the same way dbuf4tr binds it: a
 * SCALAR int8_t array in the kPackedInt8_4W layout (plain row-major int8,
 * row stride K), produced by quantize_and_pack_4w_with_group_sums.glsl.
 * QuantizedLinear.cpp's dq8ca_variant_wants_rowmajor_a() must recognize this
 * file's variant token (tsweep_dbuf4zpgtr_t...) the same way it already
 * recognizes tsweep_dbuf4tr_t/trm_t/trd_t, so graph-build time (packer
 * selection) and dispatch time (kernel selection) cannot disagree.
 *
 * B CANNOT be coopmat-staged (int4 nibble unpack; a coopmat's per-lane layout
 * is opaque to hand-assembly from unpacked registers) -- unchanged from both
 * parent files. B staging below is dbuf4zpg's byte-parallel, coalesced,
 * no-skew version, untouched.
 *
 * The loop structure is dbuf4's (both parents share it), unchanged:
 *   prologue: prefetch chunk 0 -> temp, store to slice 0 (no barrier)
 *   per iter: barrier -> prefetch(next) -> MMA(cur) -> store(next)
 * kept nested (groups x chunks) with an unconditional group epilog --
 * flattening it crashes the Xclipse PAL compiler at large spec-resolved trip
 * counts (see dbuf2's own header).
 *
 * Selected via
 * ET_VK_DQ8CA_COOPMAT_VARIANT=tsweep_dbuf4zpgtr_t<M>x<N>k<K>g<SGX><SGY>s<32|64>
 * (QuantizedLinear.cpp), additive to the tsweep_dbuf4zpg_t..., tsweep_dbuf4tr_t...
 * and tsweep_t... namespaces. NOT the default -- unvalidated until it passes
 * repeated test_llama_microbench --correctness-only runs (see
 * dq8ca_coopmat_variant()'s comment on why a single pass is not proof).
 *
 * Performs: out[M,N] = dequant(int8_act) * dequant(int_w) (+ bias)
 * via coopmat<int8> x coopmat<int8> -> coopmat<int32> on the matrix unit.
 *
 * Hard preconditions (dbuf4zpg's, plus dbuf4tr's row-major/alignment ones):
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
// coopMatLoad below has a MATCHING component type (see dbuf4tr's header for
// why the type must match on this driver).
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
// array (row stride = K int8) -- dbuf4tr's binding, unchanged. The stock
// 4h4w layout dbuf4zpg uses is NOT row-major (component index selects a row,
// non-affine), so it cannot be addressed by any coopMatLoad.
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
// No skew + coalesced write -- dbuf4zpg's B fix, unchanged (this file does
// not touch B staging at all).
const uint B_STRIDE_U32    = B_USEFUL_U32;
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
// accumulated once in the prologue. dbuf4zpg's zp-hoist, unchanged.
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


// Byte-parallel int4 -> int8 widening. dbuf4zpg's, unchanged (B-side only).
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
  // A row stride in INT8 elements (dbuf4tr's binding is row-major int8, not
  // the 4h4w ivec4 block layout dbuf4zpg used -- so A addressing below is in
  // int8 elements, not int, and derived from nblocks_x_A so it matches the
  // packer's `m_row * K4 + k4` addressing exactly (K % 4 == 0 makes them
  // equal to K directly).
  const uint a_row_stride_i8 = nblocks_x_A * 4u;

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

  // --- A staging tile map (dbuf4tr's, unmodified): one MMA_M x MMA_K coopmat
  //     tile per subgroup per slot, dealt round-robin across the
  //     NUM_SUBGROUPS subgroups so every subgroup participates. Replaces
  //     dbuf4zpg's per-thread (m4, k4) map / a_active guard entirely -- see
  //     design.md D3 for why this is reused as-is, not re-derived.
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

  // ===== INTERVENTION F: hoist loop-invariant B staging index math =====
  // dbuf4zpg's, unchanged -- B staging is untouched by this file's A-staging
  // swap. See dbuf4zpg's header for the full rationale (ablation-attributed
  // -16.8% block, mostly index arithmetic not memory traffic).
#ifdef WEIGHT_INT4
  uint b_lds_off[B_SLOTS_PER_THREAD];  // LDS store offset within a slice
  uint b_comp[B_SLOTS_PER_THREAD];     // which ivec4 component feeds this slot
  uint b_par[B_SLOTS_PER_THREAD];      // nibble parity for this slot
  uint b_n8blk[B_SLOTS_PER_THREAD];    // global texel column (N/8 blocks)
  uint b_k4off[B_SLOTS_PER_THREAD];    // k4 offset of this slot within a chunk
  [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
    const uint a           = gl_LocalInvocationID.x + si * WG_SIZE;
    const uint slab_idx    = a / B_SLAB_U32;
    const uint local_a     = a % B_SLAB_U32;
    const uint n_col       = local_a / B_STRIDE_U32;
    const uint k4_in_slab  = local_a % B_STRIDE_U32;
    const uint k4_in_chunk = slab_idx * (MMA_K >> 2u) + k4_in_slab;
    const uint n8_in_tile  = n_col >> 3u;
    const uint rem         = n_col & 7u;
    b_lds_off[si] = a;
    b_comp[si]    = rem & 3u;
    b_par[si]     = rem >> 2u;
    b_n8blk[si]   = (tile_n_start >> 3u) + n8_in_tile;
    b_k4off[si]   = k4_in_chunk;
  }
#endif

  // Prefetch temp registers. temp_A is a coopmat array (dbuf4tr's A-staging
  // technique); indices into it are [[unroll]]-resolved compile-time
  // constants, never dynamic -- dynamic indexing of a coopmat array is
  // exactly the construct the Xclipse/AMD-PAL compiler has miscompiled
  // before.
  coopmat<int8_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA>
      temp_A[A_TILES_PER_SG];
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
  // SUM_g wsc[g][n]*wsum[g][n] accumulated across ALL groups. dbuf4zpg's
  // zp-hoist, unchanged.
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

  // NOTE: dbuf4zpg builds izp_bcast/ifs_bcast AFTER the group loop, not here
  // -- that is the register-pressure saving zp-hoist buys. Unchanged.

  // dbuf4: prefetch chunk 0 into temp registers, THEN store to slice 0 (no
  // barrier here -- the main loop's first iteration barriers before
  // reading slice 0).
  //
  // A staging (dbuf4tr's technique): per-subgroup coopMatLoad straight from
  // the row-major global buffer.
  [[unroll]] for (uint s = 0; s < A_TILES_PER_SG; ++s) {
    const uint t = gl_SubgroupID + s * NUM_SUBGROUPS;
    if (t < NUM_A_TILES) {
      const uint tm = t / A_TILES_K;
      const uint tk = t % A_TILES_K;
      coopMatLoad(
          temp_A[s], t_packed_int8_input,
          (tile_m_start + tm * MMA_M) * a_row_stride_i8 + tk * MMA_K,
          a_row_stride_i8,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
#ifdef WEIGHT_INT4
  [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
#ifdef WEIGHT_BUFFER
    temp_B[si] = t_packed_weight[(b_n8blk[si] * nblocks_x_A) + b_k4off[si]];
#else
    temp_B[si] = texelFetch(t_packed_weight, ivec2(b_k4off[si], b_n8blk[si]), 0);
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
    // A staging (dbuf4tr's technique): coopMatStore into the same Ash_int8
    // slot layout dbuf4zpg's scalar scatter used to write.
    [[unroll]] for (uint s = 0; s < A_TILES_PER_SG; ++s) {
      const uint t = gl_SubgroupID + s * NUM_SUBGROUPS;
      if (t < NUM_A_TILES) {
        const uint tm = t / A_TILES_K;
        const uint tk = t % A_TILES_K;
        coopMatStore(
            temp_A[s], Ash_int8,
            tk * A_SLAB_U32 + (tm * MMA_M) * A_STRIDE_U32,
            A_STRIDE_U32,
            gl_CooperativeMatrixLayoutRowMajor);
      }
    }
#ifdef WEIGHT_INT4
    [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
      Bsh_int8[b_lds_off[si]] =
          widen_nibbles(uint(temp_B[si][b_comp[si]]), b_par[si]);
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
  //                  of group g, wsc slice (g%2) is too.
  //   2. prefetch  — chunk+1 (A tiles, B blocks) into temp; when chunk+1
  //                  starts a new group, also its wsc element. Skipped
  //                  entirely on the final chunk.
  //   3. int8 MMA  — on slice (chunk%2) into accum_int32.
  //   4. store     — temp -> A/B slice ((chunk+1)%2), unpacking the weight;
  //                  on a group boundary, wsc -> slice ((g+1)%2).
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

      // coopmat-lds-fence: barrier() alone does NOT order shared stores against a
      // subsequent coopMatLoad on the M51 Xclipse/AMD-PAL driver -- symptom is one
      // stale MMA_M-row band of A, all columns, ~2.5% of runs, no crash (observed
      // 2026-09-02 in sdpa_compute_out_coopmat.glsl). Measured cost of the fence:
      // none (see this change's results). See memory
      // `coopmat-lds-needs-explicit-memorybarriershared`.
      memoryBarrierShared();
      barrier();

      // --- 2. prefetch chunk+1 -> temp ---
      if (has_next) {
        const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;
        // A staging (dbuf4tr's technique): coopMatLoad straight from global.
        [[unroll]] for (uint s = 0; s < A_TILES_PER_SG; ++s) {
          const uint t = gl_SubgroupID + s * NUM_SUBGROUPS;
          if (t < NUM_A_TILES) {
            const uint tm = t / A_TILES_K;
            const uint tk = t % A_TILES_K;
            coopMatLoad(
                temp_A[s], t_packed_int8_input,
                (tile_m_start + tm * MMA_M) * a_row_stride_i8 + chunkK_nxt +
                    tk * MMA_K,
                a_row_stride_i8,
                gl_CooperativeMatrixLayoutRowMajor);
          }
        }
#ifdef WEIGHT_INT4
        [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
          const uint k4_blk = (chunkK_nxt >> 2u) + b_k4off[si];
#ifdef WEIGHT_BUFFER
          temp_B[si] = t_packed_weight[(b_n8blk[si] * nblocks_x_A) + k4_blk];
#else
          temp_B[si] = texelFetch(t_packed_weight, ivec2(k4_blk, b_n8blk[si]), 0);
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
        // A staging (dbuf4tr's technique): coopMatStore into the nxt slice.
        [[unroll]] for (uint s = 0; s < A_TILES_PER_SG; ++s) {
          const uint t = gl_SubgroupID + s * NUM_SUBGROUPS;
          if (t < NUM_A_TILES) {
            const uint tm = t / A_TILES_K;
            const uint tk = t % A_TILES_K;
            coopMatStore(
                temp_A[s], Ash_int8,
                nxt_a + tk * A_SLAB_U32 + (tm * MMA_M) * A_STRIDE_U32,
                A_STRIDE_U32,
                gl_CooperativeMatrixLayoutRowMajor);
          }
        }
#ifdef WEIGHT_INT4
        [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
          Bsh_int8[nxt_b + b_lds_off[si]] =
              widen_nibbles(uint(temp_B[si][b_comp[si]]), b_par[si]);
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
    // dbuf4zpg's, unchanged. Just result += float(acc) * wsc. The
    // zero-point subtract and the ifs multiply are hoisted out of the group
    // loop (applied once below).
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

  // --- Hoisted correction, applied ONCE: --------------------------------
  //   result = ifs * ( result - izp * SUM_g wsc_g*wsum_g )
  // dbuf4zpg's, unchanged. izp/ifs are loaded here rather than before the
  // group loop so they are not live across it.
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
    // coopmat-lds-fence: barrier() alone does NOT order shared stores against a
    // subsequent coopMatLoad on the M51 Xclipse/AMD-PAL driver -- symptom is one
    // stale MMA_M-row band of A, all columns, ~2.5% of runs, no crash (observed
    // 2026-09-02 in sdpa_compute_out_coopmat.glsl). Measured cost of the fence:
    // none (see this change's results). See memory
    // `coopmat-lds-needs-explicit-memorybarriershared`.
    memoryBarrierShared();
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
