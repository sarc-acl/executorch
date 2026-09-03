/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * "zpgtr2": dbuf4zpgtr with its coopMat-mediated A staging replaced by the
 * uvec4 (128-bit) A path from `shmem_double_buf4-tr2.comp` on the
 * vk_cooperative_matrix_perf `gemm-ubm` reference branch (commit e02817d,
 * 2026-09-02). Everything else -- B staging (coalesced, no skew), zp-hoist,
 * byte-parallel nibble widening, the dbuf4 loop/barrier structure, the group
 * epilog and the bias/store epilogue -- is byte-identical to dbuf4zpgtr.
 *
 * WHY THIS IS NOT A VERBATIM PORT OF -tr2, and what was deliberately dropped:
 *
 *   -tr2 changes TWO things relative to -tr. (1) A moves from coopMatLoad/
 *   coopMatStore to a raw uvec4 load + a single uvec4 LDS store. (2) The A LDS
 *   row gains one uvec4 of padding (A_ROW_PAD_SH = ELEMENTS_PER_VEC4).
 *   Only (1) is ported here. (2) is deliberately NOT ported: it is
 *   derived-negative for our layout. Their Ash is a uvec4[] whose row is a full
 *   TILE_K span, so unpadded it collapses onto 16/8/4 of the 32 LDS banks at
 *   TILE_K 32/64/128 (4/8/16 accesses per bank); one uvec4 of pad restores the
 *   2-per-bank floor. OUR Ash row is one MMA_K slab = 16 B = 4 dwords, which
 *   already tiles all 32 banks at exactly 2 accesses per bank -- the floor for
 *   the 64 dwords one coopMatLoad touches. Adding their pad would take our row
 *   stride to 8 dwords, i.e. 16 banks and 4 accesses per bank: 2x WORSE, plus
 *   8 KiB more LDS. Bank count is not assumed -- GPU__GC__NUM_LDS_BANKS = 32,
 *   from the SUMD/PAL chip register headers, identical on every mgfx variant.
 *   Full derivation: openspec/changes/dq8ca-tr2-a-staging-port/results.md 2.5.
 *
 *   -tr2 also keeps B coopMat-staged. We cannot: B is int4-nibble-packed and
 *   coopMatLoad cannot unpack nibbles. That is a standing spec constraint
 *   (dq8ca-coopmat-a-staging) and was independently re-confirmed by
 *   coopmat-tr-tilesweep-4w-port's "-trb", which unpacked int4->int8 to make it
 *   possible and measured 13.55% SLOWER.
 *
 * A staging -- the ONLY delta from dbuf4zpgtr:
 *   dbuf4zpgtr: per-SUBGROUP MMA_M x MMA_K tile, coopMatLoad(global) ->
 *               coopMatStore(LDS).
 *   this file:  per-THREAD 16-int8 slot. 16 contiguous int8 of one row is
 *               exactly one MMA_K span, which is exactly one A_STRIDE_U32 LDS
 *               slot -- so it is ONE naturally-aligned 128-bit global load and
 *               ONE 128-bit LDS store, with no shuffle on either side.
 *               Ash_int8 is therefore uvec4[] rather than uint[].
 *
 * The LDS BYTE LAYOUT IS UNCHANGED. Proved, not assumed: the set of dwords this
 * file writes is identical to the set dbuf4zpgtr's coopMatStore writes (1024
 * dwords, exact cover of [0, ASH_SLICE_U32), verified exhaustively over all 256
 * threads / all subgroup+slot pairs). That is what makes the math-loop
 * coopMatLoad -- reindexed into uvec4 units, stride 1 -- provably equivalent.
 *
 * Expected effect, stated up front so the measurement is not read as
 * confirmation of a hope: A's LDS write is ~13% of LDS dword traffic and
 * WAIT_CNT_LGKM is 13.00% of WAVE_CYCLES, so A's LDS write is ~1.7% of kernel
 * time; A's global read is 2.87% (ablation-measured). The bytes moved are
 * IDENTICAL to dbuf4zpgtr -- 16 B per thread either way -- so this cannot help
 * LDS bandwidth, only instruction issue (1 x b128 vs 2 x b64 per thread) and
 * address math. Predicted kernel effect -0.5% to -1%; hard ceiling ~1.7%.
 *
 * Registers: temp_A is 4 dwords/lane as an ivec4, the same as dbuf4zpgtr's two
 * coopmat fragments. No occupancy change is expected from this swap.
 *
 * Hard preconditions, beyond dbuf4zpgtr's:
 *   K % 16 == 0 (the 128-bit global load), satisfied by every production shape;
 *   chunkK % 16 == 0, automatic since chunkK is a multiple of WG_TILE_K;
 *   WG_SIZE % A_IV4_PER_ROW == 0, automatic (both powers of two).
 *
 * Selected via
 * ET_VK_DQ8CA_COOPMAT_VARIANT=tsweep_dbuf4zpgtr2_t<M>x<N>k<K>g<SGX><SGY>s<32|64>.
 * NOT the default. Requires the row-major (kPackedInt8_4W) activation packer,
 * so dq8ca_variant_wants_rowmajor_a() must recognise this token too.
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
// (U): uvec4-typed so the A staging store is ONE 128-bit ds_write instead of
// four 32-bit ones. Element count is ASH_SLICE_U32/4; the byte layout, and
// therefore every address the math loop reads, is unchanged -- proved by
// address-equivalence against dbuf4zpgtr's coopMatStore offset set.
const uint ASH_SLICE_V4 = ASH_SLICE_U32 / 4u;
const uint A_SLAB_V4    = A_SLAB_U32 / 4u;
shared uvec4 Ash_int8[2u * ASH_SLICE_V4];
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
  // (U) A staging map: one 128-bit (16 int8) slot per thread, NOT a per-subgroup
  // coopmat tile. 16 contiguous int8 of one row is exactly one MMA_K span, which
  // is exactly one A_STRIDE_U32 slot in LDS -- so the global load and the LDS
  // store are both a single naturally-aligned 128-bit access with no shuffling.
  const uint A_IV4_PER_ROW    = WG_TILE_K / 16u;          // slots per A row
  const uint A_IV4_TOTAL      = WG_TILE_M * A_IV4_PER_ROW; // slots per chunk
  const uint A_IV4_PER_THREAD = (A_IV4_TOTAL + WG_SIZE - 1u) / WG_SIZE;
  // WG_SIZE is a multiple of A_IV4_PER_ROW (both powers of two, A_IV4_PER_ROW =
  // WG_TILE_K/16 <= WG_SIZE), so slot = tid + s*WG_SIZE decomposes exactly as
  //   slot / A_IV4_PER_ROW = a_si + s * A_ROWS_PER_PASS
  //   slot % A_IV4_PER_ROW = a_k16              (unchanged by s)
  // which is what lets both index expressions below stay fully hoisted -- the
  // same loop-invariant-index-math hoist that is an attributed win for B here.
  const uint A_ROWS_PER_PASS = WG_SIZE / A_IV4_PER_ROW;
  // Hoisted, loop-invariant: this thread's (row, 16-int8-block) coordinates.
  const uint a_si  = gl_LocalInvocationID.x / A_IV4_PER_ROW;
  const uint a_k16 = gl_LocalInvocationID.x % A_IV4_PER_ROW;
  // LDS destination in uvec4 units: slab a_k16, row a_si. Constant across chunks.
  const uint a_lds_v4 = a_k16 * A_SLAB_V4 + a_si;
  // Global row base in int8 elements. Constant across chunks.
  const uint a_glb_base_i8 = (tile_m_start + a_si) * a_row_stride_i8 + a_k16 * 16u;

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

  // (U): temp_A is a plain ivec4 register set, not a coopmat array. Same VGPR
  // cost as dbuf4zpgtr's 2 coopmat fragments (4 dwords/lane either way), so no
  // occupancy change is expected from this swap.
  ivec4 temp_A[A_IV4_PER_THREAD];
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
  [[unroll]] for (uint s = 0; s < A_IV4_PER_THREAD; ++s) {
    const uint slot = gl_LocalInvocationID.x + s * WG_SIZE;
    if (slot < A_IV4_TOTAL) {
      temp_A[s] = t_packed_int8_input
          [(a_glb_base_i8 + s * A_ROWS_PER_PASS * a_row_stride_i8) >> 4u];
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
    [[unroll]] for (uint s = 0; s < A_IV4_PER_THREAD; ++s) {
      const uint slot = gl_LocalInvocationID.x + s * WG_SIZE;
      if (slot < A_IV4_TOTAL) {
        Ash_int8[a_lds_v4 + s * A_ROWS_PER_PASS] = uvec4(temp_A[s]);
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
      const uint cur_a_v4 = (chunk % 2u) * ASH_SLICE_V4;
      const uint cur_b = (chunk % 2u) * BSH_SLICE_U32;
      const uint nxt_a_v4 = ((chunk + 1u) % 2u) * ASH_SLICE_V4;
      const uint nxt_b = ((chunk + 1u) % 2u) * BSH_SLICE_U32;

      // dq8ca-tr2-a-staging-port task 2.2: this is the ONLY ordering point
      // between the previous iteration's Ash_int8/Bsh_int8 staging stores and
      // this iteration's coopMatLoad of them, and it was a bare barrier().
      // On the M51 Xclipse/AMD-PAL driver barrier() alone does NOT order
      // shared stores against a subsequent coopMatLoad -- symptom is one stale
      // MMA_M-row band of A, all columns, ~2.5% of runs, no crash (found
      // 2026-09-02 in sdpa_compute_out_coopmat.glsl; memory
      // `coopmat-lds-needs-explicit-memorybarriershared`). The three fenced
      // barriers already in this file guard wcorr_sh/bias_sh/Csh_out, NOT the
      // double-buffer staging. Measured cost of the fence elsewhere: 0.008%
      // CoV, i.e. free.
      memoryBarrierShared();
      barrier();

      // --- 2. prefetch chunk+1 -> temp ---
      if (has_next) {
        const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;
        // A staging (dbuf4tr's technique): coopMatLoad straight from global.
        [[unroll]] for (uint s = 0; s < A_IV4_PER_THREAD; ++s) {
          const uint slot = gl_LocalInvocationID.x + s * WG_SIZE;
          if (slot < A_IV4_TOTAL) {
            temp_A[s] = t_packed_int8_input
                [(a_glb_base_i8 + chunkK_nxt +
                  s * A_ROWS_PER_PASS * a_row_stride_i8) >> 4u];
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
        const uint slab_a_base_v4 = cur_a_v4 + k * A_SLAB_V4;
        const uint slab_b_base_u32 = cur_b + k * B_SLAB_U32;

        coopmat<int8_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
          coopMatLoad(
              matA[i], Ash_int8,
              // uvec4 units: one MMA_K row == 16 B == exactly 1 uvec4, so the
              // row stride is 1 and the offset is just the row index.
              slab_a_base_v4 + row_a,
              1u,
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
    [[unroll]] for (uint s = 0; s < A_IV4_PER_THREAD; ++s) {
      const uint slot = gl_LocalInvocationID.x + s * WG_SIZE;
      if (slot < A_IV4_TOTAL) {
        Ash_int8[nxt_a_v4 + a_lds_v4 + s * A_ROWS_PER_PASS] =
            uvec4(temp_A[s]);
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
    // dq8ca-tr2-a-staging-port task 2.2: fenced for the same reason as the
    // main-loop barrier above. This one is the WAR direction (last iteration's
    // Csh_out reads vs this iteration's coopMatStore writes), which the
    // `coopmat-lds-needs-explicit-memorybarriershared` record explicitly lists
    // as untested rather than known-safe -- and the fence is free.
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
