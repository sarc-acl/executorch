/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * INT8 convolution as a cooperative-matrix GEMM with int8 operands and int32
 * accumulation: acc[m][n] = sum_k in[m][k] * w[n][k], then the requantization
 * of q8ta_conv2d_pw.glsl (same expressions, so the output equals it bit for
 * bit). m runs over output pixels in quads of 4 consecutive widths of one row
 * (the 4W4C block), padded to whole quads; n is the output channel.
 *
 * K index k = 4 k4 + j is byte j of int k4 of a pixel's row: input channel
 * 4 k4 + j for a 1x1 convolution (t_in in 4W4C, one ivec4 = 4 pixels x 4
 * channels), or, with DIRECT, live tap t = k4 / C4 (row-major over the kernel)
 * and channel 4 (k4 % C4) + j, gathered per pixel from the input (positions
 * outside it read the input zero point, as q8ta_im2col writes them; with
 * BYTES, for any channel count, k = t C_in + ci and each byte is gathered
 * alone, as stock packs the weights of q8ta_conv2d). The
 * weights are the stock packing: texel (n4, k4) of t_weight holds output
 * channels 4 n4 .. + 3, each as the 4 bytes of K block k4.
 *
 * Shared memory, per K step of WG_TILE_K and buffer slice: A as
 * [K slab of MMA_K][m][MMA_K / 4 uint] (row-major fragments) and B as
 * [K slab][n][MMA_K / 4 uint] (column-major fragments), double buffered with
 * the next step prefetched into registers during the MMAs.
 *
 * K_SLICES > 1: work group x is channel tile x % n_tiles of K slice
 * x / n_tiles; it stores its int32 tile to t_scratch
 * ([slice][M_pad][N_pad], row-major), and the MERGE shader sums the slices
 * (exact) and runs the epilogue. MERGE: one invocation per output int (pixel,
 * 4 channels). ATOMIC (_sa<slices>): no merge dispatch; each slice counts its
 * arrival on the tile's counter and the last one sums the partials and runs
 * the epilogue.
 *
 * ASYNC (_af, not with BYTES): every load of a K step reads at coordinates
 * clamped into its tensor, without a branch or select on the loaded value;
 * which texels are outside (past K or N, or the gather's padding) is kept from
 * the coordinates alone and applied when the step is stored to shared memory,
 * which stores the same values as the other forms.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : require

#define PRECISION ${PRECISION}
#define VEC4_T vec4

layout(std430) buffer;

#include "indexing.glslh"
#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", "int", "buffer", is_scalar_array=True)}
$if DIRECT:
  ${layout_declare_tensor(B, "r", "t_in", "int", "buffer", is_scalar_array=True)}
$else:
  ${layout_declare_tensor(B, "r", "t_in", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight", "int", "texture2d", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", "float", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", "float", "buffer", is_scalar_array=False)}
$if ATOMIC:
  // The partial tiles and one arrival counter per output tile, coherent: the
  // last slice of a tile reads what the others stored.
  ${layout_declare_buffer(B, "rw", "t_scratch", "int", precision="coherent")}
  ${layout_declare_buffer(B, "rw", "t_counter", "uint", precision="coherent")}
$elif K_SLICES > 1 or MERGE:
  ${layout_declare_tensor(B, "rw", "t_scratch", "int", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  float input_scale;
  int input_zp;
  float output_inv_scale;
  int output_zp;
  // K blocks of 4 bytes (weight texture rows) and the scratch geometry.
  int K4;
  int M_pad;
  int N_pad;
  int k_slices;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "1")}
${layout_declare_spec_const(C, "int", "activation_type", "0")}
${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
// DIRECT geometry: input channel blocks, kernel, stride, padding, dilation.
${layout_declare_spec_const(C, "int", "dg_c4", "1")}
${layout_declare_spec_const(C, "int", "dg_kh", "1")}
${layout_declare_spec_const(C, "int", "dg_kw", "1")}
${layout_declare_spec_const(C, "int", "dg_sh", "1")}
${layout_declare_spec_const(C, "int", "dg_sw", "1")}
${layout_declare_spec_const(C, "int", "dg_ph", "0")}
${layout_declare_spec_const(C, "int", "dg_pw", "0")}
${layout_declare_spec_const(C, "int", "dg_dh", "1")}
${layout_declare_spec_const(C, "int", "dg_dw", "1")}
// BYTES: input channels (any count).
${layout_declare_spec_const(C, "int", "dg_cin", "4")}
// 1: the requantization of q8ta_conv2d.glsl (the direct shader), 0: that of
// q8ta_conv2d_pw.glsl.
${layout_declare_spec_const(C, "int", "direct_requant", "0")}

$if DIRECT:
  #define DIRECT
$if BYTES:
  #define BYTES
$if MERGE:
  #define MERGE
$if K_SLICES > 1:
  #define SPLIT_K
$if ATOMIC:
  #define ATOMIC_SPLIT
$if ASYNC:
  #define ASYNC_FETCH

const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};
const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};
const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
const uint K_SLICES = ${K_SLICES};

// Output pixel of padded row index m (quads of 4 widths): (n, h, w), and
// whether it lies inside the output.
struct Pixel {
  int n;
  int h;
  int w;
  bool inside;
};

Pixel pixel_of(const uint m) {
  const int W = int(outp.sizes[0][0]);
  const int H = int(outp.sizes[0][1]);
  const int W4 = div_up_4(W);
  const int quad = int(m >> 2u);
  const int row = quad / W4;
  Pixel p;
  p.n = row / H;
  p.h = row - p.n * H;
  p.w = 4 * (quad - row * W4) + int(m & 3u);
  p.inside = p.w < W && p.n < int(outp.sizes[0][3]);
  return p;
}

int outp_int_index(const Pixel p, const int c4) {
  if (get_outer_packed_dim_block_size(outp_layout) == 1) {
    return p.n * int(outp.strides[0][3]) + p.h * int(outp.strides[0][1]) +
        p.w * int(outp.strides[0][0]) + c4 * int(outp.strides[0][2]);
  }
  return mul_4(
             p.n * int(outp.strides[0][3]) + p.h * int(outp.strides[0][1]) +
             (p.w >> 2) * int(outp.strides[0][0]) +
             c4 * int(outp.strides[0][2])) +
      (p.w & 3);
}

// The requantization of q8ta_conv2d_pw.glsl (signed dot product) for the 4
// channels of block n4 of one pixel, or with direct_requant that of
// q8ta_conv2d.glsl.
int requantize(const ivec4 acc, const int n4) {
  if (direct_requant != 0) {
    const vec4 weight_sums = vec4(t_weight_sums[n4]);
    const vec4 weight_scales = vec4(t_weight_scales[n4]);
    vec4 facc = vec4(acc);
    facc -= weight_sums * input_zp;
    facc *= weight_scales * input_scale;
    if (apply_bias > 0) {
      const vec4 bias = vec4(t_bias[n4]);
      facc += bias;
    }
    if (activation_type > 0) {
      facc = max(facc, vec4(0.0));
    }
    const vec4 quantized = round(facc * output_inv_scale) + output_zp;
    return pack_into_int32(clamp(ivec4(quantized), -128, 127));
  }
  const ivec4 accum_adjusted =
      ivec4(-input_zp) * ivec4(t_weight_sums[n4]) + acc;
  const VEC4_T weight_scales = t_weight_scales[n4];
  vec4 float_out_texel;
  if (apply_bias > 0) {
    float_out_texel = fma(
        vec4(accum_adjusted),
        vec4(weight_scales) * input_scale,
        vec4(t_bias[n4]));
  } else {
    float_out_texel =
        vec4(accum_adjusted) * vec4(weight_scales * input_scale);
  }
  if (activation_type > 0) {
    float_out_texel = max(float_out_texel, vec4(0.0));
  }
  float_out_texel = round(float_out_texel * output_inv_scale) + output_zp;
  return pack_into_int32(clamp(ivec4(float_out_texel), -128, 127));
}

#ifdef MERGE

void main() {
  const int OC4 = div_up_4(int(outp.sizes[0][2]));
  const uint idx = gl_GlobalInvocationID.x;
  const uint m = idx / uint(OC4);
  const int n4 = int(idx % uint(OC4));
  if (m >= uint(M_pad)) {
    return;
  }
  const Pixel p = pixel_of(m);
  if (!p.inside) {
    return;
  }
  ivec4 acc = ivec4(0);
  for (int s = 0; s < k_slices; ++s) {
    const int base = (s * M_pad + int(m)) * N_pad + 4 * n4;
    acc += ivec4(
        t_scratch[base], t_scratch[base + 1], t_scratch[base + 2],
        t_scratch[base + 3]);
  }
  t_out[outp_int_index(p, n4)] = requantize(acc, n4);
}

#else

const uint NUM_SUBGROUPS = SG_GRID_X * SG_GRID_Y;
const uint WG_SIZE = NUM_SUBGROUPS * SUBGROUP_SIZE;
const uint SG_TILE_M = WG_TILE_M / SG_GRID_Y;
const uint SG_TILE_N = WG_TILE_N / SG_GRID_X;
const uint MMAS_M = SG_TILE_M / MMA_M;
const uint MMAS_N = SG_TILE_N / MMA_N;

// uints per fragment row (MMA_K bytes) and K slabs per step.
const uint RS = MMA_K / 4u;
const uint SLABS = WG_TILE_K / MMA_K;
const uint KQ = WG_TILE_K / 4u;
const uint A_SLAB = WG_TILE_M * RS;
const uint B_SLAB = WG_TILE_N * RS;
const uint A_SLICE = SLABS * A_SLAB;
const uint B_SLICE = SLABS * B_SLAB;
const uint B_BASE = 2u * A_SLICE;
const uint POOL_AB = 2u * (A_SLICE + B_SLICE);
// One drain pass: SG_GRID_Y x MMA_M rows of WG_TILE_N int32.
const uint C_ROWS = SG_GRID_Y * MMA_M;
const uint POOL_C = C_ROWS * WG_TILE_N;
shared uint sh[POOL_AB > POOL_C ? POOL_AB : POOL_C];

// A item: K block kq of one pixel quad (4 rows); B item: K block kq of one
// output channel block (4 columns). The K block is fastest, so neighboring
// invocations store neighboring uints of one fragment row.
const uint A_ITEMS = (WG_TILE_M / 4u) * KQ;
const uint A_PASSES = (A_ITEMS + WG_SIZE - 1u) / WG_SIZE;
const uint B_ITEMS = (WG_TILE_N / 4u) * KQ;
const uint B_PASSES = (B_ITEMS + WG_SIZE - 1u) / WG_SIZE;

coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    result[MMAS_M][MMAS_N];

#ifdef ATOMIC_SPLIT
shared uint sh_last;
#endif

#ifdef DIRECT
// Per A pass: the quad's batch offset, first input row and first input
// column (tap 0), and whether its pixels are inside the output.
struct QuadMap {
  int n_off[A_PASSES];
  int ih0[A_PASSES];
  int iw0[A_PASSES];
  bvec4 live[A_PASSES];
};

int load_in_int(const int n_off, const int ih, const int iw) {
  if (get_outer_packed_dim_block_size(inp_layout) == 1) {
    return t_in[n_off + ih * int(inp.strides[0][1]) +
                iw * int(inp.strides[0][0])];
  }
  return t_in[mul_4(n_off + ih * int(inp.strides[0][1]) +
                    (iw >> 2) * int(inp.strides[0][0])) +
              (iw & 3)];
}
#endif

void main() {
  const uvec2 sg = uvec2(gl_SubgroupID % SG_GRID_X, gl_SubgroupID / SG_GRID_X);
  const uint tid = gl_LocalInvocationID.x;

  const int W = int(outp.sizes[0][0]);
  const int H = int(outp.sizes[0][1]);
  const int W4 = div_up_4(W);
  const uint M = uint(4 * W4 * H * int(outp.sizes[0][3]));
  const uint N4 = uint(div_up_4(int(outp.sizes[0][2])));
  const uint K4u = uint(K4);

  const uint tile_m = gl_WorkGroupID.y + gl_WorkGroupID.z * gl_NumWorkGroups.y;
  if (tile_m * WG_TILE_M >= M) {
    return;
  }
  const uint tile_m_start = WG_TILE_M * tile_m;
#ifdef SPLIT_K
  const uint n_tiles = gl_NumWorkGroups.x / K_SLICES;
  const uint k_slice = gl_WorkGroupID.x / n_tiles;
  const uint tile_n_start = WG_TILE_N * (gl_WorkGroupID.x % n_tiles);
#else
  const uint tile_n_start = WG_TILE_N * gl_WorkGroupID.x;
#endif
  const uint n4_start = tile_n_start / 4u;
  const uint steps = (K4u + KQ - 1u) / KQ;
#ifdef SPLIT_K
  const uint q = (steps + K_SLICES - 1u) / K_SLICES;
  const uint s0 = min(k_slice * q, steps);
  const uint s1 = min(s0 + q, steps);
#else
  const uint s0 = 0u;
  const uint s1 = steps;
#endif

  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_N; ++j) {
      result[i][j] = coopmat<
          int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
    }
  }

  // Fixed per invocation: item K blocks, quads and channel blocks.
  uint a_kq[A_PASSES];
  uint a_row[A_PASSES];
  bool a_ok[A_PASSES];
#ifdef DIRECT
  QuadMap qm;
  const int zp_packed = pack_into_int32(ivec4(input_zp));
#else
  int a_base[A_PASSES];
#endif
  [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
    const uint item = p * WG_SIZE + tid;
    a_ok[p] = A_ITEMS % WG_SIZE == 0u || item < A_ITEMS;
    const uint it = min(item, A_ITEMS - 1u);
    a_kq[p] = it % KQ;
    a_row[p] = 4u * (it / KQ);
    const uint m = min(tile_m_start + a_row[p], M - 4u);
    const int quad = int(m >> 2u);
    const int row = quad / W4;
    const int n = row / H;
    const int h = row - n * H;
    const int w4 = quad - row * W4;
#ifdef DIRECT
    qm.n_off[p] = n * int(inp.strides[0][3]);
    qm.ih0[p] = h * dg_sh - dg_ph;
    qm.iw0[p] = 4 * w4 * dg_sw - dg_pw;
    qm.live[p] = lessThan(ivec4(4 * w4) + ivec4(0, 1, 2, 3), ivec4(W));
#else
    a_base[p] = n * int(inp.strides[0][3]) + h * int(inp.strides[0][1]) +
        w4 * int(inp.strides[0][0]);
#endif
  }
  uint b_kq[B_PASSES];
  uint b_col[B_PASSES];
  bool b_ok[B_PASSES];
  [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
    const uint item = p * WG_SIZE + tid;
    b_ok[p] = B_ITEMS % WG_SIZE == 0u || item < B_ITEMS;
    const uint it = min(item, B_ITEMS - 1u);
    b_kq[p] = it % KQ;
    b_col[p] = 4u * (it / KQ);
  }

  ivec4 temp_a[A_PASSES];
  ivec4 temp_b[B_PASSES];
#ifdef ASYNC_FETCH
#ifdef BYTES
#error "ASYNC_FETCH does not support BYTES"
#endif
  // Inside K (and N) per item, and per gathered texel inside the input.
  bool a_kin[A_PASSES];
  bool b_in[B_PASSES];
  const int in_sc = int(inp.strides[0][2]);
#ifdef DIRECT
  bvec4 a_in[A_PASSES];
  const int in_w = int(inp.sizes[0][0]);
  const int in_h = int(inp.sizes[0][1]);
  const int in_sw = int(inp.strides[0][0]);
  const int in_sh = int(inp.strides[0][1]);
  const bool in_4w = get_outer_packed_dim_block_size(inp_layout) != 1;
#endif
#endif

#define LOAD_STEP(step)                                                       \
  {                                                                           \
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {                          \
      const uint k4 = (step) * KQ + a_kq[p];                                  \
      LOAD_A(p, k4)                                                           \
    }                                                                         \
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {                          \
      const uint k4 = (step) * KQ + b_kq[p];                                  \
      const uint n4 = n4_start + b_col[p] / 4u;                               \
      LOAD_B(p, k4, n4)                                                       \
    }                                                                         \
  }

#ifdef ASYNC_FETCH
#define LOAD_B(p, k4, n4)                                                     \
  b_in[p] = k4 < K4u && n4 < N4;                                              \
  temp_b[p] = texelFetch(                                                     \
      t_weight, ivec2(min(n4, N4 - 1u), min(k4, K4u - 1u)), 0);
#else
#define LOAD_B(p, k4, n4)                                                     \
  temp_b[p] = k4 < K4u && n4 < N4 ? texelFetch(t_weight, ivec2(n4, k4), 0)    \
                                  : ivec4(0);
#endif

#if defined(DIRECT) && defined(BYTES)
// Byte b of K block k4 is input channel ci of tap t for K index
// 4 k4 + b = t dg_cin + ci (the stock packing of [C_out, align4(K)] rows).
#define LOAD_A(p, k4)                                                         \
  {                                                                           \
    temp_a[p] = ivec4(zp_packed);                                             \
    [[unroll]] for (int b = 0; b < 4; ++b) {                                  \
      const int k = 4 * int(k4) + b;                                          \
      if (k < dg_cin * dg_kh * dg_kw) {                                       \
        const int tap = k / dg_cin;                                           \
        const int ci = k - tap * dg_cin;                                      \
        const int ky = tap / dg_kw;                                           \
        const int kx = tap - ky * dg_kw;                                      \
        const int ih = qm.ih0[p] + ky * dg_dh;                                \
        const bool row_in = uint(ih) < uint(inp.sizes[0][1]);                 \
        const int c_off = qm.n_off[p] + (ci >> 2) * int(inp.strides[0][2]);   \
        [[unroll]] for (int j = 0; j < 4; ++j) {                              \
          const int iw = qm.iw0[p] + j * dg_sw + kx * dg_dw;                  \
          const int v = row_in && qm.live[p][j] &&                            \
                  uint(iw) < uint(inp.sizes[0][0])                            \
              ? load_in_int(c_off, ih, iw)                                    \
              : zp_packed;                                                    \
          temp_a[p][j] = bitfieldInsert(                                      \
              temp_a[p][j], bitfieldExtract(v, 8 * (ci & 3), 8), 8 * b, 8);   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }
#elif defined(DIRECT) && defined(ASYNC_FETCH)
// Every texel is read at its position clamped into the input; a_in marks the
// ones inside (the others are the zero point).
#define LOAD_A(p, k4)                                                         \
  {                                                                           \
    a_kin[p] = k4 < K4u;                                                      \
    const int kc = int(min(k4, K4u - 1u));                                    \
    const int tap = kc / dg_c4;                                               \
    const int c4 = kc - tap * dg_c4;                                          \
    const int ky = tap / dg_kw;                                               \
    const int kx = tap - ky * dg_kw;                                          \
    const int ih = qm.ih0[p] + ky * dg_dh;                                    \
    const bool row_in = uint(ih) < uint(in_h);                                \
    const int row_off =                                                       \
        qm.n_off[p] + c4 * in_sc + clamp(ih, 0, in_h - 1) * in_sh;            \
    [[unroll]] for (int j = 0; j < 4; ++j) {                                  \
      const int iw = qm.iw0[p] + j * dg_sw + kx * dg_dw;                      \
      const int iwc = clamp(iw, 0, in_w - 1);                                 \
      a_in[p][j] = all(bvec3(row_in, qm.live[p][j], uint(iw) < uint(in_w)));  \
      temp_a[p][j] = t_in[in_4w ? mul_4(row_off + (iwc >> 2) * in_sw) +       \
                                      (iwc & 3)                               \
                                : row_off + iwc * in_sw];                     \
    }                                                                         \
  }
#elif defined(DIRECT)
#define LOAD_A(p, k4)                                                         \
  {                                                                           \
    temp_a[p] = ivec4(0);                                                     \
    if (k4 < K4u) {                                                           \
      const int tap = int(k4) / dg_c4;                                        \
      const int c4 = int(k4) - tap * dg_c4;                                   \
      const int ky = tap / dg_kw;                                             \
      const int kx = tap - ky * dg_kw;                                        \
      const int ih = qm.ih0[p] + ky * dg_dh;                                  \
      const bool row_in = uint(ih) < uint(inp.sizes[0][1]);                   \
      const int c_off = qm.n_off[p] + c4 * int(inp.strides[0][2]);            \
      [[unroll]] for (int j = 0; j < 4; ++j) {                                \
        const int iw = qm.iw0[p] + j * dg_sw + kx * dg_dw;                    \
        temp_a[p][j] = row_in && qm.live[p][j] &&                             \
                uint(iw) < uint(inp.sizes[0][0])                              \
            ? load_in_int(c_off, ih, iw)                                      \
            : zp_packed;                                                      \
      }                                                                       \
    }                                                                         \
  }
#elif defined(ASYNC_FETCH)
#define LOAD_A(p, k4)                                                         \
  a_kin[p] = k4 < K4u;                                                        \
  temp_a[p] = t_in[a_base[p] + int(min(k4, K4u - 1u)) * in_sc];
#else
#define LOAD_A(p, k4)                                                         \
  temp_a[p] = k4 < K4u                                                        \
      ? t_in[a_base[p] + int(k4) * int(inp.strides[0][2])]                   \
      : ivec4(0);
#endif

// The word item p stores for row i: the loaded value, or what the other forms
// load in its place.
#if defined(ASYNC_FETCH) && defined(DIRECT)
#define A_VALUE(p, i)                                                         \
  (a_kin[p] ? uint(a_in[p][i] ? temp_a[p][i] : zp_packed) : 0u)
#elif defined(ASYNC_FETCH)
#define A_VALUE(p, i) (a_kin[p] ? uint(temp_a[p][i]) : 0u)
#else
#define A_VALUE(p, i) uint(temp_a[p][i])
#endif
#ifdef ASYNC_FETCH
#define B_VALUE(p, i) (b_in[p] ? uint(temp_b[p][i]) : 0u)
#else
#define B_VALUE(p, i) uint(temp_b[p][i])
#endif

#define STORE_STEP(slice)                                                     \
  {                                                                           \
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {                          \
      if (a_ok[p]) {                                                          \
        const uint o = (slice) * A_SLICE + (a_kq[p] / RS) * A_SLAB +          \
            a_row[p] * RS + a_kq[p] % RS;                                     \
        [[unroll]] for (uint i = 0; i < 4u; ++i) {                            \
          sh[o + i * RS] = A_VALUE(p, i);                                     \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {                          \
      if (b_ok[p]) {                                                          \
        const uint o = B_BASE + (slice) * B_SLICE + (b_kq[p] / RS) * B_SLAB + \
            b_col[p] * RS + b_kq[p] % RS;                                     \
        [[unroll]] for (uint i = 0; i < 4u; ++i) {                            \
          sh[o + i * RS] = B_VALUE(p, i);                                     \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }

#define MMA_STEP(slice)                                                       \
  [[unroll]] for (uint s = 0; s < SLABS; ++s) {                               \
    coopmat<int8_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA>            \
        mat_a[MMAS_M];                                                        \
    [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {                            \
      const uint row = MMA_M * (MMAS_M * sg.y + i);                           \
      coopMatLoad(                                                            \
          mat_a[i], sh, (slice) * A_SLICE + s * A_SLAB + row * RS, RS,        \
          gl_CooperativeMatrixLayoutRowMajor);                                \
    }                                                                         \
    coopmat<int8_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> mat_b;     \
    [[unroll]] for (uint j = 0; j < MMAS_N; ++j) {                            \
      const uint col = MMA_N * (MMAS_N * sg.x + j);                           \
      coopMatLoad(                                                            \
          mat_b, sh, B_BASE + (slice) * B_SLICE + s * B_SLAB + col * RS, RS,  \
          gl_CooperativeMatrixLayoutColumnMajor);                             \
      [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {                          \
        result[i][j] = coopMatMulAdd(mat_a[i], mat_b, result[i][j]);          \
      }                                                                       \
    }                                                                         \
  }

  if (s0 < s1) {
    LOAD_STEP(s0)
    STORE_STEP(0u)
    uint step;
    for (step = s0; step + 1u < s1; ++step) {
      memoryBarrierShared();
      barrier();
      LOAD_STEP(step + 1u)
      MMA_STEP((step - s0) % 2u)
      STORE_STEP((step - s0 + 1u) % 2u)
    }
    memoryBarrierShared();
    barrier();
    MMA_STEP((step - s0) % 2u)
  }

#ifdef SPLIT_K
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_N; ++j) {
      const uint row = tile_m_start + MMA_M * (MMAS_M * sg.y + i);
      const uint col = tile_n_start + MMA_N * (MMAS_N * sg.x + j);
      coopMatStore(
          result[i][j],
          t_scratch,
          (k_slice * uint(M_pad) + row) * uint(N_pad) + col,
          uint(N_pad),
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
#ifdef ATOMIC_SPLIT
  // The slice that completes the tile sums every slice's partial (int32,
  // exact) and runs the epilogue; it then resets the tile's counter for the
  // next dispatch (the counters start at zero).
  memoryBarrierBuffer();
  barrier();
  const uint tile_id = tile_m * n_tiles + gl_WorkGroupID.x % n_tiles;
  if (tid == 0u) {
    sh_last = atomicAdd(t_counter[tile_id], 1u) == K_SLICES - 1u ? 1u : 0u;
  }
  barrier();
  if (sh_last == 0u) {
    return;
  }
  memoryBarrierBuffer();
  const uint N4T = WG_TILE_N / 4u;
  for (uint t = tid; t < WG_TILE_M * N4T; t += WG_SIZE) {
    const uint m = tile_m_start + t / N4T;
    const uint n4 = n4_start + t % N4T;
    if (m < M && n4 < N4) {
      const Pixel px = pixel_of(m);
      if (px.inside) {
        ivec4 acc = ivec4(0);
        for (uint s = 0u; s < K_SLICES; ++s) {
          const uint base = (s * uint(M_pad) + m) * uint(N_pad) + 4u * n4;
          acc += ivec4(
              t_scratch[base], t_scratch[base + 1u], t_scratch[base + 2u],
              t_scratch[base + 3u]);
        }
        t_out[outp_int_index(px, int(n4))] = requantize(acc, int(n4));
      }
    }
  }
  if (tid == 0u) {
    t_counter[tile_id] = 0u;
  }
#endif
#else
  const uint N4T = WG_TILE_N / 4u;
  [[unroll]] for (uint i = 0; i < MMAS_M; ++i) {
    memoryBarrierShared();
    barrier();
    [[unroll]] for (uint j = 0; j < MMAS_N; ++j) {
      coopMatStore(
          result[i][j],
          sh,
          (sg.y * MMA_M) * WG_TILE_N + MMA_N * (MMAS_N * sg.x + j),
          WG_TILE_N,
          gl_CooperativeMatrixLayoutRowMajor);
    }
    memoryBarrierShared();
    barrier();
    for (uint t = tid; t < C_ROWS * N4T; t += WG_SIZE) {
      const uint cr = t / N4T;
      const uint lc = t % N4T;
      const uint lr = (cr / MMA_M) * SG_TILE_M + i * MMA_M + cr % MMA_M;
      const uint m = tile_m_start + lr;
      const uint n4 = n4_start + lc;
      if (m < M && n4 < N4) {
        const Pixel px = pixel_of(m);
        if (px.inside) {
          const uint h = cr * WG_TILE_N + 4u * lc;
          const ivec4 acc =
              ivec4(int(sh[h]), int(sh[h + 1u]), int(sh[h + 2u]), int(sh[h + 3u]));
          t_out[outp_int_index(px, int(n4))] = requantize(acc, int(n4));
        }
      }
    }
  }
#endif
}

#endif
