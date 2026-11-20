/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * FP32 convolution as a register-tiled GEMM with shared-memory staging
 * (change conv2d-all-used-kernels): out[m][n] = sum_k in[m][k] * w[n][k] +
 * bias[n], then the conv2d epilogue; m is the output pixel, n the output
 * channel. Operands, accumulation and epilogue are float32.
 *
 * Each work group computes a WG_TILE_M x WG_TILE_N tile; invocation t owns TM
 * consecutive pixels (row t / NT_N of the thread grid) and TN4 output channel
 * texels t % NT_N + j * NT_N, as vec4 accumulators. Per K step of WG_TILE_K
 * input channels the A tile (input texels, as A_sh[k][m]) and the B tile
 * (packed weights, B_sh[k][n4] vec4) are staged into shared memory, double
 * buffered with the next step prefetched into registers during the FMAs.
 *
 * K index k = 4 k4 + j is channel j of texel k4 of a pixel's row: input
 * texel k4 for a 1x1 convolution, or, with DIRECT, input channels
 * 4 (k4 % C4) .. of live tap k4 / C4 (row-major over dg_taps_h x dg_taps_w),
 * read at (oy stride_h - padding_h + ty dg_dilation_h, ox stride_w - padding_w
 * + tx dg_dilation_w); positions outside the input and channels at or above
 * dg_in_channels are zero. The weights are the 4OC x 4IC blocked packing of
 * pack_fp_linear_weight over [C_out, K]: vec4 entry ((k / 4) N4 + n4) 4 + k % 4
 * holds output channels 4 n4 .. + 3 of K index k.
 *
 * K_SLICES > 1: work group x is channel tile x % n_tiles of K slice
 * x / n_tiles; it stores its partial texels (no bias) to t_scratch at
 * (slice M + m) N4 + n4, and the MERGE shader sums the slices and runs the
 * epilogue, one invocation per output texel.
 */

#version 450 core

#extension GL_EXT_control_flow_attributes : require

#define PRECISION ${PRECISION}
#define VEC4_T vec4

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", "float", "texture3d")}
${layout_declare_tensor(B, "r", "t_in", "float", "texture3d")}
${layout_declare_tensor(B, "r", "t_weight", "float", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", "float", "texture2d")}
${layout_declare_tensor(B, "r", "t_residual", "float", "texture3d")}
${layout_declare_tensor(B, "r", "t_scale", "float", "texture2d")}
$if K_SLICES > 1 or MERGE:
  ${layout_declare_tensor(B, "rw", "t_scratch", "float", "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(push_constant) uniform restrict Block {
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
  float activation_a;
  float activation_b;
  float residual_alpha;
  // Rows of the packed weights (K / 4) and the K slices of the scratch.
  int K4;
  int k_slices;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "bool", "has_activation", "false")}
${layout_declare_spec_const(C, "int", "activation_op", "0")}
${layout_declare_spec_const(C, "bool", "has_residual", "false")}
${layout_declare_spec_const(C, "bool", "has_scale", "false")}
${layout_declare_spec_const(C, "int", "residual_op", "0")}
${layout_declare_spec_const(C, "uint", "dg_in_channels", "4")}
${layout_declare_spec_const(C, "uint", "dg_taps_h", "1")}
${layout_declare_spec_const(C, "uint", "dg_taps_w", "1")}
${layout_declare_spec_const(C, "uint", "dg_dilation_h", "1")}
${layout_declare_spec_const(C, "uint", "dg_dilation_w", "1")}

#include "conv2d_epilogue.glslh"

$if DIRECT:
  #define DIRECT
$if MERGE:
  #define MERGE
$if K_SLICES > 1:
  #define SPLIT_K

const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};
const uint TM = ${TM};
const uint TN4 = ${TN4};
const uint K_SLICES = ${K_SLICES};

vec4 epilogue_texel(vec4 v, const uint m, const uint n4, const uint W) {
  v += texelFetch(t_bias, ivec2(n4, 0), 0);
  const vec4 scale = has_scale ? texelFetch(t_scale, ivec2(n4, 0), 0) : vec4(1);
  const ivec3 pos = ivec3(m % W, m / W, n4);
  return apply_epilogue(v, pos, scale);
}

#ifdef MERGE

void main() {
  const uint W = uint(out_sizes.x);
  const uint M = W * uint(out_sizes.y);
  const uint N4 = uint(div_up_4(out_sizes.z));
  const uint t = gl_GlobalInvocationID.x;
  if (t >= M * N4) {
    return;
  }
  const uint m = t / N4;
  const uint n4 = t % N4;
  vec4 v = vec4(0);
  for (int s = 0; s < k_slices; ++s) {
    v += t_scratch[(uint(s) * M + m) * N4 + n4];
  }
  imageStore(t_out, ivec3(m % W, m / W, n4), epilogue_texel(v, m, n4, W));
}

#else

const uint NT_N = WG_TILE_N / (4u * TN4);
const uint NT_M = WG_TILE_M / TM;
const uint WG_SIZE = NT_M * NT_N;
const uint KQ = WG_TILE_K / 4u;
const uint N4T = WG_TILE_N / 4u;

// [slice][k][m] floats and [slice][k][n4] vec4.
shared float A_sh[2u * WG_TILE_K * WG_TILE_M];
shared vec4 B_sh[2u * WG_TILE_K * N4T];

// A item: texel kq of pixel m (pixels fastest); B item: entry k of channel
// texel n4 (texels fastest).
const uint A_ITEMS = WG_TILE_M * KQ;
const uint A_PASSES = (A_ITEMS + WG_SIZE - 1u) / WG_SIZE;
const uint B_ITEMS = WG_TILE_K * N4T;
const uint B_PASSES = (B_ITEMS + WG_SIZE - 1u) / WG_SIZE;

#ifdef DIRECT
const uint DG_C4 = (dg_in_channels + 3u) / 4u;
const uint DG_TAPS = dg_taps_h * dg_taps_w;

vec4 fetch_in(const ivec2 origin, const uint k4, const uint K4u) {
  const uint tap = k4 / DG_C4;
  const uint c4 = k4 - tap * DG_C4;
  const uint ty = tap / dg_taps_w;
  const ivec2 pos = origin +
      ivec2((tap - ty * dg_taps_w) * dg_dilation_w, ty * dg_dilation_h);
  const bool inside = k4 < K4u && uint(pos.x) < uint(in_sizes.x) &&
      uint(pos.y) < uint(in_sizes.y);
  const vec4 v = texelFetch(
      t_in, ivec3(inside ? pos : ivec2(0), inside ? int(c4) : 0), 0);
  const bvec4 lanes =
      lessThan(uvec4(4u * c4) + uvec4(0u, 1u, 2u, 3u), uvec4(dg_in_channels));
  return inside ? mix(vec4(0), v, lanes) : vec4(0);
}
#else
vec4 fetch_in(const ivec2 pos, const uint k4, const uint K4u) {
  return k4 < K4u ? texelFetch(t_in, ivec3(pos, int(k4)), 0) : vec4(0);
}
#endif

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint W = uint(out_sizes.x);
  const uint M = W * uint(out_sizes.y);
  const uint N4 = uint(div_up_4(out_sizes.z));
  const uint K4u = uint(K4);

  const uint tile_m = gl_WorkGroupID.y + gl_WorkGroupID.z * gl_NumWorkGroups.y;
  if (tile_m * WG_TILE_M >= M) {
    return;
  }
  const uint tile_m_start = WG_TILE_M * tile_m;
#ifdef SPLIT_K
  const uint n_tiles = gl_NumWorkGroups.x / K_SLICES;
  const uint k_slice = gl_WorkGroupID.x / n_tiles;
  const uint n4_start = N4T * (gl_WorkGroupID.x % n_tiles);
#else
  const uint n4_start = N4T * gl_WorkGroupID.x;
#endif
  const uint steps = (K4u + KQ - 1u) / KQ;
#ifdef SPLIT_K
  const uint q = (steps + K_SLICES - 1u) / K_SLICES;
  const uint s0 = min(k_slice * q, steps);
  const uint s1 = min(s0 + q, steps);
#else
  const uint s0 = 0u;
  const uint s1 = steps;
#endif

  ivec2 a_pos[A_PASSES];
  uint a_m[A_PASSES];
  uint a_kq[A_PASSES];
  bool a_ok[A_PASSES];
  [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
    const uint item = p * WG_SIZE + tid;
    a_ok[p] = A_ITEMS % WG_SIZE == 0u || item < A_ITEMS;
    const uint it = min(item, A_ITEMS - 1u);
    a_m[p] = it % WG_TILE_M;
    a_kq[p] = it / WG_TILE_M;
    // Rows past M read the last pixel; the output stage does not store them.
    const uint m = min(tile_m_start + a_m[p], M - 1u);
    a_pos[p] = ivec2(m % W, m / W);
#ifdef DIRECT
    a_pos[p] = a_pos[p] * ivec2(stride_w, stride_h) -
        ivec2(padding_w, padding_h);
#endif
  }
  uint b_k[B_PASSES];
  uint b_n4[B_PASSES];
  bool b_ok[B_PASSES];
  [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
    const uint item = p * WG_SIZE + tid;
    b_ok[p] = B_ITEMS % WG_SIZE == 0u || item < B_ITEMS;
    const uint it = min(item, B_ITEMS - 1u);
    b_n4[p] = it % N4T;
    b_k[p] = it / N4T;
  }

  vec4 temp_a[A_PASSES];
  vec4 temp_b[B_PASSES];
  vec4 acc[TM][TN4];
  [[unroll]] for (uint i = 0; i < TM; ++i) {
    [[unroll]] for (uint j = 0; j < TN4; ++j) {
      acc[i][j] = vec4(0);
    }
  }
  const uint tm0 = (tid / NT_N) * TM;
  const uint tn = tid % NT_N;

#define LOAD_STEP(step)                                                     \
  {                                                                         \
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {                        \
      temp_a[p] = fetch_in(a_pos[p], (step) * KQ + a_kq[p], K4u);           \
    }                                                                       \
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {                        \
      const uint k = (step) * WG_TILE_K + b_k[p];                           \
      const uint n4 = n4_start + b_n4[p];                                   \
      temp_b[p] = (k >> 2u) < K4u && n4 < N4                                \
          ? t_weight[((k >> 2u) * N4 + n4) * 4u + (k & 3u)]                  \
          : vec4(0);                                                        \
    }                                                                       \
  }

#define STORE_STEP(slice)                                                   \
  {                                                                         \
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {                        \
      if (a_ok[p]) {                                                        \
        const uint o = ((slice) * WG_TILE_K + 4u * a_kq[p]) * WG_TILE_M +   \
            a_m[p];                                                         \
        A_sh[o] = temp_a[p].x;                                              \
        A_sh[o + WG_TILE_M] = temp_a[p].y;                                  \
        A_sh[o + 2u * WG_TILE_M] = temp_a[p].z;                             \
        A_sh[o + 3u * WG_TILE_M] = temp_a[p].w;                             \
      }                                                                     \
    }                                                                       \
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {                        \
      if (b_ok[p]) {                                                        \
        B_sh[((slice) * WG_TILE_K + b_k[p]) * N4T + b_n4[p]] = temp_b[p];   \
      }                                                                     \
    }                                                                       \
  }

#define FMA_STEP(slice)                                                     \
  [[unroll]] for (uint k = 0; k < WG_TILE_K; ++k) {                         \
    const uint ab = ((slice) * WG_TILE_K + k) * WG_TILE_M + tm0;            \
    float a[TM];                                                            \
    [[unroll]] for (uint i = 0; i < TM; ++i) {                              \
      a[i] = A_sh[ab + i];                                                  \
    }                                                                       \
    [[unroll]] for (uint j = 0; j < TN4; ++j) {                             \
      const vec4 b = B_sh[((slice) * WG_TILE_K + k) * N4T + tn + j * NT_N]; \
      [[unroll]] for (uint i = 0; i < TM; ++i) {                            \
        acc[i][j] = fma(vec4(a[i]), b, acc[i][j]);                          \
      }                                                                     \
    }                                                                       \
  }

  if (s0 < s1) {
    LOAD_STEP(s0)
    STORE_STEP(0u)
    uint step;
    for (step = s0; step + 1u < s1; ++step) {
      memoryBarrierShared();
      barrier();
      LOAD_STEP(step + 1u)
      FMA_STEP((step - s0) % 2u)
      STORE_STEP((step - s0 + 1u) % 2u)
    }
    memoryBarrierShared();
    barrier();
    FMA_STEP((step - s0) % 2u)
  }

  [[unroll]] for (uint i = 0; i < TM; ++i) {
    const uint m = tile_m_start + tm0 + i;
    if (m < M) {
      [[unroll]] for (uint j = 0; j < TN4; ++j) {
        const uint n4 = n4_start + tn + j * NT_N;
        if (n4 < N4) {
#ifdef SPLIT_K
          t_scratch[(k_slice * M + m) * N4 + n4] = acc[i][j];
#else
          imageStore(
              t_out,
              ivec3(m % W, m / W, n4),
              epilogue_texel(acc[i][j], m, n4, W));
#endif
        }
      }
    }
  }
}

#endif
