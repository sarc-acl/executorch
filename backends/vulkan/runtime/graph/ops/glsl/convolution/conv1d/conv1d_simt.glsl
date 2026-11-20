/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * conv1d as a register-tiled GEMM with shared-memory staging and float32
 * accumulation (change conv2d-all-used-kernels, the data path of
 * conv2d_simt.glsl). Input and output tensors are [N, C, L] textures with the
 * standard axis map.
 *
 * PW (conv1d_pw role, height-packed: texel (l, c4, n) holds 4 channels): rows
 * m = n L + l of every batch, K = input channels, columns = output channel
 * texels; A item = input texel (l, k4, n), B = the packed weights of
 * pack_fp_linear_weight ([k4][n4][4] vec4 entries), output texel (l, n4, n).
 *
 * Otherwise (conv1d role, channels-packed: texel (l, c, n4) holds 4 batches)
 * the GEMM is transposed so a column is one output texel: rows m = output
 * channels, columns j = n4 L_out + l, K index k = t C_in + ci over the kernel
 * taps t; A = the weights as [K4][C_out] vec4 (4 consecutive k of channel m),
 * B entry (k, j) = input texel (l stride - padding + t dilation, ci, n4),
 * zero outside the input; output texel (l, m, n4).
 *
 * Bias and the clamp of conv_with_clamp after the sum. K_SLICES > 1: split K
 * over work groups with a float32 scratch and the MERGE shader.
 */

#version 450 core

#extension GL_EXT_control_flow_attributes : require

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}

${define_required_extensions("texture3d", DTYPE)}
${define_required_extensions("buffer", DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=True)}
$if K_SLICES > 1 or MERGE:
  ${layout_declare_tensor(B, "rw", "t_scratch", "float", "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(push_constant) uniform restrict Block {
  int K4;
  int stride;
  int padding;
  int dilation;
  int taps;
  float out_min;
  float out_max;
  int k_slices;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "bool", "has_bias", "true")}
${layout_declare_spec_const(C, "bool", "has_clamp", "false")}

$if PW:
  #define PW
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

// GEMM extents: rows M, column texels N4.
#ifdef PW
#define GEMM_M (uint(out_sizes.x) * uint(out_sizes.z))
#define GEMM_N4 uint(div_up_4(out_sizes.y))
#else
#define GEMM_M uint(out_sizes.y)
#define GEMM_N4 (uint(out_sizes.x) * uint(div_up_4(out_sizes.z)))
#endif

void store_out(vec4 v, const uint m, const uint n4) {
#ifdef PW
  const uint L = uint(out_sizes.x);
  const ivec3 pos = ivec3(m % L, n4, m / L);
  if (has_bias) {
    const uint C = uint(out_sizes.y);
    const uint c = 4u * n4;
    v += vec4(
        float(t_bias[c]),
        c + 1u < C ? float(t_bias[c + 1u]) : 0.0,
        c + 2u < C ? float(t_bias[c + 2u]) : 0.0,
        c + 3u < C ? float(t_bias[c + 3u]) : 0.0);
  }
#else
  const uint L = uint(out_sizes.x);
  const ivec3 pos = ivec3(n4 % L, m, n4 / L);
  if (has_bias) {
    v += vec4(float(t_bias[m]));
  }
#endif
  if (has_clamp) {
    v = clamp(v, vec4(out_min), vec4(out_max));
  }
  imageStore(t_out, pos, VEC4_T(v));
}

#ifdef MERGE

void main() {
  const uint M = GEMM_M;
  const uint N4 = GEMM_N4;
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
  store_out(v, m, n4);
}

#else

const uint NT_N = WG_TILE_N / (4u * TN4);
const uint NT_M = WG_TILE_M / TM;
const uint WG_SIZE = NT_M * NT_N;
const uint KQ = WG_TILE_K / 4u;
const uint N4T = WG_TILE_N / 4u;

shared float A_sh[2u * WG_TILE_K * WG_TILE_M];
shared vec4 B_sh[2u * WG_TILE_K * N4T];

const uint A_ITEMS = WG_TILE_M * KQ;
const uint A_PASSES = (A_ITEMS + WG_SIZE - 1u) / WG_SIZE;
const uint B_ITEMS = WG_TILE_K * N4T;
const uint B_PASSES = (B_ITEMS + WG_SIZE - 1u) / WG_SIZE;

// 4 K values (k4) of row m.
vec4 fetch_a(const uint m, const uint k4) {
  if (k4 >= uint(K4)) {
    return vec4(0);
  }
#ifdef PW
  const uint L = uint(out_sizes.x);
  return vec4(texelFetch(t_in, ivec3(m % L, k4, m / L), 0));
#else
  return vec4(t_weight[k4 * uint(out_sizes.y) + m]);
#endif
}

// The 4 columns of column texel n4 at K index k.
vec4 fetch_b(const uint k, const uint n4) {
#ifdef PW
  const uint N4 = uint(div_up_4(out_sizes.y));
  return (k >> 2u) < uint(K4) && n4 < N4
      ? vec4(t_weight[((k >> 2u) * N4 + n4) * 4u + (k & 3u)])
      : vec4(0);
#else
  const uint L = uint(out_sizes.x);
  const uint C_in = uint(in_sizes.y);
  const uint t = k / C_in;
  const uint ci = k - t * C_in;
  const int l = int(n4 % L) * stride - padding + int(t) * dilation;
  const bool inside = t < uint(taps) && uint(l) < uint(in_sizes.x) &&
      n4 < GEMM_N4;
  const vec4 v = vec4(texelFetch(
      t_in, ivec3(inside ? l : 0, inside ? int(ci) : 0, int(n4 / L)), 0));
  return inside ? v : vec4(0);
#endif
}

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint M = GEMM_M;
  const uint N4 = GEMM_N4;
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

  uint a_m[A_PASSES];
  uint a_row[A_PASSES];
  uint a_kq[A_PASSES];
  bool a_ok[A_PASSES];
  [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
    const uint item = p * WG_SIZE + tid;
    a_ok[p] = A_ITEMS % WG_SIZE == 0u || item < A_ITEMS;
    const uint it = min(item, A_ITEMS - 1u);
    a_m[p] = it % WG_TILE_M;
    a_kq[p] = it / WG_TILE_M;
    a_row[p] = min(tile_m_start + a_m[p], M - 1u);
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
      temp_a[p] = fetch_a(a_row[p], (step) * KQ + a_kq[p]);                 \
    }                                                                       \
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {                        \
      temp_b[p] =                                                           \
          fetch_b((step) * WG_TILE_K + b_k[p], n4_start + b_n4[p]);         \
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
          store_out(acc[i][j], m, n4);
#endif
        }
      }
    }
  }
}

#endif
