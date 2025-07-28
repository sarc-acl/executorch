/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

// Define tile sizes for optimal memory access patterns
#define TILE_SIZE_M 8   // Tile size for output rows
#define TILE_SIZE_N 8   // Tile size for output columns
#define TILE_SIZE_K 8   // Tile size for K dimension

$if MAT2_IS_TRANSPOSED:
  #define MAT2_IS_TRANSPOSED

$if HAS_BIAS:
  #define HAS_BIAS

${layout_declare_tensor(B, "w", "out_tensor", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "mat1_tensor", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "mat2_tensor", DTYPE, "texture3d")}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "bias_tensor", DTYPE, "texture3d")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "mat2_sizes")}
$if HAS_BIAS:
  ${layout_declare_ubo(B, "ivec4", "bias_sizes")}
  ${layout_declare_ubo(B, "float", "alpha", "float", "beta")}

#include "indexing_utils.h"

// Workgroup size matches tile dimensions
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int out_packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "mat1_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 mat1_axis_map = unhash_axis_map(mat1_layout);
const lowp int mat1_packed_dim = unhash_packed_dim(mat1_layout);

${layout_declare_spec_const(C, "int", "mat2_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 mat2_axis_map = unhash_axis_map(mat2_layout);
const lowp int mat2_packed_dim = unhash_packed_dim(mat2_layout);

$if HAS_BIAS:
  ${layout_declare_spec_const(C, "int", "bias_layout", "DEFAULT_LAYOUT")}
  const lowp ivec4 bias_axis_map = unhash_axis_map(bias_layout);
  const lowp int bias_packed_dim = unhash_packed_dim(bias_layout);

// Shared memory for tiling - use smaller tiles to avoid memory issues
shared vec4 mat1_tile[TILE_SIZE_M][TILE_SIZE_K];
shared vec4 mat2_tile[TILE_SIZE_K][TILE_SIZE_N];
shared float out_tile[TILE_SIZE_M][TILE_SIZE_N];

#ifdef HAS_BIAS
vec4 get_bias_texel_W_packed(ivec3 logical_pos) {
  ivec3 bias_pos = ivec3(0);
  if (bias_sizes.y == 1) {
    bias_pos[bias_axis_map.y] = 0;
  } else {
    bias_pos[bias_axis_map.y] = logical_pos.y;
  }
  if (bias_sizes.x == 1) {
    bias_pos[bias_axis_map.x] = 0;
    vec4 bias_texel = texelFetch(bias_tensor, bias_pos, 0);
    // Only the first value is valid, the rest is 0 padding
    return vec4(bias_texel.x);
  } else {
    bias_pos[bias_axis_map.x] = logical_pos.x;
  }

  return texelFetch(bias_tensor, bias_pos, 0);
}
#endif // HAS_BIAS

void matmul_tiled_k_dim_packed(ivec3 lpos) {
  // Get local thread ID and workgroup size
  const uint local_idx = gl_LocalInvocationID.x;
  const uint local_idy = gl_LocalInvocationID.y;
  const uint workgroup_size_x = gl_WorkGroupSize.x;
  const uint workgroup_size_y = gl_WorkGroupSize.y;
  const uint workgroup_id_x = gl_WorkGroupID.x;
  const uint workgroup_id_y = gl_WorkGroupID.y;
  
  // Initialize position for reading from mat1
  ivec3 mat1_pos;
  mat1_pos[mat1_axis_map.x] = 0;
  mat1_pos[mat1_axis_map.y] = lpos.y;
  mat1_pos[mat1_axis_map.z] = lpos.z;
#ifdef MAT2_IS_TRANSPOSED
  const int mat2_k_axis = mat2_axis_map.x;
  const int mat2_row_axis = mat2_axis_map.y;
#else
  const int mat2_k_axis = mat2_axis_map.y;
  const int mat2_row_axis = mat2_axis_map.x;
#endif // MAT2_IS_TRANSPOSED

  // Initialize position for reading from mat2
  ivec3 mat2_pos;
  mat2_pos[mat2_k_axis] = 0;
  mat2_pos[mat2_row_axis] = lpos.x;
#ifndef MAT2_IS_TRANSPOSED
  mat2_pos[mat2_axis_map.z] = lpos.z;
#else
  mat2_pos[mat2_axis_map.z] = 0;
#endif // MAT2_IS_TRANSPOSED

  float sum = 0;
  const int K = divup4(mat1_sizes.x);

  // Process K dimension in chunks that fit in shared memory
  const int chunk_size = min(TILE_SIZE_K, K);
  const int num_chunks = (K + chunk_size - 1) / chunk_size;

  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    // Calculate start position for this chunk
    const int k_start = chunk * chunk_size;
    const int k_end = min(k_start + chunk_size, K);

    // Load mat1 data into shared memory
    int k_idx = k_start + int(local_idx);
    int row_idx = mat1_pos[mat1_axis_map.y];
    if (k_idx < mat1_sizes[mat1_axis_map.x] && row_idx < mat1_sizes[mat1_axis_map.y]) {
      ivec3 pos = mat1_pos;
      pos[mat1_axis_map.x] = k_idx;
      mat1_tile[local_idx][local_idy] = texelFetch(mat1_tensor, pos, 0);
    }
    else {
      mat1_tile[local_idx][local_idy] = vec4(0.0);
    }

    // Load mat2 data into shared memory
    k_idx = k_start + int(local_idy);
    int col_idx = mat2_pos[mat2_row_axis];
    if (col_idx < mat2_sizes[mat2_row_axis] && k_idx < mat2_sizes[mat2_k_axis]) {
      ivec3 pos = mat2_pos;
      pos[mat2_k_axis] = k_idx;
      mat2_tile[local_idy][local_idx] = texelFetch(mat2_tensor, pos, 0);
    }
    else {
      mat2_tile[local_idy][local_idx] = vec4(0.0);
    }

    // Ensure all threads finish loading before computation
    barrier();

    // Compute
    for (int i = 0; i < k_end - k_start; ++i) {
      const vec4 mat1_tex = mat1_tile[i][local_idy];
      const vec4 mat2_tex = mat2_tile[i][local_idx];

      sum += dot(mat1_tex, mat2_tex);
    }

    // Ensure all threads finish using shared memory before next chunk
    if (chunk < num_chunks - 1) {
      barrier();
    }
  }

  // Because the out matrix is M x N/4, we need to use out_tile
  // to grab the out texels of other threads and condense into vec4s
  out_tile[local_idy][local_idx] = sum;

  barrier();

  if (local_idx%4 == 0) {

    vec4 texel = vec4(out_tile[local_idy][local_idx + 0],
                      out_tile[local_idy][local_idx + 1],
                      out_tile[local_idy][local_idx + 2],
                      out_tile[local_idy][local_idx + 3]);
    lpos.x /= 4;
#ifdef HAS_BIAS
    vec4 bias_texel = get_bias_texel_W_packed(lpos);
    texel = beta * bias_texel + alpha * texel;
#endif // HAS_BIAS

    write_texel_lpos(out_tensor, lpos, texel, out_axis_map);
  }
}

vec4 matmul_naive_k_dim_packed_row_dim_packed(const ivec3 out_lpos) {
  ivec3 mat1_pos;
  mat1_pos[mat1_axis_map.x] = 0;
  mat1_pos[mat1_axis_map.y] = out_lpos.y;
  mat1_pos[mat1_axis_map.z] = out_lpos.z;

  ivec3 mat2_pos;
  mat2_pos[mat2_axis_map.x] = out_lpos.x;
  mat2_pos[mat2_axis_map.y] = 0;
  mat2_pos[mat2_axis_map.z] = out_lpos.z;

  ivec3 mat2_pos_offset = ivec3(0);
  mat2_pos_offset[mat2_axis_map.y] = 1;

  const int mat2_y_axis = mat2_axis_map.y;

  vec4 texel = vec4(0);
  const int K = divup4(mat1_sizes.x);

  for (int i = 0;
       i < K;
       ++i, mat1_pos[mat1_axis_map.x]++, mat2_pos[mat2_axis_map.y]+=4) {
    const vec4 mat1_tex = texelFetch(mat1_tensor, mat1_pos, 0);

    for (int r = 0; r < 4; ++r) {
      // On-demand construction of mat2_pos appears to provide the lowest
      // latency. Surprisingly, this doesn't translate to mat1_pos.
      ivec3 mat2_pos = ivec3(0);
      mat2_pos[mat2_axis_map.x] = out_lpos.x;
      mat2_pos[mat2_axis_map.y] = 4 * i + r;
      mat2_pos[mat2_axis_map.z] = out_lpos.z;

      vec4 mat1_comp_vec = vec4(mat1_tex[r]);
      texel = fma(mat1_comp_vec, texelFetch(mat2_tensor, mat2_pos, 0), texel);
    }
  }

  return texel;
}

void main() {
  ivec3 out_lpos = ivec3(gl_GlobalInvocationID);

  vec4 texel = vec4(0);

#ifdef MAT2_IS_TRANSPOSED
  if (mat2_packed_dim == W_DIM) {
    matmul_tiled_k_dim_packed(out_lpos);
    return;
  } else {
    if (any(greaterThanEqual(out_lpos, out_limits))) {
      return;
    }
    texel = matmul_naive_k_dim_packed_row_dim_packed(out_lpos);
  }
#else
  if (mat2_packed_dim == W_DIM) {
    if (any(greaterThanEqual(out_lpos, out_limits))) {
      return;
    }
    texel = matmul_naive_k_dim_packed_row_dim_packed(out_lpos);
  } else {
    matmul_tiled_k_dim_packed(out_lpos);
    return;
  }
#endif // MAT2_IS_TRANSPOSED

#ifdef HAS_BIAS
    vec4 bias_texel = get_bias_texel_W_packed(out_lpos);
    texel = beta * bias_texel + alpha * texel;
#endif // HAS_BIAS

    write_texel_lpos(out_tensor, out_lpos, texel, out_axis_map);
}
