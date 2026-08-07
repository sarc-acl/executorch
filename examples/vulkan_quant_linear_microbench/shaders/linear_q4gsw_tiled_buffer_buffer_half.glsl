/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core


#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require


#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require


#define PRECISION highp
#define VEC4_T f16vec4
#define T float16_t

#define OUTPUT_BUFFER
#define INPUT_BUFFER
#define WEIGHT_BUFFER

#define TILE_N8 1

#define TILE_M4 1
#define TILE_K4 1
#define TILE_N4 2

#define TILE_M 4
#define TILE_K 4
#define TILE_N 8

layout(std430) buffer;

#include "common.glslh"


layout(set = 0, binding = 0) buffer PRECISION restrict writeonly t_outputBuffer {
    f16vec4 t_output[];
};


layout(set = 0, binding = 1) buffer PRECISION restrict readonly t_inputBuffer {
    f16vec4 t_input[];
};


layout(set = 0, binding = 2) buffer PRECISION restrict readonly t_packed_int4_weightBuffer {
    ivec4 t_packed_int4_weight[];
};


layout(set = 0, binding = 3) buffer PRECISION restrict readonly t_weight_scalesBuffer {
    f16vec4 t_weight_scales[];
};


layout(set = 0, binding = 4) buffer PRECISION restrict readonly t_biasBuffer {
    f16vec4 t_bias[];
};



layout(set = 0, binding = 5) uniform PRECISION restrict readonly output_sizes_UBO {
  ivec4 output_sizes;
};

layout(set = 0, binding = 6) uniform PRECISION restrict readonly input_sizes_UBO {
  ivec4 input_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int apply_bias = 0;
layout(constant_id = 4) const int K4_per_group = 0;

#include "linear_fp_input_tile_load.glslh"
#include "linear_int4_weight_tile_load.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_fp_bias_load.glslh"
#include "linear_fp_output_tile_fp_int4_compute.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "linear_fp_output_tile_store.glslh"

void main() {
  const int out_tile_x = int(gl_GlobalInvocationID.x);
  const int out_tile_y = int(gl_GlobalInvocationID.y);

  const int n = out_tile_x * TILE_N;
  const int m = out_tile_y * TILE_M;

  const int n8 = div_8(n);
  const int n4 = div_4(n);
  const int m4 = div_4(m);

  if (n >= output_sizes.x || m >= output_sizes.y) {
    return;
  }

  const int M = input_sizes.y;
  const int K4 = div_up_4(input_sizes.x);
  const int N4 = div_up_4(output_sizes.x); // number of texels in each row
  const int N8 = div_up_8(output_sizes.x); // number of texels in each row

  // VEC4_T out_texels[4][2];
  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  Int4WeightTile int4_weight_tile;

  FPPerOutChannelParams weight_scales_tile;
  FPPerOutChannelParams weight_zeros_tile;
  weight_zeros_tile.data[0] = VEC4_T(0.0);
  weight_zeros_tile.data[1] = VEC4_T(0.0);

  const int num_groups = K4 / K4_per_group;

  for (int group_i = 0; group_i < num_groups; ++group_i) {
    // Load quantization scales and zeros for the current group
    load_weight_scales_tile_for_group(weight_scales_tile, n4, group_i, N4);

    for (int k4_inner = 0; k4_inner < K4_per_group; k4_inner++) {
      const int k4 = group_i * K4_per_group + k4_inner;

      load_input_tile_no_checks(in_tile, k4, m, K4, M);
      load_int4_weight_tile(int4_weight_tile, k4, n8, K4);

      fp_accumulate_with_int4_weight(
          out_tile,
          in_tile,
          int4_weight_tile,
          weight_scales_tile,
          weight_zeros_tile);
    }
  }

  if (apply_bias > 0) {
    FPPerOutChannelParams bias_tile;
    load_bias_tile(bias_tile, n4);
    add_bias_to_out_tile(out_tile, bias_tile);
  }

  write_output_tile_with_checks(out_tile, n4, m, N4, M);
}
