/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

// For input/output tensors


// For int8 input scales/zps


// For weight scales and bias

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require


#define PRECISION highp
#define VEC4_T vec4
#define T float

#define PACKED_INT8_INPUT_BUFFER

#define TILE_N8 1

#define TILE_M4 1
#define TILE_K4 1
#define TILE_N4 2

#define TILE_M 4
#define TILE_K 4
#define TILE_N 8

layout(std430) buffer;

#include "common.glslh"

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D t_output;
layout(set = 0, binding = 1) uniform PRECISION sampler3D t_input;

layout(set = 0, binding = 2) buffer PRECISION restrict readonly t_packed_int8_inputBuffer {
    ivec4 t_packed_int8_input[];
};


layout(set = 0, binding = 3) buffer PRECISION restrict readonly t_int8_input_sumsBuffer {
    ivec4 t_int8_input_sums[];
};

layout(set = 0, binding = 4) uniform PRECISION sampler3D t_int8_input_scales;
layout(set = 0, binding = 5) uniform PRECISION isampler3D t_int8_input_zps;
layout(set = 0, binding = 6) uniform PRECISION isampler2D t_packed_int4_weight;

layout(set = 0, binding = 7) buffer PRECISION restrict readonly t_weight_sumsBuffer {
    ivec4 t_weight_sums[];
};


layout(set = 0, binding = 8) buffer PRECISION restrict readonly t_weight_scalesBuffer {
    f16vec4 t_weight_scales[];
};


layout(set = 0, binding = 9) buffer PRECISION restrict readonly t_biasBuffer {
    f16vec4 t_bias[];
};



layout(set = 0, binding = 10) uniform PRECISION restrict readonly output_sizes_UBO {
  ivec4 output_sizes;
};

layout(set = 0, binding = 11) uniform PRECISION restrict readonly input_sizes_UBO {
  ivec4 input_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int apply_bias = 0;
layout(constant_id = 4) const int K4_per_group = 0;

#include "linear_fp_input_tile_load.glslh"
#include "linear_int8_input_tile_load.glslh"
#include "linear_int8_input_scales_zps_load.glslh"
#include "linear_int4_weight_tile_load.glslh"
#include "linear_int_weight_sums_load.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_int8_input_sums_load.glslh"
#include "linear_fp_output_tile_int8_int8_compute.glslh"
#include "linear_fp_output_tile_int8_int4_compute.glslh"
#include "linear_fp_output_tile_fp_int4_compute.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "linear_fp_output_tile_store.glslh"
#include "linear_fp_bias_load.glslh"

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
  const int M4 = div_up_4(M);
  const int N4 = div_up_4(output_sizes.x);
  const int N8 = div_up_8(output_sizes.x);

  FPOutTile out_tile;
  initialize(out_tile);

  Int32Accum out_accum;
  initialize(out_accum);

  Int8InputTile int8_in_tile;
  Int4WeightTile int4_weight_tile;

  Int8InputScales input_scales;
  Int8InputZeroPoints input_zps;
  load_int8_input_scales_and_zps(input_scales, input_zps, m4);

  FPPerOutChannelParams weight_scales_tile;
  IntPerOutChannelParams weight_sums_tile;

  IntPerInChannelParams int8_input_sums_tile;

  const int num_groups = K4 / K4_per_group;
  const int group_size = mul_4(K4_per_group);

  for (int group_i = 0; group_i < num_groups; ++group_i) {
    // Reset int accumulator
    initialize(out_accum);
    for (int k4_inner = 0; k4_inner < K4_per_group; k4_inner++) {
      const int k4 = group_i * K4_per_group + k4_inner;

      load_int8_input_tile(int8_in_tile, k4, m4, K4);
      load_int4_weight_tile(int4_weight_tile, k4, n8, K4);

      int_accumulate_with_int4_weight(
          out_accum, int8_in_tile, int4_weight_tile);
    }

    load_weight_scales_tile_for_group(weight_scales_tile, n4, group_i, N4);
    load_weight_sums_tile_for_group(weight_sums_tile, n4, group_i, N4);
    load_int8_input_sums_tile_for_group(int8_input_sums_tile, m4, group_i, M4);

    accumulate_out_tile_with_int_accum_from_int4_weights(
        out_tile,
        out_accum,
        int8_input_sums_tile,
        input_scales,
        input_zps,
        weight_sums_tile,
        weight_scales_tile,
        group_size);
  }

  if (apply_bias > 0) {
    FPPerOutChannelParams bias_tile;
    load_bias_tile(bias_tile, n4);
    add_bias_to_out_tile(out_tile, bias_tile);
  }

  write_output_tile_with_checks(out_tile, n4, m, N4, M);
}
