/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * FP16 1x1 convolution (stride 1, padding 0, batch 1) as a cooperative-matrix
 * GEMM: out[m][n] = sum_k in[m][k] * w[n][k] + bias[n], with m the output
 * pixel, n the output channel and k the input channel.
 *
 * Bindings, uniforms, push constants and specialization constants are those of
 * conv2d_pw_tiled, so one dispatch node can pick either shader on resize. This
 * file is the template header; the body is conv2d_pw_coopmat_body.glslh.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

// Accumulator type, also the type of the bias add and the epilogue: fp32
// accumulation keeps fp32 until the output store.
$if ACC_TYPE == "f32":
  #define ACC_FP32
  #define ACC_T float
  #define VEC4_T vec4
$else:
  #define ACC_T float16_t
  #define VEC4_T ${buffer_gvec_type(DTYPE, 4)}

// The result tile is staged through shared memory in DRAIN_PASSES passes of
// DRAIN_G accumulator row blocks per subgroup, the fewest passes whose staged
// rows fit DRAIN_BUDGET_BYTES (32 KiB, so with the A/B pool every variant stays
// within the 48 KiB of the smallest measured device limit).
$DRAIN_BUDGET_BYTES = 32768
$ACC_BYTES = 4 if ACC_TYPE == "f32" else 2
$MMAS_M = WG_TILE_M // SG_GRID_Y // MMA_M
$DRAIN_G = max([g for g in (1, 2, 4, 8, 16) if MMAS_M % g == 0 and SG_GRID_Y * MMA_M * g * WG_TILE_N * ACC_BYTES <= DRAIN_BUDGET_BYTES])
#define DRAIN_BUDGET_BYTES ${DRAIN_BUDGET_BYTES}
#define DRAIN_G ${DRAIN_G}
#define DRAIN_PASSES ${MMAS_M // DRAIN_G}

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
$if IMG_INPUT:
  // The input as a storage image (imageLoad), for _img variants.
  ${layout_declare_image(B, "r", "t_in", DTYPE)}
$else:
  ${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
$if W16:
  // The packed weights as uvec4: one element is two adjacent k entries
  // (4 output channels each) of a 4OC x 4IC block.
  ${layout_declare_buffer(B, "r", "t_weight_u4", "uint", is_scalar_array=False)}
$else:
  ${layout_declare_tensor(B, "r", "t_weight_packed", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "texture2d")}
${layout_declare_tensor(B, "r", "t_residual", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_scale", DTYPE, "texture2d")}

${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

// The tiled shaders' block, then an unrounded copy of the epilogue params:
// fp32 accumulation computes the epilogue in fp32 and reads the copy, fp16
// accumulation reads the f16-rounded params as the tiled shaders do.
layout(push_constant) uniform restrict Block {
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
$if ACC_TYPE == "f32":
  float activation_a_f16;
  float activation_b_f16;
  float residual_alpha_f16;
  float activation_a;
  float activation_b;
  float residual_alpha;
$else:
  float activation_a;
  float activation_b;
  float residual_alpha;
  float activation_a_f32;
  float activation_b_f32;
  float residual_alpha_f32;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// stride_1_padding_0 is always 1 here; it is declared to keep the tiled
// shader's specialization constant IDs.
${layout_declare_spec_const(C, "int", "stride_1_padding_0", "0")}
${layout_declare_spec_const(C, "bool", "has_activation", "false")}
${layout_declare_spec_const(C, "int", "activation_op", "0")}
${layout_declare_spec_const(C, "bool", "has_residual", "false")}
${layout_declare_spec_const(C, "bool", "has_scale", "false")}
${layout_declare_spec_const(C, "int", "residual_op", "0")}

#include "conv2d_epilogue.glslh"


$if FRAG_LAYOUT:
  #define FRAG_LAYOUT
$if DIAG_ONCE:
  #define DIAG_ONCE
$if ALIGNED:
  #define ALIGNED
$if W16:
  #define W16
$if IMG_INPUT:
  #define IMG_INPUT
$if NO_EPILOGUE:
  #define NO_EPILOGUE
$if BCM:
  #define BCM

const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};

const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};

const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};

#include "conv2d_pw_coopmat_body.glslh"
