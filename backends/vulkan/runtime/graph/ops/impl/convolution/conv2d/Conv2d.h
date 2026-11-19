/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/binary/BinaryOp.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/unary/UnaryOp.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vkcompute {

enum class Conv2dMethod : uint8_t {
  Depthwise,
  Pointwise,
  SlidingWindow,
  Transposed,
};

/*
 * Element-wise ops applied to the convolution output:
 *
 *   out = activation(residual_op(conv, residual * scale, residual_alpha))
 *
 * `residual` has the output's sizes and `scale` is a constant tensor with one
 * value per output channel at dim -3. Both are optional; `scale` requires a
 * residual. The activation is one of clamp, relu and leaky_relu.
 */
struct Conv2dEpilogue final {
  ValueRef residual = kDummyValueRef;
  ValueRef scale = kDummyValueRef;
  BinaryOpType residual_op = BinaryOpType::kAdd;
  float residual_alpha = 1.0f;
  std::optional<UnaryOpType> activation;
  float activation_a = 0.0f;
  float activation_b = 0.0f;

  bool has_residual() const {
    return is_valid(residual);
  }

  bool has_scale() const {
    return is_valid(scale);
  }

  bool has_activation() const {
    return activation.has_value();
  }
};

// Uniforms read by the epilogue of conv2d shaders.
struct Conv2dEpilogueParams final {
  float activation_a;
  float activation_b;
  float residual_alpha;
};

/*
 * Tensors bound to the residual and scale slots of conv2d shaders. Absent
 * operands are bound to the convolution input and packed bias, which have the
 * slots' storage types; the shaders never read them.
 */
struct Conv2dEpilogueArgs final {
  ValueRef residual;
  ValueRef scale;

  // `packed_scale` is the result of prepack_conv2d_residual_scale().
  Conv2dEpilogueArgs(
      const Conv2dEpilogue& epilogue,
      const ValueRef in,
      const ValueRef packed_bias,
      const ValueRef packed_scale);
};

// Prepacks the epilogue's residual scale like a bias. Returns kDummyValueRef
// if the epilogue has no scale.
ValueRef prepack_conv2d_residual_scale(
    ComputeGraph& graph,
    const Conv2dEpilogue& epilogue,
    const ValueRef weight_data,
    const bool transposed);

// Called by the resize functions of conv2d nodes, which pass the epilogue's
// residual (or kDummyValueRef) as their last resize arg. Checks that the
// residual still has the resized output's sizes.
void check_conv2d_residual_on_resize(
    ComputeGraph* graph,
    const ValueRef out,
    const std::vector<ValueRef>& resize_args);

// With `f16_epilogue`, for shaders that compute the epilogue in f16, the params
// are rounded to the nearest f16 so that the shaders convert them exactly. Some
// GPUs round f32 to f16 conversions toward zero.
Conv2dEpilogueParams create_conv2d_epilogue_params(
    const Conv2dEpilogue& epilogue,
    const bool f16_epilogue);

// Appends the has_activation, activation_op, has_residual, has_scale and
// residual_op specialization constants.
void append_conv2d_epilogue_spec_vars(
    vkapi::SpecVarList& spec_vars,
    const Conv2dEpilogue& epilogue);

// A variant of the conv2d_pw_coopmat shader (conv2d_pw_coopmat.yaml): its
// cooperative matrix shape, output tile (pixels x channels), K step, subgroup
// grid, required subgroup size, accumulator type, shared-memory layout, body
// options and K slices.
struct Conv2dPwCoopmatVariant final {
  // Body options. A variant with options is named after its parent, the
  // variant without its last option, plus the option suffixes in this order
  // (conv2d_pw_coopmat_option_suffix), before the dtype suffix.
  enum Option : uint32_t {
    // "_g<X>x<Y>": a subgroup grid other than the shipped rule's.
    kThreadGrid = 1u << 0,
    // "_al": no K and N bounds checks; K % tile_k == 0 and N % tile_n == 0.
    kAligned = 1u << 1,
    // "_w16": one 16-byte load of two adjacent k entries of a packed block.
    kWideWeightLoad = 1u << 2,
    // "_img": the input is read as a storage image (imageLoad).
    kImageInput = 1u << 3,
    // "_epi0": no bias, epilogue or scale in the output stage.
    kNoEpilogue = 1u << 4,
    // "_bcm": the B tile is staged column-major.
    kColumnMajorB = 1u << 5,
    // "_diag_once": diagnostic; the first K step's tiles are staged once and
    // multiplied on every K step.
    kDiagOnce = 1u << 6,
    // "_alc": no bounds checks in the K loop for any N; weight reads past the
    // last output channel are clamped. K % tile_k == 0.
    kAlignedClampN = 1u << 7,
  };

  const char* name;
  uint32_t mma_m;
  uint32_t mma_n;
  uint32_t mma_k;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t tile_k;
  uint32_t sg_grid_x;
  uint32_t sg_grid_y;
  uint32_t subgroup_size;
  bool fp32_accumulation;
  bool frag_layout;
  uint32_t options = 0u;
  // K slices of a split-K variant ("_sk<n>"), 1 otherwise.
  uint32_t k_slices = 1u;

  uint32_t wg_size() const {
    return sg_grid_x * sg_grid_y * subgroup_size;
  }

  bool has(const Option option) const {
    return (options & option) != 0u;
  }

  // Bytes of shared memory the shader declares.
  uint32_t shared_bytes() const;
};

// "_g<X>x<Y>", "_al", "_alc", "_w16", "_img", "_epi0", "_bcm", "_sk<n>",
// "_diag_once" for the variant's options and K slices, in this order.
std::string conv2d_pw_coopmat_option_suffix(
    const Conv2dPwCoopmatVariant& variant);

// Everything wrong with a variant table: duplicate names, a name that is not
// conv2d_pw_coopmat_<family>_t<M>x<N>_<acc><option suffix>_half, a variant
// with options whose parent (the name without the last suffix) is missing or
// differs in more than that option, and options that need another option.
std::vector<std::string> conv2d_pw_coopmat_registry_problems(
    const std::vector<Conv2dPwCoopmatVariant>& variants);

// The device limits a variant's work group and shared memory must fit.
struct Conv2dPwCoopmatLimits final {
  uint32_t max_invocations; // maxComputeWorkGroupInvocations
  uint32_t max_size_x; // maxComputeWorkGroupSize.x
  uint32_t max_subgroups; // maxComputeWorkgroupSubgroups
  uint32_t max_shared_bytes; // maxComputeSharedMemorySize
};

// The first of these limits the variant exceeds, named with its value.
std::optional<std::string> conv2d_pw_coopmat_limits_unsupported(
    const Conv2dPwCoopmatVariant& variant,
    const Conv2dPwCoopmatLimits& limits);

const std::vector<Conv2dPwCoopmatVariant>& conv2d_pw_coopmat_variants();

// The variant with this shader name, or nullptr.
const Conv2dPwCoopmatVariant* find_conv2d_pw_coopmat_variant(
    const std::string& name);

// Work group counts {channel tiles, pixel tiles in y, in z} of a dispatch of
// the variant for N output channels and M output pixels.
utils::uvec3 conv2d_pw_coopmat_workgroups(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const uint32_t N,
    const uint32_t M);

// The K steps [begin, end) of each slice of a split-K variant for K input
// channels (design D6): q = ceil(steps / k_slices), slice j covers
// [j q, min((j + 1) q, steps)); the variant is eligible when every slice
// holds at least two steps.
std::vector<std::pair<uint32_t, uint32_t>> conv2d_pw_coopmat_k_slices(
    const Conv2dPwCoopmatVariant& variant,
    const uint32_t K);

// The merge shader of a split-K variant (k_slices > 1).
std::string conv2d_pw_coopmat_merge_shader(
    const Conv2dPwCoopmatVariant& variant);

// The largest output plane (pixels) the output's texture can hold, for which
// a split-K variant sizes its scratch.
uint32_t conv2d_pw_coopmat_scratch_capacity(
    ComputeGraph& graph,
    const ValueRef out);

// Rows of N4 * 16 fp16 that the packed weight tensor of a split-K variant
// carries after the weights as scratch for M output pixels (0 otherwise).
uint32_t conv2d_pw_coopmat_scratch_rows(
    const Conv2dPwCoopmatVariant& variant,
    const uint32_t M);

// Whether the device enumerates the variant's fp16 cooperative matrix shape
// with its accumulator type at subgroup scope, can require its subgroup size
// for compute pipelines, and has room for its work group size and shared
// memory.
bool conv2d_pw_coopmat_device_supports(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant);

// The first device requirement of the variant that is not met, naming the
// device limit, or nullopt when conv2d_pw_coopmat_device_supports() holds.
std::optional<std::string> conv2d_pw_coopmat_device_unsupported(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant);

// Names the first axis whose work group count exceeds the device maximum.
std::optional<std::string> conv2d_pw_coopmat_grid_unsupported(
    const utils::uvec3& grid,
    const utils::uvec3& max_count);

// Storage the pointwise prepacking gives the packed weights of `weight_data`.
utils::StorageType conv2d_pw_weight_storage(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const ValueRef out);

// Why the convolution cannot run on the variant at its current sizes, with
// packed weights in `weight_storage`, or nullopt when it is eligible
// (conv2d_pw_coopmat_eligible is this with the packed weight's storage).
// `has_bias_or_epilogue`: the convolution has a bias, an activation, a
// residual or a scale, which an _epi0 variant cannot apply.
std::optional<std::string> conv2d_pw_coopmat_ineligible(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const utils::StorageType weight_storage,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed,
    const bool has_bias_or_epilogue);

// Whether a pointwise convolution may run on the variant at its current
// sizes: the device supports the variant, the convolution is FP16,
// non-transposed, ungrouped, stride 1, padding 0, batch 1, with more than one
// output pixel, texture input and output, and packed weights in a buffer (which
// needs float16_buffers_enabled()), and its work group counts fit the device.
bool conv2d_pw_coopmat_eligible(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed,
    const bool has_bias_or_epilogue);

// Kernel requested from conv2d_pw_impl. kAuto is the production pick and the
// only kind automatic routing uses. kTiledWide always picks the 2x2-pixel x
// 8-channel tile. kCoopmat picks `coopmat_variant`: conv2d_pw_impl throws if
// the convolution is not eligible for it when the graph is built; on a later
// resize to a shape it cannot run, the node takes the kAuto pick, re-evaluated
// on every resize.
struct Conv2dPwRequest final {
  enum class Kind : uint8_t {
    kAuto,
    kTiledWide,
    kCoopmat,
  };
  Kind kind = Kind::kAuto;
  const Conv2dPwCoopmatVariant* coopmat_variant = nullptr;
};

void conv2d_pw_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed_val,
    const Conv2dEpilogue& epilogue,
    const Conv2dPwRequest& request = {});

// `b1x1_override` forces or disables the 1x1 output tile of 3x3 and 5x5
// kernels instead of choosing it by GPU.
void conv2d_dw_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const Conv2dEpilogue& epilogue,
    const std::optional<bool> b1x1_override = std::nullopt);

void resize_conv2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

// Returns the epilogue described by the arguments of aten.convolution (none),
// conv_with_clamp (a clamp) or conv2d.with_epilogue, and checks that its
// operands are supported.
Conv2dEpilogue conv2d_epilogue_from_args(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

// Implements aten.convolution, conv_with_clamp and conv2d.with_epilogue for 4D
// inputs.
void conv2d_with_epilogue_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace vkcompute
