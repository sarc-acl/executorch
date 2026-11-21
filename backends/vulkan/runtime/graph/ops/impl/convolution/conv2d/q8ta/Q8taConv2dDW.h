/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>

namespace vkcompute {

ValueRef prepack_quantized_conv2d_dw_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef weight_data,
    const ValueRef kernel_size);

void add_q8ta_conv2d_dw_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const uint32_t activation_type,
    const ValueRef packed_int8_output,
    // Output widths per thread: 4 (stock), 2 or 1 (q8ta_conv2d_dw_w2 / _w1).
    const uint32_t tile_w = 4u);

// q8ta_conv2d_dw with the given width tile (the benchmark's dw_tile route);
// the operator takes 4 unless an int8 table row selects q8ta_conv2d_dw_w2 or
// q8ta_conv2d_dw_w1 with ETVK_CONV2D_COOPMAT=1.
void q8ta_conv2d_dw_tiled_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const uint32_t tile_w);

void add_conv2d_dw_q8ta_q8csw_q8to_4w4c_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef packed_int8_output);

} // namespace vkcompute
