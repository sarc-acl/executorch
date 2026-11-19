// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>

#include <algorithm>

namespace vkcompute::shader_profiling::reference {
namespace {

/*
 * Reference for raw shader cases of the pointwise convolution shaders
 * (conv2d_pw_tiled*_bufw, conv2d_pw_coopmat*), stride 1, padding 0, no
 * epilogue. Tensor bindings, in order: output [1, N, H, W], input [1, K, H, W],
 * the 4OC x 4IC packed weight buffer [K4, N4 * 16] as written by
 * pack_fp_linear_weight, and the bias [1, N]. Later tensor bindings (residual,
 * scale) are ignored.
 */
Tensor conv2d_pw_packed_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  std::vector<std::string> bindings;
  for (const Json& binding : spec.at("bindings")) {
    if (binding.contains("tensor")) {
      bindings.push_back(binding.at("tensor"));
    }
  }
  require(
      bindings.size() >= 4,
      "conv2d_pw_packed requires output, input, packed weight and bias bindings");
  const Tensor& in = tensors.at(bindings[1]);
  const Tensor& w = tensors.at(bindings[2]);
  const Tensor& bias = tensors.at(bindings[3]);
  const auto out_sizes = spec.at("tensors")
                             .at(bindings[0])
                             .at("sizes")
                             .get<std::vector<int64_t>>();
  require(
      in.sizes.size() == 4 && out_sizes.size() == 4 && in.sizes[0] == 1 &&
          out_sizes[0] == 1 && in.sizes[2] == out_sizes[2] &&
          in.sizes[3] == out_sizes[3],
      "conv2d_pw_packed requires batch 1 and equal input and output planes");
  const int64_t K = in.sizes[1];
  const int64_t N = out_sizes[1];
  const int64_t plane = out_sizes[2] * out_sizes[3];
  const int64_t K4 = (K + 3) / 4;
  const int64_t N4 = (N + 3) / 4;
  require(
      w.sizes == std::vector<int64_t>({K4, N4 * 16}),
      "conv2d_pw_packed requires a [K4, N4 * 16] packed weight");
  require(
      static_cast<int64_t>(bias.values.size()) == N,
      "conv2d_pw_packed requires N bias values");

  Tensor output{out_sizes, std::vector<double>(N * plane)};
  parallel_for(N, threads, [&](size_t n_index) {
    const int64_t n = static_cast<int64_t>(n_index);
    for (int64_t p = 0; p < plane; ++p) {
      double acc = 0;
      for (int64_t k = 0; k < K; ++k) {
        const int64_t packed = (((k / 4) * N4 + n / 4) * 4 + k % 4) * 4 + n % 4;
        acc += in.values[k * plane + p] * w.values[packed];
      }
      output.values[n * plane + p] = acc + bias.values[n];
    }
  });
  return output;
}

/*
 * Reference for raw shader cases that check where a pointwise shader stores:
 * the bindings of conv2d_pw_packed, with an output texture larger than the
 * convolution's output (the shader's out_sizes, passed as a uniform, say
 * [W, H, N]; H and W are the input's, N the bias's). The expected output is
 * the output's initial content with the convolution's [N, H, W] region
 * replaced by the convolution, so a store outside that region fails.
 */
Tensor conv2d_pw_packed_guard_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  std::vector<std::string> bindings;
  for (const Json& binding : spec.at("bindings")) {
    if (binding.contains("tensor")) {
      bindings.push_back(binding.at("tensor"));
    }
  }
  require(
      bindings.size() >= 4,
      "conv2d_pw_packed_guard requires output, input, weight and bias bindings");
  const auto found = tensors.find(bindings[0]);
  require(
      found != tensors.end(),
      "conv2d_pw_packed_guard requires initial output data");
  Tensor output = found->second;
  const Tensor& in = tensors.at(bindings[1]);
  const Tensor& w = tensors.at(bindings[2]);
  const Tensor& bias = tensors.at(bindings[3]);
  require(
      in.sizes.size() == 4 && output.sizes.size() == 4,
      "conv2d_pw_packed_guard requires 4-D input and output");
  const int64_t K = in.sizes[1];
  const int64_t H = in.sizes[2];
  const int64_t W = in.sizes[3];
  const int64_t N = static_cast<int64_t>(bias.values.size());
  const int64_t Ht = output.sizes[2];
  const int64_t Wt = output.sizes[3];
  require(
      output.sizes[1] >= N && Ht >= H && Wt == W,
      "conv2d_pw_packed_guard requires an output at least [N, H, W] wide W");
  const int64_t K4 = (K + 3) / 4;
  const int64_t N4 = (N + 3) / 4;
  require(
      w.sizes == std::vector<int64_t>({K4, N4 * 16}),
      "conv2d_pw_packed_guard requires a [K4, N4 * 16] packed weight");
  parallel_for(N, threads, [&](size_t n_index) {
    const int64_t n = static_cast<int64_t>(n_index);
    for (int64_t y = 0; y < H; ++y) {
      for (int64_t x = 0; x < W; ++x) {
        double acc = 0;
        for (int64_t k = 0; k < K; ++k) {
          const int64_t packed =
              (((k / 4) * N4 + n / 4) * 4 + k % 4) * 4 + n % 4;
          acc += in.values[(k * H + y) * W + x] * w.values[packed];
        }
        output.values[(n * Ht + y) * Wt + x] = acc + bias.values[n];
      }
    }
  });
  return output;
}

/*
 * Reference of the _diag_once diagnostics of conv2d_pw_coopmat for
 * aten.convolution.default cases with an optional bias and no epilogue: each
 * output is steps times the products of the first K step's input channels,
 * plus the bias, with steps = ceil(K / tile_k) of the variant the case's route
 * names. Its value depends on K and on the data, so a passing diagnostic row
 * shows that every K step ran.
 */
Tensor conv2d_pw_diag_once_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Json& route = spec.at("route");
  require(
      route.is_object() && route.contains("options") &&
          route.at("options").contains("variant"),
      "conv2d_pw_diag_once requires a pw_coopmat route with a variant");
  const std::string name = route.at("options").at("variant");
  const Conv2dPwCoopmatVariant* const variant =
      find_conv2d_pw_coopmat_variant(name);
  require(
      variant != nullptr && variant->has(Conv2dPwCoopmatVariant::kDiagOnce),
      "conv2d_pw_diag_once requires a _diag_once variant, not " + name);
  require(
      spec.at("operator") == "aten.convolution.default",
      "conv2d_pw_diag_once supports aten.convolution.default only");
  const Tensor& in = tensor_arg(spec, 0, tensors);
  const Tensor& w = tensor_arg(spec, 1, tensors);
  const Tensor* bias =
      spec.at("args").at(2).is_null() ? nullptr : &tensor_arg(spec, 2, tensors);
  const std::string out_name = spec.at("args").back().at("tensor");
  const auto out_sizes =
      spec.at("tensors").at(out_name).at("sizes").get<std::vector<int64_t>>();
  require(
      in.sizes.size() == 4 && w.sizes.size() == 4 && out_sizes.size() == 4 &&
          in.sizes[0] == 1 && w.sizes[1] == in.sizes[1] && w.sizes[2] == 1 &&
          w.sizes[3] == 1 && out_sizes[1] == w.sizes[0],
      "conv2d_pw_diag_once requires a batch-1 1x1 convolution");
  const int64_t K = in.sizes[1];
  const int64_t N = w.sizes[0];
  const int64_t plane = in.sizes[2] * in.sizes[3];
  const int64_t tile_k = variant->tile_k;
  const int64_t steps = (K + tile_k - 1) / tile_k;
  const int64_t first = std::min(tile_k, K);
  Tensor output{out_sizes, std::vector<double>(N * plane)};
  parallel_for(N, threads, [&](size_t n_index) {
    const int64_t n = static_cast<int64_t>(n_index);
    for (int64_t p = 0; p < plane; ++p) {
      double acc = 0;
      for (int64_t k = 0; k < first; ++k) {
        acc += in.values[k * plane + p] * w.values[n * K + k];
      }
      output.values[n * plane + p] =
          double(steps) * acc + (bias ? bias->values[n] : 0.0);
    }
  });
  return output;
}

} // namespace

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(conv2d_pw_packed, conv2d_pw_packed_reference);
  REGISTER_REFERENCE_FN(conv2d_pw_diag_once, conv2d_pw_diag_once_reference);
  REGISTER_REFERENCE_FN(
      conv2d_pw_packed_guard,
      conv2d_pw_packed_guard_reference);
}

} // namespace vkcompute::shader_profiling::reference
