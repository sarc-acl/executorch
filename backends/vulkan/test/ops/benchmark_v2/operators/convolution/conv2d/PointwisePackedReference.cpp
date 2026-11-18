// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>

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

} // namespace

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(conv2d_pw_packed, conv2d_pw_packed_reference);
}

} // namespace vkcompute::shader_profiling::reference
