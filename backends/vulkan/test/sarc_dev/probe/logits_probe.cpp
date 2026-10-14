/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// logits_probe <model.pte> <prompts.txt> <out.bin>
//
// SARC development zone. prompts.txt holds one prompt per line as
// space-separated token ids. Each prompt is one prefill call,
// forward(tokens [1, n], input_pos [0]), exactly as llama_main issues it; the
// logits of the last position are appended to out.bin as fp32 [vocab]. Kernel
// selection comes from the environment, as for llama_main.

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;

namespace {

float half_to_float(const uint16_t h) {
  const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
  float v;
  if (exp == 0) {
    v = std::ldexp(static_cast<float>(man), -24);
  } else if (exp == 31) {
    v = man == 0 ? INFINITY : NAN;
  } else {
    v = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
  }
  return sign ? -v : v;
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <model.pte> <prompts.txt> <out.bin>\n", argv[0]);
    return 2;
  }
  Module module(argv[1], Module::LoadMode::MmapUseMlockIgnoreErrors);
  if (module.load_method("forward") != Error::Ok) {
    std::fprintf(stderr, "cannot load forward from %s\n", argv[1]);
    return 1;
  }
  std::ifstream prompts(argv[2]);
  FILE* out = std::fopen(argv[3], "wb");
  if (!prompts || out == nullptr) {
    std::fprintf(stderr, "cannot open %s or %s\n", argv[2], argv[3]);
    return 1;
  }
  std::string line;
  int index = 0;
  while (std::getline(prompts, line)) {
    std::vector<int64_t> ids;
    std::istringstream fields(line);
    for (int64_t id; fields >> id;) {
      ids.push_back(id);
    }
    if (ids.empty()) {
      continue;
    }
    int64_t pos = 0;
    auto tokens = from_blob(
        ids.data(), {1, static_cast<int>(ids.size())}, ScalarType::Long);
    auto start_pos = from_blob(&pos, {1}, ScalarType::Long);
    const auto result = module.forward({tokens, start_pos});
    if (!result.ok()) {
      std::fprintf(
          stderr, "prompt %d: forward failed, error %d\n", index,
          static_cast<int>(result.error()));
      return 1;
    }
    const auto logits = result->at(0).toTensor();
    const size_t vocab = static_cast<size_t>(logits.size(logits.dim() - 1));
    const size_t first = static_cast<size_t>(logits.numel()) - vocab;
    std::vector<float> last(vocab);
    if (logits.scalar_type() == ScalarType::Float) {
      std::memcpy(
          last.data(), logits.const_data_ptr<float>() + first,
          vocab * sizeof(float));
    } else if (logits.scalar_type() == ScalarType::Half) {
      const uint16_t* h =
          static_cast<const uint16_t*>(logits.const_data_ptr()) + first;
      std::transform(h, h + vocab, last.begin(), half_to_float);
    } else {
      std::fprintf(stderr, "prompt %d: unexpected logits dtype\n", index);
      return 1;
    }
    std::fwrite(last.data(), sizeof(float), vocab, out);
    const size_t top = static_cast<size_t>(
        std::max_element(last.begin(), last.end()) - last.begin());
    std::printf(
        "prompt %d tokens %zu vocab %zu top %zu logit %.6f\n", index, ids.size(),
        vocab, top, static_cast<double>(last[top]));
    std::fflush(stdout);
    ++index;
  }
  std::fclose(out);
  return 0;
}
