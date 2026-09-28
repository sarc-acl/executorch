/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC development zone: environment-variable overrides for kernel selection
// and the q4gsw sweep candidates. Not part of a release (make-release.sh drops
// every sarc_dev/ directory), so a release build has no env-var behaviour.
//
//   ET_VK_SARC_UNVERIFIED=1           also use rows marked kUnverified
//   ET_VK_FORCE_TILED_LINEAR=1        SARC linear ops use their non-SARC
//                                     fallback kernel (the tiled baseline;
//                                     SDPA stays as selected, as on 1.4)
//   ET_VK_DISABLE_COOPMAT=1           SARC SDPA ops use the upstream kernels
//   ET_VK_SARC_Q4GSW_VARIANT=<tile>   use this sweep/release tile for 4w
//                                     prefill wherever it fits, e.g.
//                                     t128x128k32g24s32f32c; builds 4w on the
//                                     SARC path even on devices without rows
//   ET_VK_SARC_DQ8CA_VARIANT=<tile>   same for 8da4w prefill (dq8ca sweep
//                                     candidates below, then release rows)

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace vkcompute {
namespace sarc {
namespace {

bool env_true(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::strcmp(v, "0") != 0 &&
      std::strcmp(v, "false") != 0 && std::strcmp(v, "off") != 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
      s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Sweep candidates: glsl/sarc_dev/sarc_linear_q4gsw_coopmat_sweep.yaml.
// Keep in sync with that yaml (test_sarc_select checks the names exist).
constexpr TileDims tile(uint32_t sgx, uint32_t sgy, bool csh_in_ash) {
  return {128, 128, 32, sgx, sgy, 32, 16, csh_in_ash};
}
const Row kQ4gswCandidates[] = {
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32", tile(4, 2, false),
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g24s32", tile(2, 4, false),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32", tile(4, 2, false),
     kBufTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32c", tile(4, 2, true),
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g24s32f32c", tile(2, 4, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
};

// dq8ca (8da4w) sweep candidates: glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpgtr_sweep.yaml.
const Row kDq8caCandidates[] = {
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_sweep_t128x64k32g42s32du",
     {128, 64, 32, 4, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified, true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_sweep_t128x64k32g42s32dus",
     {128, 64, 32, 4, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified, true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_sweep_t128x64k32g42s32dus1",
     {128, 64, 32, 4, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified, true},
};

std::string& requested_dq8ca_variant() {
  static std::string v = [] {
    const char* e = std::getenv("ET_VK_SARC_DQ8CA_VARIANT");
    return std::string(e != nullptr ? e : "");
  }();
  return v;
}

std::string& requested_variant() {
  static std::string v = [] {
    const char* e = std::getenv("ET_VK_SARC_Q4GSW_VARIANT");
    return std::string(e != nullptr ? e : "");
  }();
  return v;
}

std::optional<Choice> dev_select(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const std::optional<Choice>& table_choice) {
  const bool linear =
      shape.op == Op::kQ4gswLinear || shape.op == Op::kDq8caLinear;
  if (linear && env_true("ET_VK_FORCE_TILED_LINEAR")) {
    return std::nullopt;
  }
  if (!linear && env_true("ET_VK_DISABLE_COOPMAT")) {
    return std::nullopt;
  }
  const std::string& want = shape.op == Op::kDq8caLinear
      ? requested_dq8ca_variant()
      : requested_variant();
  if (want.empty() || !linear) {
    return table_choice;
  }
  // Exact tile token: prefer a candidate, then a release row.
  for (const auto* store : {&candidates(), &rows()}) {
    for (const Row& row : *store) {
      if (row.op == shape.op && ends_with(row.kernel_base, "_" + want) &&
          q4gsw_coopmat_fits(device, shape, row)) {
        return Choice{row.kernel_base, row.dims, row.rowmajor_a};
      }
    }
  }
  return std::nullopt;
}

struct Registrar {
  Registrar() {
    register_candidates(
        kQ4gswCandidates, sizeof(kQ4gswCandidates) / sizeof(kQ4gswCandidates[0]));
    register_candidates(
        kDq8caCandidates, sizeof(kDq8caCandidates) / sizeof(kDq8caCandidates[0]));
    Override o;
    o.allow_unverified = env_true("ET_VK_SARC_UNVERIFIED");
    o.force_path =
        !requested_variant().empty() || !requested_dq8ca_variant().empty();
    o.select = dev_select;
    set_override(o);
    if (o.allow_unverified || o.force_path) {
      std::cerr << "[sarc_dev] overrides active: unverified="
                << o.allow_unverified << " variant=" << requested_variant()
                << " dq8ca_variant=" << requested_dq8ca_variant()
                << std::endl;
    }
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
