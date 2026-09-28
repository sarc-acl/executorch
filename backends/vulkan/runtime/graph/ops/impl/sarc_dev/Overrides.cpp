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
//   ET_VK_SARC_DQ8CA_VARIANT=<tile>   use the dq8ca candidate or release row whose
//                                     kernel ends in this token (e.g. zpgtr_t128x64k32g42s32)
//                                     for 8da4w wherever it fits (prefill only);
//                                     needs a device with active dq8ca rows

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
constexpr TileDims tile_mnk(
    uint32_t m, uint32_t n, uint32_t k, uint32_t sgx, uint32_t sgy, bool csh_in_ash) {
  return {m, n, k, sgx, sgy, 32, 16, csh_in_ash};
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
    // RX 7900 XTX 2026-09-27 (openspec/changes/sarc-1.5-7900xtx-4w).
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t256x128k32g24s32f32c", tile_mnk(256, 128, 32, 2, 4, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t256x128k32g44s32f32c", tile_mnk(256, 128, 32, 4, 4, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x256k32g42s32f32c", tile_mnk(128, 256, 32, 4, 2, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt", tile_mnk(128, 128, 32, 4, 2, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
    // Adreno 840 (S26) 2026-09-28: the release adreno tile with the fp32
    // group total in registers (ACC_GROUP_FP32_REG).
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g21s64m64x32x16gr",
     {64, 64, 32, 2, 1, 64, 64, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
};

// RX 7600 2026-09-28: zpg sweep candidates
// (glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpg_sweep.yaml), ET_VK_SARC_DQ8CA_VARIANT.
constexpr TileDims dq_tile(uint32_t m, uint32_t n, uint32_t sgx, uint32_t sgy) {
  return {m, n, 32, sgx, sgy, 32, 16, false};
}
const Row kDq8caCandidates[] = {
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x128k32g42s32", dq_tile(128, 128, 4, 2),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x128k32g24s32", dq_tile(128, 128, 2, 4),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g24s32", dq_tile(256, 64, 2, 4),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g22s32", dq_tile(128, 64, 2, 2),
     kTex3dTex2d, nullptr, Status::kUnverified},
};

std::string& requested_variant() {
  static std::string v = [] {
    const char* e = std::getenv("ET_VK_SARC_Q4GSW_VARIANT");
    return std::string(e != nullptr ? e : "");
  }();
  return v;
}

const std::string& requested_dq8ca_variant() {
  static const std::string v = [] {
    const char* e = std::getenv("ET_VK_SARC_DQ8CA_VARIANT");
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
  if (shape.op == Op::kDq8caLinear && !requested_dq8ca_variant().empty() &&
      device_has_active_rows(device, shape.op)) {
    for (const auto* store : {&candidates(), &rows()}) {
      for (const Row& row : *store) {
        if (row.op == shape.op &&
            ends_with(row.kernel_base, "_" + requested_dq8ca_variant()) &&
            q4gsw_coopmat_fits(device, shape, row)) {
          return Choice{row.kernel_base, row.dims, row.rowmajor_a};
        }
      }
    }
    return std::nullopt;
  }
  const std::string& want = requested_variant();
  if (want.empty() || shape.op != Op::kQ4gswLinear) {
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
    o.force_path = !requested_variant().empty();
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
