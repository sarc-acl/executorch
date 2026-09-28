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
//   ET_VK_SARC_DQ8CA_VARIANT=<tile>   same for 8da4w prefill: the dq8ca sweep
//                                     candidate or release row whose kernel ends
//                                     in this token (e.g. zpgtr_t128x64k32g42s32);
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
    // Mali-G1-Ultra MC12 (vivo V2502A): subgroup 16, fp16 MMA 16x32x32 (the
    // only large fp16 shape it exposes), 32 KiB LDS, SH_F16V4 (suffix h).
    // A storage bit is set only where Ash + Bsh (+ Csh) < 32 KiB (the fit
    // check tests Csh alone).
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g22s16m16x32x32f32",
     {64, 64, 32, 2, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g22s16m16x32x32f32h",
     {64, 64, 32, 2, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g21s16m16x32x32f32h",
     {64, 64, 32, 2, 1, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g24s16m16x32x32f32h",
     {64, 64, 32, 2, 4, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g12s16m16x32x32f32h",
     {64, 64, 32, 1, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x128k32g41s16m16x32x32f32h",
     {64, 128, 32, 4, 1, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x128k32g42s16m16x32x32f32h",
     {64, 128, 32, 4, 2, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x128k32g44s16m16x32x32f32h",
     {64, 128, 32, 4, 4, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x64k32g22s16m16x32x32f32h",
     {128, 64, 32, 2, 2, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x64k32g24s16m16x32x32f32h",
     {128, 64, 32, 2, 4, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t32x64k32g21s16m16x32x32f32h",
     {32, 64, 32, 2, 1, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t32x32k32g12s16m16x32x32f32h",
     {32, 32, 32, 1, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g22s16m16x32x32h",
     {64, 64, 32, 2, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g24s16m16x32x32h",
     {64, 64, 32, 2, 4, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x128k32g42s16m16x32x32h",
     {64, 128, 32, 4, 2, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g22s16m16x32x32gah",
     {64, 64, 32, 2, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x64k32g24s16m16x32x32gah",
     {64, 64, 32, 2, 4, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x128k32g44s16m16x32x32h",
     {64, 128, 32, 4, 4, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x64k32g28s16m16x32x32h",
     {128, 64, 32, 2, 8, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t32x64k32g22s16m16x32x32h",
     {32, 64, 32, 2, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t32x128k32g42s16m16x32x32h",
     {32, 128, 32, 4, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x32k32g14s16m16x32x32h",
     {64, 32, 32, 1, 4, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x128k32g44s16m16x32x32gah",
     {64, 128, 32, 4, 4, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x64k32g28s16m16x32x32gah",
     {128, 64, 32, 2, 8, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t32x64k32g22s16m16x32x32gah",
     {32, 64, 32, 2, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t32x128k32g42s16m16x32x32gah",
     {32, 128, 32, 4, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x32k32g14s16m16x32x32gah",
     {64, 32, 32, 1, 4, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x64k32g28s16m16x32x32f32h",
     {128, 64, 32, 2, 8, 16, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t32x64k32g22s16m16x32x32f32h",
     {32, 64, 32, 2, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t32x128k32g42s16m16x32x32f32h",
     {32, 128, 32, 4, 2, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x32k32g14s16m16x32x32f32h",
     {64, 32, 32, 1, 4, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    // CSH_BAND (suffix b): texture3d drain one band at a time.
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x128k32g44s16m16x32x32hb",
     {64, 128, 32, 4, 4, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x128k32g44s16m16x32x32gahb",
     {64, 128, 32, 4, 4, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x64k32g28s16m16x32x32hb",
     {128, 64, 32, 2, 8, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x64k32g28s16m16x32x32gahb",
     {128, 64, 32, 2, 8, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
};

// dq8ca (8da4w) sweep candidates, selected with ET_VK_SARC_DQ8CA_VARIANT.
// Keep in sync with glsl/sarc_dev/sarc_linear_dq8ca_coopmat_{zpg,zpgtr}_sweep.yaml.
constexpr TileDims dq_tile(uint32_t m, uint32_t n, uint32_t sgx, uint32_t sgy) {
  return {m, n, 32, sgx, sgy, 32, 16, false};
}
const Row kDq8caCandidates[] = {
    // RX 7600 2026-09-28: RDNA zpg tiles with larger per-subgroup tiles.
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
    // M51 texture3d register-pressure study (zpgtr, row-major A):
    // du = DRAIN_UNROLL, dus / dus1 = + B_SEL_EARLY_N 2 / 1.
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
  // Exact tile token per op: 4w from ET_VK_SARC_Q4GSW_VARIANT, 8da4w from
  // ET_VK_SARC_DQ8CA_VARIANT (the latter only on devices with dq8ca rows).
  const std::string& want = shape.op == Op::kDq8caLinear
      ? requested_dq8ca_variant()
      : requested_variant();
  if (want.empty() || !linear ||
      (shape.op == Op::kDq8caLinear &&
       !device_has_active_rows(device, shape.op))) {
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
    // Only the 4w variable forces the SARC build path on devices without rows;
    // the 8da4w variable needs active dq8ca rows (see dev_select).
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
