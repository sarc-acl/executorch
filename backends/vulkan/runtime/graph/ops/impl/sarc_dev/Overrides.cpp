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
//   ET_VK_SARC_DEV_PROFILE=<name>     a named set of preferred sweep tiles (see
//                                     kProfiles); shapes they do not cover keep
//                                     the table's choice. Takes precedence over
//                                     the *_VARIANT variables; FORCE_TILED_LINEAR
//                                     and DISABLE_COOPMAT still win
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
    // 780M prefill refine 2026-10-03, batch 1 (texture3d screen).
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g22s32f32c", tile_mnk(128, 128, 32, 2, 2, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t256x128k32g42s32f32c", tile_mnk(256, 128, 32, 4, 2, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
    // 780M prefill refine 2026-10-03, batch 2 (wave64, texture3d screen).
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g22s64f32c",
     {128, 128, 32, 2, 2, 64, 16, true}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x256k32g22s64f32c",
     {128, 256, 32, 2, 2, 64, 16, true}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g21s64f32c",
     {128, 128, 32, 2, 1, 64, 16, true}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g12s64f32c",
     {128, 128, 32, 1, 2, 64, 16, true}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t256x128k32g22s64f32c",
     {256, 128, 32, 2, 2, 64, 16, true}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    // 780M prefill refine 2026-10-03, batch 3.
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x256k32g41s32f32c",
     {64, 256, 32, 4, 1, 32, 16, true}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_q4gsw_t128x256k32g42s32f32cp", tile_mnk(128, 256, 32, 4, 2, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
    // 780M prefill refine 2026-10-03, candidate 4: texel-wise weight staging
    // (glsl/sarc_dev/sarc_dev_linear_q4gsw_coopmat_bx.yaml).
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_linear_q4gsw_coopmat_bx_t128x256k32g42s32f32c", tile_mnk(128, 256, 32, 4, 2, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32c", tile_mnk(128, 128, 32, 4, 2, true),
     kTex3dTex2d, nullptr, Status::kUnverified},
    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_q4gsw.yaml):
    // the release 780M tile with shader-clock phase counters written over its output.
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_q4gsw_t128x128k32g42s32f32cp", tile(4, 2, true),
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
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
    // 780M prefill refine 2026-10-03, batch 1 (texture3d screen).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k64g22s32",
     {128, 64, 64, 2, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k64g42s32",
     {128, 64, 64, 4, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g21s32",
     {64, 64, 32, 2, 1, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x128k32g22s32",
     {128, 128, 32, 2, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g22s32",
     {64, 64, 32, 2, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t64x128k32g22s32",
     {64, 128, 32, 2, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g24s32",
     {128, 64, 32, 2, 4, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x32k32g22s32",
     {128, 32, 32, 2, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g42s32",
     {256, 64, 32, 4, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g21s32",
     {128, 64, 32, 2, 1, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g12s32",
     {128, 64, 32, 1, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    // 780M prefill refine 2026-10-03, batch 2 (wave64, texture3d screen).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g22s64",
     {128, 64, 32, 2, 2, 64, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g21s64",
     {128, 64, 32, 2, 1, 64, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g12s64",
     {128, 64, 32, 1, 2, 64, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x128k32g22s64",
     {128, 128, 32, 2, 2, 64, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g22s64",
     {256, 64, 32, 2, 2, 64, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k64g22s64",
     {128, 64, 64, 2, 2, 64, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    // 780M prefill refine 2026-10-03, batch 3.
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t64x128k32g41s32",
     {64, 128, 32, 4, 1, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t256x32k32g14s32",
     {256, 32, 32, 1, 4, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_dq8ca_zpg_t128x64k32g22s32p", dq_tile(128, 64, 2, 2),
     kTex3dTex2d, nullptr, Status::kUnverified},
    // 780M prefill refine 2026-10-03, candidate 2: texel-wise weight staging
    // (glsl/sarc_dev/sarc_dev_linear_dq8ca_coopmat_zpg_bt.yaml).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32",
     {128, 64, 32, 2, 2, 32, 16, false}, kTex3dTex2d | kBufTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x128k32g42s32",
     {128, 128, 32, 4, 2, 32, 16, false}, kTex3dTex2d | kBufTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k64g42s32",
     {128, 64, 64, 4, 2, 32, 16, false}, kTex3dTex2d | kBufTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k64g22s32",
     {128, 64, 64, 2, 2, 32, 16, false}, kTex3dTex2d | kBufTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_dq8ca_coopmat_zpg_bt_t128x64k32g22s32p", dq_tile(128, 64, 2, 2),
     kTex3dTex2d, nullptr, Status::kUnverified},
    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_dq8ca_zpg.yaml).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_dq8ca_zpg_t128x64k32g42s32p", dq_tile(128, 64, 4, 2),
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
    // 780M prefill refine 2026-10-03, candidate 3 (SDPA): glsl/sarc_dev/sarc_sdpa_{qk,av}_coopmat_sweep.yaml.
    // Selected only through ET_VK_SARC_DEV_PROFILE.
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_sweep_t128x64k32g22s64nf",
     {128, 64, 32, 2, 2, 64, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_sweep_t128x64k32g42s32nf",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_sweep_t128x64k32g42s32",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_sweep_t128x64k32g24s32nf",
     {128, 64, 32, 2, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_sweep_t128x64k32g22s32nf",
     {128, 64, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_sweep_t64x64k32g42s32",
     {64, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_sweep_t64x64k32g24s32",
     {64, 64, 32, 2, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    // 780M prefill refine 2026-10-03, candidate 5: QK^T with packed staging
    // (glsl/sarc_dev/sarc_sdpa_qk_coopmat_pk.yaml).
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_pk_t128x64k32g22s64nf",
     {128, 64, 32, 2, 2, 64, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_pk_t128x64k64g22s64nf",
     {128, 64, 64, 2, 2, 64, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_pk_t128x64k32g42s32nf",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_pk_t128x64k64g42s32nf",
     {128, 64, 64, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    // 780M prefill refine 2026-10-03, candidate 6: attn*V with multi-pass staging
    // (glsl/sarc_dev/sarc_sdpa_av_coopmat_ml.yaml).
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_ml_t64x128k32g42s32",
     {64, 128, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_ml_t64x128k32g22s64",
     {64, 128, 32, 2, 2, 64, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_ml_t128x128k32g42s32",
     {128, 128, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_ml_t128x64k32g42s32",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_pk_t128x128k32g42s32nf",
     {128, 128, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_pk_t64x64k32g22s32nf",
     {64, 64, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_pk_t128x64k32g24s32nf",
     {128, 64, 32, 2, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_pk_t256x64k32g42s32nf",
     {256, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
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

// ET_VK_SARC_DEV_PROFILE=<name>: a named set of preferred sweep tiles. A shape
// that a preferred tile covers (the row fits and the entry's predicate holds)
// runs it; every other shape keeps the table's choice. Unlike the *_VARIANT
// variables it never sends a shape to the non-SARC fallback and never builds
// the SARC path on a device without rows.
struct Preference {
  Op op;
  const char* token; // tile token of a candidate or release row
  bool (*shape_ok)(const ShapeInfo&); // or null
};
// The wide 4w tile stages A once per 256 output columns but leaves 2 workgroups
// per WGP (54 KiB LDS); it only pays once the dispatch is at least 4 tiles wide.
bool n_at_least_1024(const ShapeInfo& s) {
  return s.N >= 1024;
}
// 780M, openspec/changes/sarc-1.5-780m-prefill-refine.
const Preference k780mRefine1[] = {
    {Op::kDq8caLinear, "t128x64k32g22s32", nullptr},
    {Op::kQ4gswLinear, "t128x256k32g42s32f32c", n_at_least_1024},
};
const Preference k780mRefine1Dq[] = {
    {Op::kDq8caLinear, "t128x64k32g22s32", nullptr},
};
const Preference k780mRefine1Q4[] = {
    {Op::kQ4gswLinear, "t128x256k32g42s32f32c", n_at_least_1024},
};
const Preference k780mRefine2[] = {
    {Op::kDq8caLinear, "bt_t128x64k32g22s32", nullptr},
    {Op::kQ4gswLinear, "t128x256k32g42s32f32c", n_at_least_1024},
};
// refine2 plus the SDPA prefill kernels: QK^T without the never-read mask
// fill (valid with the truncated SARC softmax only) and the subgroup-32 attn*V tile.
const Preference k780mRefine3[] = {
    {Op::kDq8caLinear, "bt_t128x64k32g22s32", nullptr},
    {Op::kQ4gswLinear, "t128x256k32g42s32f32c", n_at_least_1024},
    {Op::kSdpaQk, "t128x64k32g22s64nf", nullptr},
    {Op::kSdpaAv, "t64x64k32g42s32", nullptr},
};
// refine3 with the 4w texel-wise weight staging on the wide tile.
const Preference k780mRefine4[] = {
    {Op::kDq8caLinear, "bt_t128x64k32g22s32", nullptr},
    {Op::kQ4gswLinear, "bx_t128x256k32g42s32f32c", n_at_least_1024},
    {Op::kSdpaQk, "t128x64k32g22s64nf", nullptr},
    {Op::kSdpaAv, "t64x64k32g42s32", nullptr},
};
// Single-kernel SDPA screening profiles (qk-<tile>, av-<tile>).
const Preference kQk_t128x64k32g22s64nf[] = {{Op::kSdpaQk, "t128x64k32g22s64nf", nullptr}};
const Preference kQk_t128x64k32g42s32nf[] = {{Op::kSdpaQk, "t128x64k32g42s32nf", nullptr}};
const Preference kQk_t128x64k32g42s32[] = {{Op::kSdpaQk, "t128x64k32g42s32", nullptr}};
const Preference kQk_t128x64k32g24s32nf[] = {{Op::kSdpaQk, "t128x64k32g24s32nf", nullptr}};
const Preference kQk_t128x64k32g22s32nf[] = {{Op::kSdpaQk, "t128x64k32g22s32nf", nullptr}};
const Preference kAv_t64x64k32g42s32[] = {{Op::kSdpaAv, "t64x64k32g42s32", nullptr}};
const Preference kAv_t64x64k32g24s32[] = {{Op::kSdpaAv, "t64x64k32g24s32", nullptr}};
const Preference kQkPk_t128x64k32g22s64nf[] = {{Op::kSdpaQk, "pk_t128x64k32g22s64nf", nullptr}};
const Preference kQkPk_t128x64k64g22s64nf[] = {{Op::kSdpaQk, "pk_t128x64k64g22s64nf", nullptr}};
const Preference kQkPk_t128x64k32g42s32nf[] = {{Op::kSdpaQk, "pk_t128x64k32g42s32nf", nullptr}};
const Preference kQkPk_t128x64k64g42s32nf[] = {{Op::kSdpaQk, "pk_t128x64k64g42s32nf", nullptr}};
const Preference kAvMl_t64x128k32g42s32[] = {{Op::kSdpaAv, "ml_t64x128k32g42s32", nullptr}};
const Preference kAvMl_t64x128k32g22s64[] = {{Op::kSdpaAv, "ml_t64x128k32g22s64", nullptr}};
const Preference kAvMl_t128x128k32g42s32[] = {{Op::kSdpaAv, "ml_t128x128k32g42s32", nullptr}};
const Preference kAvMl_t128x64k32g42s32[] = {{Op::kSdpaAv, "ml_t128x64k32g42s32", nullptr}};
// refine3 with the packed-staging QK^T kernel.
const Preference k780mRefine5[] = {
    {Op::kDq8caLinear, "bt_t128x64k32g22s32", nullptr},
    {Op::kQ4gswLinear, "t128x256k32g42s32f32c", n_at_least_1024},
    {Op::kSdpaQk, "pk_t128x64k32g42s32nf", nullptr},
    {Op::kSdpaAv, "t64x64k32g42s32", nullptr},
};
// refine5 with the 64 x 64 packed QK^T tile and the 128-column attn*V tile
// where head_dim allows it (3B, 8B); head_dim 64 falls through to 64 x 64.
const Preference k780mRefine6[] = {
    {Op::kDq8caLinear, "bt_t128x64k32g22s32", nullptr},
    {Op::kQ4gswLinear, "t128x256k32g42s32f32c", n_at_least_1024},
    {Op::kSdpaQk, "pk_t64x64k32g22s32nf", nullptr},
    {Op::kSdpaAv, "ml_t64x128k32g42s32", nullptr},
    {Op::kSdpaAv, "t64x64k32g42s32", nullptr},
};
const Preference kQkPk_t128x128k32g42s32nf[] = {{Op::kSdpaQk, "pk_t128x128k32g42s32nf", nullptr}};
const Preference kQkPk_t64x64k32g22s32nf[] = {{Op::kSdpaQk, "pk_t64x64k32g22s32nf", nullptr}};
const Preference kQkPk_t128x64k32g24s32nf[] = {{Op::kSdpaQk, "pk_t128x64k32g24s32nf", nullptr}};
const Preference kQkPk_t256x64k32g42s32nf[] = {{Op::kSdpaQk, "pk_t256x64k32g42s32nf", nullptr}};
// xe2 begin: Intel Xe2 profiles (openspec/changes/sarc-1.5-xe2-prefill-refine, tools/gen_xe2.py)
// Single-kernel screening profiles and the xe2-refineN candidates. They take effect on a device whose
// SDPA base rows are active (impl/sarc_dev/Xe2Sdpa.cpp, ET_VK_SARC_UNVERIFIED=1).
// attn*V: ShapeInfo::N is head_dim.
bool xe2_head_dim_128(const ShapeInfo& s) {
  return s.N >= 128;
}
// 8da4w linear: an output at least four times as wide as the input (1B w1 / w3).
bool xe2_wide_output(const ShapeInfo& s) {
  return s.N >= 4 * s.K;
}
const Preference kXe2_qk_t128x64k32g44s16m8nf[] = {{Op::kSdpaQk, "sweep_t128x64k32g44s16m8nf", nullptr}};
const Preference kXe2_qk_t128x64k32g44s16m8[] = {{Op::kSdpaQk, "sweep_t128x64k32g44s16m8", nullptr}};
const Preference kXe2_qk_t128x64k32g48s16m8nf[] = {{Op::kSdpaQk, "sweep_t128x64k32g48s16m8nf", nullptr}};
const Preference kXe2_qk_t128x64k32g24s16m8nf[] = {{Op::kSdpaQk, "sweep_t128x64k32g24s16m8nf", nullptr}};
const Preference kXe2_qk_t64x64k32g44s16m8nf[] = {{Op::kSdpaQk, "sweep_t64x64k32g44s16m8nf", nullptr}};
const Preference kXe2_qk_t64x64k32g24s16m8nf[] = {{Op::kSdpaQk, "sweep_t64x64k32g24s16m8nf", nullptr}};
const Preference kXe2_qk_t64x128k32g44s16m8nf[] = {{Op::kSdpaQk, "sweep_t64x128k32g44s16m8nf", nullptr}};
const Preference kXe2_qk_t64x128k32g84s16m8nf[] = {{Op::kSdpaQk, "sweep_t64x128k32g84s16m8nf", nullptr}};
const Preference kXe2_qkpk_t128x64k32g44s16m8nf[] = {{Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr}};
const Preference kXe2_qkpk_t128x64k64g44s16m8nf[] = {{Op::kSdpaQk, "pk_t128x64k64g44s16m8nf", nullptr}};
const Preference kXe2_qkpk_t128x64k32g48s16m8nf[] = {{Op::kSdpaQk, "pk_t128x64k32g48s16m8nf", nullptr}};
const Preference kXe2_qkpk_t128x64k32g24s16m8nf[] = {{Op::kSdpaQk, "pk_t128x64k32g24s16m8nf", nullptr}};
const Preference kXe2_qkpk_t64x64k32g44s16m8nf[] = {{Op::kSdpaQk, "pk_t64x64k32g44s16m8nf", nullptr}};
const Preference kXe2_qkpk_t64x64k64g44s16m8nf[] = {{Op::kSdpaQk, "pk_t64x64k64g44s16m8nf", nullptr}};
const Preference kXe2_qkpk_t64x128k32g44s16m8nf[] = {{Op::kSdpaQk, "pk_t64x128k32g44s16m8nf", nullptr}};
const Preference kXe2_qkpk_t64x128k32g84s16m8nf[] = {{Op::kSdpaQk, "pk_t64x128k32g84s16m8nf", nullptr}};
const Preference kXe2_av_t64x64k32g44s16m8[] = {{Op::kSdpaAv, "sweep_t64x64k32g44s16m8", nullptr}};
const Preference kXe2_avml_t64x64k32g44s16m8[] = {{Op::kSdpaAv, "ml_t64x64k32g44s16m8", nullptr}};
const Preference kXe2_avml_t64x64k32g24s16m8[] = {{Op::kSdpaAv, "ml_t64x64k32g24s16m8", nullptr}};
const Preference kXe2_avml_t64x64k32g48s16m8[] = {{Op::kSdpaAv, "ml_t64x64k32g48s16m8", nullptr}};
const Preference kXe2_avml_t128x64k32g44s16m8[] = {{Op::kSdpaAv, "ml_t128x64k32g44s16m8", nullptr}};
const Preference kXe2_avml_t128x64k32g48s16m8[] = {{Op::kSdpaAv, "ml_t128x64k32g48s16m8", nullptr}};
const Preference kXe2_avml_t64x128k32g44s16m8[] = {{Op::kSdpaAv, "ml_t64x128k32g44s16m8", nullptr}};
const Preference kXe2_avml_t64x128k32g84s16m8[] = {{Op::kSdpaAv, "ml_t64x128k32g84s16m8", nullptr}};
const Preference kXe2_avml_t128x128k32g48s16m8[] = {{Op::kSdpaAv, "ml_t128x128k32g48s16m8", nullptr}};
const Preference kXe2_avml_t128x128k32g88s16m8[] = {{Op::kSdpaAv, "ml_t128x128k32g88s16m8", nullptr}};
const Preference kXe2_avml_t32x64k32g42s16m8[] = {{Op::kSdpaAv, "ml_t32x64k32g42s16m8", nullptr}};
const Preference kXe2_qkfr_t128x64k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t128x64k32g44s16m8nf", nullptr}};
const Preference kXe2_qkfr_t128x64k32g48s16m8nf[] = {{Op::kSdpaQk, "xe2_t128x64k32g48s16m8nf", nullptr}};
const Preference kXe2_qkfr_t128x64k32g24s16m8nf[] = {{Op::kSdpaQk, "xe2_t128x64k32g24s16m8nf", nullptr}};
const Preference kXe2_qkfr_t128x64k64g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t128x64k64g44s16m8nf", nullptr}};
const Preference kXe2_qkfr_t64x64k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x64k32g44s16m8nf", nullptr}};
const Preference kXe2_qkfr_t64x64k32g24s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x64k32g24s16m8nf", nullptr}};
const Preference kXe2_qkfr_t64x64k64g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x64k64g44s16m8nf", nullptr}};
const Preference kXe2_qkfr_t64x128k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x128k32g44s16m8nf", nullptr}};
const Preference kXe2_qkfr_t64x128k32g84s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x128k32g84s16m8nf", nullptr}};
const Preference kXe2_avfr_t64x64k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t64x64k32g44s16m8", nullptr}};
const Preference kXe2_avfr_t64x64k32g24s16m8[] = {{Op::kSdpaAv, "xe2_t64x64k32g24s16m8", nullptr}};
const Preference kXe2_avfr_t64x64k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t64x64k32g48s16m8", nullptr}};
const Preference kXe2_avfr_t128x64k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t128x64k32g44s16m8", nullptr}};
const Preference kXe2_avfr_t128x64k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t128x64k32g48s16m8", nullptr}};
const Preference kXe2_avfr_t64x128k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t64x128k32g44s16m8", nullptr}};
const Preference kXe2_avfr_t64x128k32g84s16m8[] = {{Op::kSdpaAv, "xe2_t64x128k32g84s16m8", nullptr}};
const Preference kXe2_avfr_t128x128k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t128x128k32g48s16m8", nullptr}};
const Preference kXe2_avfr_t128x128k32g88s16m8[] = {{Op::kSdpaAv, "xe2_t128x128k32g88s16m8", nullptr}};
const Preference kXe2_avfr_t32x64k32g42s16m8[] = {{Op::kSdpaAv, "xe2_t32x64k32g42s16m8", nullptr}};
const Preference kXe2_avfr_t128x64k32g24s16m8[] = {{Op::kSdpaAv, "xe2_t128x64k32g24s16m8", nullptr}};
const Preference kXe2_avfr_t256x64k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t256x64k32g48s16m8", nullptr}};
const Preference kXe2_avfr_t256x64k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t256x64k32g44s16m8", nullptr}};
const Preference kXe2_avfr_t128x128k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t128x128k32g44s16m8", nullptr}};
const Preference kXe2_avfr_t256x128k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t256x128k32g48s16m8", nullptr}};
const Preference kXe2_qkc_t128x64k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", nullptr}};
const Preference kXe2_qkc_t128x64k32g24s16m8nf[] = {{Op::kSdpaQk, "xe2c_t128x64k32g24s16m8nf", nullptr}};
const Preference kXe2_qkc_t64x128k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2c_t64x128k32g44s16m8nf", nullptr}};
const Preference kXe2_qkc_t64x64k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2c_t64x64k32g44s16m8nf", nullptr}};
const Preference kXe2_qkc_t128x64k64g44s16m8nf[] = {{Op::kSdpaQk, "xe2c_t128x64k64g44s16m8nf", nullptr}};
const Preference kXe2_qkc_t64x128k32g42s16m8nf[] = {{Op::kSdpaQk, "xe2c_t64x128k32g42s16m8nf", nullptr}};
const Preference kXe2_refine1[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
};
const Preference kXe2_refine2[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
};
const Preference kXe2_refine3[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
};
const Preference kXe2_refine4[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "bt_t128x64k64g44s16m8", xe2_wide_output},
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
};
const Preference kXe2_dq_k64[] = {
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
};
// xe2 end
// b580 begin: Arc B580 profiles (openspec/changes/sarc-1.5-b580-prefill-refine, tools/gen_b580.py)
// Single-kernel screening profiles and the b580-refineN candidates. They take effect on the Arc B580 only
// (SDPA base rows of impl/sarc_dev/B580Sdpa.cpp, ET_VK_SARC_UNVERIFIED=1).
// attn*V: ShapeInfo::N is head_dim.
bool b580_head_dim_128(const ShapeInfo& s) {
  return s.N >= 128;
}
const Preference kB580_qk_sweep_t128x64k32g44s16m8nf[] = {{Op::kSdpaQk, "sweep_t128x64k32g44s16m8nf", nullptr}};
const Preference kB580_av_sweep_t64x64k32g44s16m8[] = {{Op::kSdpaAv, "sweep_t64x64k32g44s16m8", nullptr}};
const Preference kB580_qk_sweep_t128x64k32g44s16m8[] = {{Op::kSdpaQk, "sweep_t128x64k32g44s16m8", nullptr}};
const Preference kB580_qk_sweep_t128x64k32g48s16m8nf[] = {{Op::kSdpaQk, "sweep_t128x64k32g48s16m8nf", nullptr}};
const Preference kB580_qk_sweep_t128x64k32g24s16m8nf[] = {{Op::kSdpaQk, "sweep_t128x64k32g24s16m8nf", nullptr}};
const Preference kB580_qk_sweep_t64x64k32g44s16m8nf[] = {{Op::kSdpaQk, "sweep_t64x64k32g44s16m8nf", nullptr}};
const Preference kB580_qk_sweep_t64x64k32g24s16m8nf[] = {{Op::kSdpaQk, "sweep_t64x64k32g24s16m8nf", nullptr}};
const Preference kB580_qk_sweep_t64x128k32g44s16m8nf[] = {{Op::kSdpaQk, "sweep_t64x128k32g44s16m8nf", nullptr}};
const Preference kB580_qk_sweep_t64x128k32g84s16m8nf[] = {{Op::kSdpaQk, "sweep_t64x128k32g84s16m8nf", nullptr}};
const Preference kB580_qk_pk_t128x64k32g44s16m8nf[] = {{Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr}};
const Preference kB580_qk_pk_t128x64k64g44s16m8nf[] = {{Op::kSdpaQk, "pk_t128x64k64g44s16m8nf", nullptr}};
const Preference kB580_qk_pk_t128x64k32g48s16m8nf[] = {{Op::kSdpaQk, "pk_t128x64k32g48s16m8nf", nullptr}};
const Preference kB580_qk_pk_t128x64k32g24s16m8nf[] = {{Op::kSdpaQk, "pk_t128x64k32g24s16m8nf", nullptr}};
const Preference kB580_qk_pk_t64x64k32g44s16m8nf[] = {{Op::kSdpaQk, "pk_t64x64k32g44s16m8nf", nullptr}};
const Preference kB580_qk_pk_t64x64k64g44s16m8nf[] = {{Op::kSdpaQk, "pk_t64x64k64g44s16m8nf", nullptr}};
const Preference kB580_qk_pk_t64x128k32g44s16m8nf[] = {{Op::kSdpaQk, "pk_t64x128k32g44s16m8nf", nullptr}};
const Preference kB580_qk_pk_t64x128k32g84s16m8nf[] = {{Op::kSdpaQk, "pk_t64x128k32g84s16m8nf", nullptr}};
const Preference kB580_av_ml_t64x64k32g44s16m8[] = {{Op::kSdpaAv, "ml_t64x64k32g44s16m8", nullptr}};
const Preference kB580_av_ml_t64x64k32g24s16m8[] = {{Op::kSdpaAv, "ml_t64x64k32g24s16m8", nullptr}};
const Preference kB580_av_ml_t64x64k32g48s16m8[] = {{Op::kSdpaAv, "ml_t64x64k32g48s16m8", nullptr}};
const Preference kB580_av_ml_t128x64k32g44s16m8[] = {{Op::kSdpaAv, "ml_t128x64k32g44s16m8", nullptr}};
const Preference kB580_av_ml_t128x64k32g48s16m8[] = {{Op::kSdpaAv, "ml_t128x64k32g48s16m8", nullptr}};
const Preference kB580_av_ml_t64x128k32g44s16m8[] = {{Op::kSdpaAv, "ml_t64x128k32g44s16m8", nullptr}};
const Preference kB580_av_ml_t64x128k32g84s16m8[] = {{Op::kSdpaAv, "ml_t64x128k32g84s16m8", nullptr}};
const Preference kB580_av_ml_t128x128k32g48s16m8[] = {{Op::kSdpaAv, "ml_t128x128k32g48s16m8", nullptr}};
const Preference kB580_av_ml_t128x128k32g88s16m8[] = {{Op::kSdpaAv, "ml_t128x128k32g88s16m8", nullptr}};
const Preference kB580_av_ml_t32x64k32g42s16m8[] = {{Op::kSdpaAv, "ml_t32x64k32g42s16m8", nullptr}};
const Preference kB580_qk_xe2_t128x64k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t128x64k32g44s16m8nf", nullptr}};
const Preference kB580_qk_xe2_t128x64k32g48s16m8nf[] = {{Op::kSdpaQk, "xe2_t128x64k32g48s16m8nf", nullptr}};
const Preference kB580_qk_xe2_t128x64k32g24s16m8nf[] = {{Op::kSdpaQk, "xe2_t128x64k32g24s16m8nf", nullptr}};
const Preference kB580_qk_xe2_t128x64k64g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t128x64k64g44s16m8nf", nullptr}};
const Preference kB580_qk_xe2_t64x64k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x64k32g44s16m8nf", nullptr}};
const Preference kB580_qk_xe2_t64x64k32g24s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x64k32g24s16m8nf", nullptr}};
const Preference kB580_qk_xe2_t64x64k64g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x64k64g44s16m8nf", nullptr}};
const Preference kB580_qk_xe2_t64x128k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x128k32g44s16m8nf", nullptr}};
const Preference kB580_qk_xe2_t64x128k32g84s16m8nf[] = {{Op::kSdpaQk, "xe2_t64x128k32g84s16m8nf", nullptr}};
const Preference kB580_av_xe2_t64x64k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t64x64k32g44s16m8", nullptr}};
const Preference kB580_av_xe2_t64x64k32g24s16m8[] = {{Op::kSdpaAv, "xe2_t64x64k32g24s16m8", nullptr}};
const Preference kB580_av_xe2_t64x64k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t64x64k32g48s16m8", nullptr}};
const Preference kB580_av_xe2_t128x64k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t128x64k32g44s16m8", nullptr}};
const Preference kB580_av_xe2_t128x64k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t128x64k32g48s16m8", nullptr}};
const Preference kB580_av_xe2_t64x128k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t64x128k32g44s16m8", nullptr}};
const Preference kB580_av_xe2_t64x128k32g84s16m8[] = {{Op::kSdpaAv, "xe2_t64x128k32g84s16m8", nullptr}};
const Preference kB580_av_xe2_t128x128k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t128x128k32g48s16m8", nullptr}};
const Preference kB580_av_xe2_t128x128k32g88s16m8[] = {{Op::kSdpaAv, "xe2_t128x128k32g88s16m8", nullptr}};
const Preference kB580_av_xe2_t32x64k32g42s16m8[] = {{Op::kSdpaAv, "xe2_t32x64k32g42s16m8", nullptr}};
const Preference kB580_av_xe2_t128x64k32g24s16m8[] = {{Op::kSdpaAv, "xe2_t128x64k32g24s16m8", nullptr}};
const Preference kB580_av_xe2_t256x64k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t256x64k32g48s16m8", nullptr}};
const Preference kB580_av_xe2_t256x64k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t256x64k32g44s16m8", nullptr}};
const Preference kB580_av_xe2_t128x128k32g44s16m8[] = {{Op::kSdpaAv, "xe2_t128x128k32g44s16m8", nullptr}};
const Preference kB580_av_xe2_t256x128k32g48s16m8[] = {{Op::kSdpaAv, "xe2_t256x128k32g48s16m8", nullptr}};
const Preference kB580_qk_xe2c_t128x64k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", nullptr}};
const Preference kB580_qk_xe2c_t128x64k32g24s16m8nf[] = {{Op::kSdpaQk, "xe2c_t128x64k32g24s16m8nf", nullptr}};
const Preference kB580_qk_xe2c_t64x128k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2c_t64x128k32g44s16m8nf", nullptr}};
const Preference kB580_qk_xe2c_t64x64k32g44s16m8nf[] = {{Op::kSdpaQk, "xe2c_t64x64k32g44s16m8nf", nullptr}};
const Preference kB580_qk_xe2c_t128x64k64g44s16m8nf[] = {{Op::kSdpaQk, "xe2c_t128x64k64g44s16m8nf", nullptr}};
const Preference kB580_qk_xe2c_t64x128k32g42s16m8nf[] = {{Op::kSdpaQk, "xe2c_t64x128k32g42s16m8nf", nullptr}};
// candidate 0 = the B70's accepted xe2-refine1 kernels, selected for this card
const Preference kB580_refine0[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
};
// candidate 1 = refine0 + the balanced K = 64 8da4w tile with texel-wise weight staging (B580 screen 2: 1.31x at kernel level)
const Preference kB580_refine1[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
};
// the 8da4w tile of refine1 alone, on the release SDPA path
const Preference kB580_dq_k64[] = {
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
};
// b580 end
struct Profile {
  const char* name;
  const Preference* prefs;
  size_t count;
};
const Profile kProfiles[] = {
    {"780m-refine1", k780mRefine1, sizeof(k780mRefine1) / sizeof(Preference)},
    {"780m-refine2", k780mRefine2, sizeof(k780mRefine2) / sizeof(Preference)},
    {"780m-refine3", k780mRefine3, sizeof(k780mRefine3) / sizeof(Preference)},
    {"780m-refine4", k780mRefine4, sizeof(k780mRefine4) / sizeof(Preference)},
    {"qk-t128x64k32g22s64nf", kQk_t128x64k32g22s64nf, 1},
    {"qk-t128x64k32g42s32nf", kQk_t128x64k32g42s32nf, 1},
    {"qk-t128x64k32g42s32", kQk_t128x64k32g42s32, 1},
    {"qk-t128x64k32g24s32nf", kQk_t128x64k32g24s32nf, 1},
    {"qk-t128x64k32g22s32nf", kQk_t128x64k32g22s32nf, 1},
    {"av-t64x64k32g42s32", kAv_t64x64k32g42s32, 1},
    {"av-t64x64k32g24s32", kAv_t64x64k32g24s32, 1},
    {"qkpk-t128x64k32g22s64nf", kQkPk_t128x64k32g22s64nf, 1},
    {"qkpk-t128x64k64g22s64nf", kQkPk_t128x64k64g22s64nf, 1},
    {"qkpk-t128x64k32g42s32nf", kQkPk_t128x64k32g42s32nf, 1},
    {"qkpk-t128x64k64g42s32nf", kQkPk_t128x64k64g42s32nf, 1},
    {"avml-t64x128k32g42s32", kAvMl_t64x128k32g42s32, 1},
    {"avml-t64x128k32g22s64", kAvMl_t64x128k32g22s64, 1},
    {"avml-t128x128k32g42s32", kAvMl_t128x128k32g42s32, 1},
    {"avml-t128x64k32g42s32", kAvMl_t128x64k32g42s32, 1},
    {"780m-refine5", k780mRefine5, sizeof(k780mRefine5) / sizeof(Preference)},
    {"780m-refine6", k780mRefine6, sizeof(k780mRefine6) / sizeof(Preference)},
    {"qkpk-t128x128k32g42s32nf", kQkPk_t128x128k32g42s32nf, 1},
    {"qkpk-t64x64k32g22s32nf", kQkPk_t64x64k32g22s32nf, 1},
    {"qkpk-t128x64k32g24s32nf", kQkPk_t128x64k32g24s32nf, 1},
    {"qkpk-t256x64k32g42s32nf", kQkPk_t256x64k32g42s32nf, 1},
    {"780m-refine1-dq", k780mRefine1Dq, sizeof(k780mRefine1Dq) / sizeof(Preference)},
    {"780m-refine1-q4", k780mRefine1Q4, sizeof(k780mRefine1Q4) / sizeof(Preference)},
    // xe2 begin: Intel Xe2 profiles (tools/gen_xe2.py)
    {"xe2-qk-t128x64k32g44s16m8nf", kXe2_qk_t128x64k32g44s16m8nf, 1},
    {"xe2-qk-t128x64k32g44s16m8", kXe2_qk_t128x64k32g44s16m8, 1},
    {"xe2-qk-t128x64k32g48s16m8nf", kXe2_qk_t128x64k32g48s16m8nf, 1},
    {"xe2-qk-t128x64k32g24s16m8nf", kXe2_qk_t128x64k32g24s16m8nf, 1},
    {"xe2-qk-t64x64k32g44s16m8nf", kXe2_qk_t64x64k32g44s16m8nf, 1},
    {"xe2-qk-t64x64k32g24s16m8nf", kXe2_qk_t64x64k32g24s16m8nf, 1},
    {"xe2-qk-t64x128k32g44s16m8nf", kXe2_qk_t64x128k32g44s16m8nf, 1},
    {"xe2-qk-t64x128k32g84s16m8nf", kXe2_qk_t64x128k32g84s16m8nf, 1},
    {"xe2-qkpk-t128x64k32g44s16m8nf", kXe2_qkpk_t128x64k32g44s16m8nf, 1},
    {"xe2-qkpk-t128x64k64g44s16m8nf", kXe2_qkpk_t128x64k64g44s16m8nf, 1},
    {"xe2-qkpk-t128x64k32g48s16m8nf", kXe2_qkpk_t128x64k32g48s16m8nf, 1},
    {"xe2-qkpk-t128x64k32g24s16m8nf", kXe2_qkpk_t128x64k32g24s16m8nf, 1},
    {"xe2-qkpk-t64x64k32g44s16m8nf", kXe2_qkpk_t64x64k32g44s16m8nf, 1},
    {"xe2-qkpk-t64x64k64g44s16m8nf", kXe2_qkpk_t64x64k64g44s16m8nf, 1},
    {"xe2-qkpk-t64x128k32g44s16m8nf", kXe2_qkpk_t64x128k32g44s16m8nf, 1},
    {"xe2-qkpk-t64x128k32g84s16m8nf", kXe2_qkpk_t64x128k32g84s16m8nf, 1},
    {"xe2-av-t64x64k32g44s16m8", kXe2_av_t64x64k32g44s16m8, 1},
    {"xe2-avml-t64x64k32g44s16m8", kXe2_avml_t64x64k32g44s16m8, 1},
    {"xe2-avml-t64x64k32g24s16m8", kXe2_avml_t64x64k32g24s16m8, 1},
    {"xe2-avml-t64x64k32g48s16m8", kXe2_avml_t64x64k32g48s16m8, 1},
    {"xe2-avml-t128x64k32g44s16m8", kXe2_avml_t128x64k32g44s16m8, 1},
    {"xe2-avml-t128x64k32g48s16m8", kXe2_avml_t128x64k32g48s16m8, 1},
    {"xe2-avml-t64x128k32g44s16m8", kXe2_avml_t64x128k32g44s16m8, 1},
    {"xe2-avml-t64x128k32g84s16m8", kXe2_avml_t64x128k32g84s16m8, 1},
    {"xe2-avml-t128x128k32g48s16m8", kXe2_avml_t128x128k32g48s16m8, 1},
    {"xe2-avml-t128x128k32g88s16m8", kXe2_avml_t128x128k32g88s16m8, 1},
    {"xe2-avml-t32x64k32g42s16m8", kXe2_avml_t32x64k32g42s16m8, 1},
    {"xe2-qkfr-t128x64k32g44s16m8nf", kXe2_qkfr_t128x64k32g44s16m8nf, 1},
    {"xe2-qkfr-t128x64k32g48s16m8nf", kXe2_qkfr_t128x64k32g48s16m8nf, 1},
    {"xe2-qkfr-t128x64k32g24s16m8nf", kXe2_qkfr_t128x64k32g24s16m8nf, 1},
    {"xe2-qkfr-t128x64k64g44s16m8nf", kXe2_qkfr_t128x64k64g44s16m8nf, 1},
    {"xe2-qkfr-t64x64k32g44s16m8nf", kXe2_qkfr_t64x64k32g44s16m8nf, 1},
    {"xe2-qkfr-t64x64k32g24s16m8nf", kXe2_qkfr_t64x64k32g24s16m8nf, 1},
    {"xe2-qkfr-t64x64k64g44s16m8nf", kXe2_qkfr_t64x64k64g44s16m8nf, 1},
    {"xe2-qkfr-t64x128k32g44s16m8nf", kXe2_qkfr_t64x128k32g44s16m8nf, 1},
    {"xe2-qkfr-t64x128k32g84s16m8nf", kXe2_qkfr_t64x128k32g84s16m8nf, 1},
    {"xe2-avfr-t64x64k32g44s16m8", kXe2_avfr_t64x64k32g44s16m8, 1},
    {"xe2-avfr-t64x64k32g24s16m8", kXe2_avfr_t64x64k32g24s16m8, 1},
    {"xe2-avfr-t64x64k32g48s16m8", kXe2_avfr_t64x64k32g48s16m8, 1},
    {"xe2-avfr-t128x64k32g44s16m8", kXe2_avfr_t128x64k32g44s16m8, 1},
    {"xe2-avfr-t128x64k32g48s16m8", kXe2_avfr_t128x64k32g48s16m8, 1},
    {"xe2-avfr-t64x128k32g44s16m8", kXe2_avfr_t64x128k32g44s16m8, 1},
    {"xe2-avfr-t64x128k32g84s16m8", kXe2_avfr_t64x128k32g84s16m8, 1},
    {"xe2-avfr-t128x128k32g48s16m8", kXe2_avfr_t128x128k32g48s16m8, 1},
    {"xe2-avfr-t128x128k32g88s16m8", kXe2_avfr_t128x128k32g88s16m8, 1},
    {"xe2-avfr-t32x64k32g42s16m8", kXe2_avfr_t32x64k32g42s16m8, 1},
    {"xe2-avfr-t128x64k32g24s16m8", kXe2_avfr_t128x64k32g24s16m8, 1},
    {"xe2-avfr-t256x64k32g48s16m8", kXe2_avfr_t256x64k32g48s16m8, 1},
    {"xe2-avfr-t256x64k32g44s16m8", kXe2_avfr_t256x64k32g44s16m8, 1},
    {"xe2-avfr-t128x128k32g44s16m8", kXe2_avfr_t128x128k32g44s16m8, 1},
    {"xe2-avfr-t256x128k32g48s16m8", kXe2_avfr_t256x128k32g48s16m8, 1},
    {"xe2-qkc-t128x64k32g44s16m8nf", kXe2_qkc_t128x64k32g44s16m8nf, 1},
    {"xe2-qkc-t128x64k32g24s16m8nf", kXe2_qkc_t128x64k32g24s16m8nf, 1},
    {"xe2-qkc-t64x128k32g44s16m8nf", kXe2_qkc_t64x128k32g44s16m8nf, 1},
    {"xe2-qkc-t64x64k32g44s16m8nf", kXe2_qkc_t64x64k32g44s16m8nf, 1},
    {"xe2-qkc-t128x64k64g44s16m8nf", kXe2_qkc_t128x64k64g44s16m8nf, 1},
    {"xe2-qkc-t64x128k32g42s16m8nf", kXe2_qkc_t64x128k32g42s16m8nf, 1},
    {"xe2-sdpa0", nullptr, 0},
    {"xe2-refine1", kXe2_refine1, sizeof(kXe2_refine1) / sizeof(Preference)},
    {"xe2-refine2", kXe2_refine2, sizeof(kXe2_refine2) / sizeof(Preference)},
    {"xe2-refine3", kXe2_refine3, sizeof(kXe2_refine3) / sizeof(Preference)},
    {"xe2-refine4", kXe2_refine4, sizeof(kXe2_refine4) / sizeof(Preference)},
    {"xe2-dq-k64", kXe2_dq_k64, sizeof(kXe2_dq_k64) / sizeof(Preference)},
    // xe2 end
    // b580 begin: Arc B580 profiles (tools/gen_b580.py)
    {"b580-qk-sweep-t128x64k32g44s16m8nf", kB580_qk_sweep_t128x64k32g44s16m8nf, 1},
    {"b580-av-sweep-t64x64k32g44s16m8", kB580_av_sweep_t64x64k32g44s16m8, 1},
    {"b580-qk-sweep-t128x64k32g44s16m8", kB580_qk_sweep_t128x64k32g44s16m8, 1},
    {"b580-qk-sweep-t128x64k32g48s16m8nf", kB580_qk_sweep_t128x64k32g48s16m8nf, 1},
    {"b580-qk-sweep-t128x64k32g24s16m8nf", kB580_qk_sweep_t128x64k32g24s16m8nf, 1},
    {"b580-qk-sweep-t64x64k32g44s16m8nf", kB580_qk_sweep_t64x64k32g44s16m8nf, 1},
    {"b580-qk-sweep-t64x64k32g24s16m8nf", kB580_qk_sweep_t64x64k32g24s16m8nf, 1},
    {"b580-qk-sweep-t64x128k32g44s16m8nf", kB580_qk_sweep_t64x128k32g44s16m8nf, 1},
    {"b580-qk-sweep-t64x128k32g84s16m8nf", kB580_qk_sweep_t64x128k32g84s16m8nf, 1},
    {"b580-qk-pk-t128x64k32g44s16m8nf", kB580_qk_pk_t128x64k32g44s16m8nf, 1},
    {"b580-qk-pk-t128x64k64g44s16m8nf", kB580_qk_pk_t128x64k64g44s16m8nf, 1},
    {"b580-qk-pk-t128x64k32g48s16m8nf", kB580_qk_pk_t128x64k32g48s16m8nf, 1},
    {"b580-qk-pk-t128x64k32g24s16m8nf", kB580_qk_pk_t128x64k32g24s16m8nf, 1},
    {"b580-qk-pk-t64x64k32g44s16m8nf", kB580_qk_pk_t64x64k32g44s16m8nf, 1},
    {"b580-qk-pk-t64x64k64g44s16m8nf", kB580_qk_pk_t64x64k64g44s16m8nf, 1},
    {"b580-qk-pk-t64x128k32g44s16m8nf", kB580_qk_pk_t64x128k32g44s16m8nf, 1},
    {"b580-qk-pk-t64x128k32g84s16m8nf", kB580_qk_pk_t64x128k32g84s16m8nf, 1},
    {"b580-av-ml-t64x64k32g44s16m8", kB580_av_ml_t64x64k32g44s16m8, 1},
    {"b580-av-ml-t64x64k32g24s16m8", kB580_av_ml_t64x64k32g24s16m8, 1},
    {"b580-av-ml-t64x64k32g48s16m8", kB580_av_ml_t64x64k32g48s16m8, 1},
    {"b580-av-ml-t128x64k32g44s16m8", kB580_av_ml_t128x64k32g44s16m8, 1},
    {"b580-av-ml-t128x64k32g48s16m8", kB580_av_ml_t128x64k32g48s16m8, 1},
    {"b580-av-ml-t64x128k32g44s16m8", kB580_av_ml_t64x128k32g44s16m8, 1},
    {"b580-av-ml-t64x128k32g84s16m8", kB580_av_ml_t64x128k32g84s16m8, 1},
    {"b580-av-ml-t128x128k32g48s16m8", kB580_av_ml_t128x128k32g48s16m8, 1},
    {"b580-av-ml-t128x128k32g88s16m8", kB580_av_ml_t128x128k32g88s16m8, 1},
    {"b580-av-ml-t32x64k32g42s16m8", kB580_av_ml_t32x64k32g42s16m8, 1},
    {"b580-qk-xe2-t128x64k32g44s16m8nf", kB580_qk_xe2_t128x64k32g44s16m8nf, 1},
    {"b580-qk-xe2-t128x64k32g48s16m8nf", kB580_qk_xe2_t128x64k32g48s16m8nf, 1},
    {"b580-qk-xe2-t128x64k32g24s16m8nf", kB580_qk_xe2_t128x64k32g24s16m8nf, 1},
    {"b580-qk-xe2-t128x64k64g44s16m8nf", kB580_qk_xe2_t128x64k64g44s16m8nf, 1},
    {"b580-qk-xe2-t64x64k32g44s16m8nf", kB580_qk_xe2_t64x64k32g44s16m8nf, 1},
    {"b580-qk-xe2-t64x64k32g24s16m8nf", kB580_qk_xe2_t64x64k32g24s16m8nf, 1},
    {"b580-qk-xe2-t64x64k64g44s16m8nf", kB580_qk_xe2_t64x64k64g44s16m8nf, 1},
    {"b580-qk-xe2-t64x128k32g44s16m8nf", kB580_qk_xe2_t64x128k32g44s16m8nf, 1},
    {"b580-qk-xe2-t64x128k32g84s16m8nf", kB580_qk_xe2_t64x128k32g84s16m8nf, 1},
    {"b580-av-xe2-t64x64k32g44s16m8", kB580_av_xe2_t64x64k32g44s16m8, 1},
    {"b580-av-xe2-t64x64k32g24s16m8", kB580_av_xe2_t64x64k32g24s16m8, 1},
    {"b580-av-xe2-t64x64k32g48s16m8", kB580_av_xe2_t64x64k32g48s16m8, 1},
    {"b580-av-xe2-t128x64k32g44s16m8", kB580_av_xe2_t128x64k32g44s16m8, 1},
    {"b580-av-xe2-t128x64k32g48s16m8", kB580_av_xe2_t128x64k32g48s16m8, 1},
    {"b580-av-xe2-t64x128k32g44s16m8", kB580_av_xe2_t64x128k32g44s16m8, 1},
    {"b580-av-xe2-t64x128k32g84s16m8", kB580_av_xe2_t64x128k32g84s16m8, 1},
    {"b580-av-xe2-t128x128k32g48s16m8", kB580_av_xe2_t128x128k32g48s16m8, 1},
    {"b580-av-xe2-t128x128k32g88s16m8", kB580_av_xe2_t128x128k32g88s16m8, 1},
    {"b580-av-xe2-t32x64k32g42s16m8", kB580_av_xe2_t32x64k32g42s16m8, 1},
    {"b580-av-xe2-t128x64k32g24s16m8", kB580_av_xe2_t128x64k32g24s16m8, 1},
    {"b580-av-xe2-t256x64k32g48s16m8", kB580_av_xe2_t256x64k32g48s16m8, 1},
    {"b580-av-xe2-t256x64k32g44s16m8", kB580_av_xe2_t256x64k32g44s16m8, 1},
    {"b580-av-xe2-t128x128k32g44s16m8", kB580_av_xe2_t128x128k32g44s16m8, 1},
    {"b580-av-xe2-t256x128k32g48s16m8", kB580_av_xe2_t256x128k32g48s16m8, 1},
    {"b580-qk-xe2c-t128x64k32g44s16m8nf", kB580_qk_xe2c_t128x64k32g44s16m8nf, 1},
    {"b580-qk-xe2c-t128x64k32g24s16m8nf", kB580_qk_xe2c_t128x64k32g24s16m8nf, 1},
    {"b580-qk-xe2c-t64x128k32g44s16m8nf", kB580_qk_xe2c_t64x128k32g44s16m8nf, 1},
    {"b580-qk-xe2c-t64x64k32g44s16m8nf", kB580_qk_xe2c_t64x64k32g44s16m8nf, 1},
    {"b580-qk-xe2c-t128x64k64g44s16m8nf", kB580_qk_xe2c_t128x64k64g44s16m8nf, 1},
    {"b580-qk-xe2c-t64x128k32g42s16m8nf", kB580_qk_xe2c_t64x128k32g42s16m8nf, 1},
    {"b580-sdpa0", nullptr, 0},
    {"b580-refine0", kB580_refine0, sizeof(kB580_refine0) / sizeof(Preference)},
    {"b580-refine1", kB580_refine1, sizeof(kB580_refine1) / sizeof(Preference)},
    {"b580-dq-k64", kB580_dq_k64, sizeof(kB580_dq_k64) / sizeof(Preference)},
    // b580 end
};
const Profile* requested_profile() {
  static const Profile* p = []() -> const Profile* {
    const char* e = std::getenv("ET_VK_SARC_DEV_PROFILE");
    if (e == nullptr || *e == 0) {
      return nullptr;
    }
    for (const Profile& pr : kProfiles) {
      if (std::strcmp(pr.name, e) == 0) {
        return &pr;
      }
    }
    std::cerr << "[sarc_dev] unknown ET_VK_SARC_DEV_PROFILE=" << e << std::endl;
    std::abort();
  }();
  return p;
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
  if (const Profile* profile = requested_profile()) {
    if (table_choice.has_value()) {
      for (size_t i = 0; i < profile->count; i++) {
        const Preference& pref = profile->prefs[i];
        if (pref.op != shape.op ||
            (pref.shape_ok != nullptr && !pref.shape_ok(shape))) {
          continue;
        }
        for (const auto* store : {&candidates(), &rows()}) {
          for (const Row& row : *store) {
            if (row.op == shape.op &&
                ends_with(row.kernel_base, std::string("_") + pref.token) &&
                q4gsw_coopmat_fits(device, shape, row)) {
              return Choice{row.kernel_base, row.dims, row.rowmajor_a};
            }
          }
        }
      }
    }
    return table_choice;
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
    if (requested_profile() != nullptr) {
      std::cerr << "[sarc_dev] profile active: " << requested_profile()->name
                << std::endl;
    }
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
