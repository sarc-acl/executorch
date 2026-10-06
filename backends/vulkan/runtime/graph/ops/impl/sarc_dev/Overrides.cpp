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

// --- 780m begin (openspec/changes/sarc-1.5-780m-prefill-refine, 2026-10-04) ---
// Parameter-space candidates (Space780m.inc, generated by that change's
// tools/gen_space.py) and exact-name selection, one variable per op:
//   ET_VK_SARC_780M_{Q4,DQ,QK,AV}=<kernel base>
// and ET_VK_SARC_780M_PROFILE=<name> for a kernel per shape, the softmax
// variant and the fused attention kernels (below).
// A shape the kernel does not fit keeps what the selection above returned, so
// nothing falls back to the tiled kernels and the earlier variables still win
// when they disable an op.
namespace vkcompute {
namespace sarc {

// The profile's fused attention kernels, one per head_dim ("" = none), for
// 780m/Sdpa780mFused.cpp. That file needs the graph headers, which this file
// (GPU-free, test_sarc_select) must not include, so it hands its
// Override::sdpa_fused_* functions over from its own static initializer. The
// first Registrar above starts from an empty Override and the order of the two
// files' initializers is not defined: the pair is kept here and applied by
// whichever of the two runs last.
const char* sdpa_fused_variants_780m();
void register_sdpa_fused_780m(
    void (*add)(ComputeGraph&, const std::vector<int32_t>&),
    bool (*serves)(ComputeGraph*, const std::vector<int32_t>&));

namespace {

const Row k780mSpace[] = {
#include "Space780m.inc"
};

// ET_VK_SARC_780M_PROFILE=<name>: a kernel per shape among the parameter-space
// candidates. An entry applies to the shapes its predicate accepts and its
// kernel fits; the first one wins; a shape without an entry keeps the earlier
// selection (ET_VK_SARC_DEV_PROFILE and the table).
struct Pick780m {
  Op op;
  const char* kernel_base;
  bool (*shape_ok)(const ShapeInfo&);
};
// refine9: the 4w result of the round-2 search (that change's
// results/780m/space/confirm-4w). The 256-row tile halves how often the
// weights are staged and wins once K is large; N = 1024 and K below 4096 keep
// the 128 x 256 tile, now with B staged column-major; N = 8192 at K = 2048 is
// the one shape where the 16-subgroup grid of the 256-row tile is fastest.
bool q4_big_k(const ShapeInfo& s) {
  return s.N >= 2048 && s.K >= 4096;
}
bool q4_wide_small_k(const ShapeInfo& s) {
  return s.N >= 8192 && s.K < 3072;
}
bool q4_n_at_least_1024(const ShapeInfo& s) {
  return s.N >= 1024;
}
const Pick780m k780mRefine9[] = {
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g18s32f32bbt",
     q4_big_k},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32bbt",
     q4_wide_small_k},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t128x256k32g42s32f32cbt",
     q4_n_at_least_1024},
};
// refine10: refine9 after two more refinement rounds and a geometry scan
// (confirm2-4w): the in-Ash drain instead of the band drain on the 256-row
// tiles, their 2 x 4 grid at K = 3072, and a 2 x 4 grid of the 128 x 128 tile
// with column-major B for the small shapes at K = 2048.
bool q4_k_at_least_4096(const ShapeInfo& s) {
  return s.N >= 1024 && s.K >= 4096;
}
bool q4_k_3072(const ShapeInfo& s) {
  return s.N >= 2048 && s.K >= 3072 && s.K < 4096;
}
bool q4_k_below_3072(const ShapeInfo& s) {
  return s.K < 3072;
}
const Pick780m k780mRefine10[] = {
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g18s32f32cbt",
     q4_k_at_least_4096},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbt",
     q4_k_3072},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbt",
     q4_wide_small_k},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t128x128k32g24s32f32cbt",
     q4_k_below_3072},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t128x256k32g42s32f32cbt",
     q4_n_at_least_1024},
};
// refine11: refine10 plus the 8da4w result of the round-2 search (that change's
// results/780m/space/confirm-8da4w, all 2,238 surviving configurations
// screened): the 256 x 64 tile with K-chunks of 64 where N >= 2048 and
// K <= 4096 (wq_wo, w1_w3). The other shapes are tied with the 780m-refine3
// kernel and keep it.
bool dq_wide_k_up_to_4096(const ShapeInfo& s) {
  return s.N >= 2048 && s.K <= 4096;
}
const Pick780m k780mRefine11[] = {
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g18s32f32cbt",
     q4_k_at_least_4096},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbt",
     q4_k_3072},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbt",
     q4_wide_small_k},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t128x128k32g24s32f32cbt",
     q4_k_below_3072},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t128x256k32g42s32f32cbt",
     q4_n_at_least_1024},
    {Op::kDq8caLinear,
     "sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1",
     dq_wide_k_up_to_4096},
};
// A profile may also name a softmax variant (Override::softmax_variant:
// glsl/sarc_dev/sarc_sdpa_attn_weights_softmax_780m_<tag>.yaml) and the fused
// attention kernels (Sdpa780mFused.cpp). c7 to c11 are that change's
// candidates 7 to 11, each on top of ET_VK_SARC_DEV_PROFILE=780m-refine3.
// Softmax r3 is only valid in front of a SARC attn*V kernel.
struct Profile780m {
  const char* name;
  const Pick780m* picks;
  size_t count;
  const char* softmax_variant;
  const char* fused;
};
const char kFused780mTwoPass[] =
    "fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk";
const char kFused780mOnePass[] =
    "fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko";
const Profile780m k780mProfiles[] = {
    {"refine9", k780mRefine9, sizeof(k780mRefine9) / sizeof(Pick780m), nullptr, ""},
    {"refine10", k780mRefine10, sizeof(k780mRefine10) / sizeof(Pick780m), nullptr, ""},
    {"softmax-r1", nullptr, 0, "780m_r1", ""},
    {"c7", nullptr, 0, "780m_r3", ""},
    {"c8", nullptr, 0, "780m_r3", kFused780mTwoPass},
    {"c9", k780mRefine9, sizeof(k780mRefine9) / sizeof(Pick780m), "780m_r3", kFused780mOnePass},
    {"c10", k780mRefine10, sizeof(k780mRefine10) / sizeof(Pick780m), "780m_r3", kFused780mOnePass},
    {"c11", k780mRefine11, sizeof(k780mRefine11) / sizeof(Pick780m), "780m_r3", kFused780mOnePass},
};
const Profile780m* active_profile_780m() {
  static const Profile780m* const active = []() -> const Profile780m* {
    const char* e = std::getenv("ET_VK_SARC_780M_PROFILE");
    for (const Profile780m& p : k780mProfiles) {
      if (e != nullptr && std::string(e) == p.name) {
        return &p;
      }
    }
    return nullptr;
  }();
  return active;
}

std::optional<Choice> (*select_before_780m)(
    const DeviceInfo&,
    const ShapeInfo&,
    const std::optional<Choice>&) = nullptr;

std::optional<Choice> select_780m(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const std::optional<Choice>& table_choice) {
  static const std::string wanted[] = {
      std::getenv("ET_VK_SARC_780M_Q4") ? std::getenv("ET_VK_SARC_780M_Q4") : "",
      std::getenv("ET_VK_SARC_780M_DQ") ? std::getenv("ET_VK_SARC_780M_DQ") : "",
      std::getenv("ET_VK_SARC_780M_QK") ? std::getenv("ET_VK_SARC_780M_QK") : "",
      std::getenv("ET_VK_SARC_780M_AV") ? std::getenv("ET_VK_SARC_780M_AV") : "",
  };
  const std::optional<Choice> before =
      select_before_780m(device, shape, table_choice);
  const std::string& want = wanted[static_cast<size_t>(shape.op)];
  const Profile780m* active = active_profile_780m();
  if (want.empty() && active != nullptr && before.has_value()) {
    for (size_t i = 0; i < active->count; ++i) {
      const Pick780m& pick = active->picks[i];
      if (pick.op != shape.op || !pick.shape_ok(shape)) {
        continue;
      }
      for (const Row& row : candidates()) {
        if (row.op == shape.op && row.kernel_base == std::string(pick.kernel_base) &&
            q4gsw_coopmat_fits(device, shape, row)) {
          return Choice{row.kernel_base, row.dims, row.rowmajor_a};
        }
      }
    }
    return before;
  }
  if (want.empty() || !before.has_value()) {
    return before;
  }
  for (const Row& row : candidates()) {
    if (row.op == shape.op && want == row.kernel_base &&
        q4gsw_coopmat_fits(device, shape, row)) {
      return Choice{row.kernel_base, row.dims, row.rowmajor_a};
    }
  }
  return before;
}

Override& fused_780m() {
  static Override fused;
  return fused;
}

struct Registrar780m {
  Registrar780m() {
    register_candidates(k780mSpace, sizeof(k780mSpace) / sizeof(k780mSpace[0]));
    Override o = get_override();
    select_before_780m = o.select;
    o.select = select_780m;
    const Profile780m* active = active_profile_780m();
    if (active != nullptr && std::getenv("ET_VK_DISABLE_COOPMAT") == nullptr) {
      o.softmax_variant = active->softmax_variant;
    }
    o.sdpa_fused_add = fused_780m().sdpa_fused_add;
    o.sdpa_fused_serves = fused_780m().sdpa_fused_serves;
    set_override(o);
  }
} registrar_780m;

} // namespace

void register_sdpa_fused_780m(
    void (*add)(ComputeGraph&, const std::vector<int32_t>&),
    bool (*serves)(ComputeGraph*, const std::vector<int32_t>&)) {
  fused_780m().sdpa_fused_add = add;
  fused_780m().sdpa_fused_serves = serves;
  Override o = get_override();
  o.sdpa_fused_add = add;
  o.sdpa_fused_serves = serves;
  set_override(o);
}

const char* sdpa_fused_variants_780m() {
  const Profile780m* active = active_profile_780m();
  return active != nullptr ? active->fused : "";
}
} // namespace sarc
} // namespace vkcompute
// --- 780m end ---

// --- m51 begin (openspec/changes/sarc-1.5-m51-prefill-refine, 2026-10-06) ---
// ET_VK_SARC_M51_PROFILE=<name>: the M51 candidates, on top of
// ET_VK_SARC_UNVERIFIED=1 (the xclipse rows). A profile names the fused
// attention kernels per head_dim (m51/SdpaM51Fused.cpp); shapes and calls they
// do not cover keep what the selection above returns. Not meant together with
// ET_VK_SARC_780M_PROFILE: for the fused node the M51 profile wins.
namespace vkcompute {
namespace sarc {
namespace {

// 4w per-shape screen (glsl/sarc_dev/sarc_linear_q4gsw_coopmat_sweep.yaml, m51
// block): the xclipse row's fp32 accumulation and one-pass texture3d drain on
// other tiles; and the phase-timing twin of the row (MEASUREMENT ONLY,
// glsl/sarc_dev/sarc_dev_prof_q4gsw.yaml). Selected with ET_VK_SARC_Q4GSW_VARIANT.
constexpr TileDims xp_tile(
    uint32_t m, uint32_t n, uint32_t k, uint32_t sgx, uint32_t sgy, uint32_t sg) {
  return {m, n, k, sgx, sgy, sg, 16, false, /*csh_full=*/true, /*csh_pool=*/true};
}
const Row kM51Q4[] = {
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g22s32f32xp", xp_tile(128, 128, 32, 2, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k16g42s32f32xp", xp_tile(128, 128, 16, 4, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32xp", xp_tile(128, 128, 32, 4, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k16g24s32f32xp", xp_tile(128, 128, 16, 2, 4, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x64k16g22s32f32xp", xp_tile(128, 64, 16, 2, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t64x128k16g22s32f32xp", xp_tile(64, 128, 16, 2, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k16g21s64f32xp", xp_tile(128, 128, 16, 2, 1, 64),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g21s64f32xp", xp_tile(128, 128, 32, 2, 1, 64),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k16g22s64f32xp", xp_tile(128, 128, 16, 2, 2, 64),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k16g12s64f32xp", xp_tile(128, 128, 16, 1, 2, 64),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_q4gsw_t128x128k16g22s32f32xpp", xp_tile(128, 128, 16, 2, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
};

struct ProfileM51 {
  const char* name;
  const char* fused;
};
const ProfileM51 kM51Profiles[] = {
    {"c1", "fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk"},
};
const ProfileM51* active_profile_m51() {
  static const ProfileM51* const active = []() -> const ProfileM51* {
    const char* e = std::getenv("ET_VK_SARC_M51_PROFILE");
    if (e == nullptr || *e == 0) {
      return nullptr;
    }
    for (const ProfileM51& p : kM51Profiles) {
      if (std::strcmp(p.name, e) == 0) {
        return &p;
      }
    }
    std::cerr << "[sarc_dev] unknown ET_VK_SARC_M51_PROFILE=" << e << std::endl;
    std::abort();
  }();
  return active;
}

Override& fused_m51() {
  static Override fused;
  return fused;
}

std::optional<Choice> (*select_before_m51)(
    const DeviceInfo&,
    const ShapeInfo&,
    const std::optional<Choice>&) = nullptr;

// The fused node registers from another file, and the 780M's node does too, in
// an order the linker decides. So the M51 node is installed into the override
// on the first selection, which every SDPA node makes before its launch geometry
// asks whether the fused node serves its call.
void install_fused_m51() {
  static const bool installed = [] {
    if (fused_m51().sdpa_fused_add == nullptr ||
        (active_profile_m51() == nullptr &&
         std::getenv("ET_VK_SARC_M51_SDPA_FUSED") == nullptr)) {
      return false;
    }
    Override o = get_override();
    o.sdpa_fused_add = fused_m51().sdpa_fused_add;
    o.sdpa_fused_serves = fused_m51().sdpa_fused_serves;
    set_override(o);
    return true;
  }();
  (void)installed;
}

std::optional<Choice> select_m51(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const std::optional<Choice>& table_choice) {
  install_fused_m51();
  return select_before_m51(device, shape, table_choice);
}

struct RegistrarM51 {
  RegistrarM51() {
    register_candidates(kM51Q4, sizeof(kM51Q4) / sizeof(kM51Q4[0]));
    Override o = get_override();
    select_before_m51 = o.select;
    o.select = select_m51;
    set_override(o);
    if (active_profile_m51() != nullptr) {
      std::cerr << "[sarc_dev] m51 profile active: " << active_profile_m51()->name
                << std::endl;
    }
  }
} registrar_m51;

} // namespace

void register_sdpa_fused_m51(
    void (*add)(ComputeGraph&, const std::vector<int32_t>&),
    bool (*serves)(ComputeGraph*, const std::vector<int32_t>&)) {
  fused_m51().sdpa_fused_add = add;
  fused_m51().sdpa_fused_serves = serves;
}

const char* sdpa_fused_variants_m51() {
  const ProfileM51* active = active_profile_m51();
  return active != nullptr ? active->fused : "";
}
} // namespace sarc
} // namespace vkcompute
// --- m51 end ---
