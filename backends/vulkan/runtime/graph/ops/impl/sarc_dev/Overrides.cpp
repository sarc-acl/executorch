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
    // >>> 4070ti lin-q4gsw-rows
    // RTX 4070 Ti SUPER 4w sweep tiles (glsl/sarc_dev/sarc_linear_q4gsw_coopmat_4070ti.yaml), texture3d.
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t256x128k16g42s32gac", {256, 128, 16, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t256x128k16g24s32gac", {256, 128, 16, 2, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t256x256k16g44s32gac", {256, 256, 16, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t128x256k16g42s32ga", {128, 256, 16, 4, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t128x256k16g42s32gac", {128, 256, 16, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t128x128k32g42s32ga", {128, 128, 32, 4, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t128x128k32g42s32gac", {128, 128, 32, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t128x128k32g44s32gac", {128, 128, 32, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t128x128k32g24s32gac", {128, 128, 32, 2, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t128x128k16g42s32ga", {128, 128, 16, 4, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t128x128k16g42s32gac", {128, 128, 16, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t256x128k16g42s32gabt", {256, 128, 16, 4, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t256x128k16g42s32gacbt", {256, 128, 16, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_4070ti_t128x128k32g42s32gacbt", {128, 128, 32, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    // <<< 4070ti lin-q4gsw-rows
    // >>> orin prof-q4gsw-rows
    // Jetson Orin phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_orin_q4gsw.yaml): the shipped
    // Orin 4w tiles with shader-clock phase counters written over their output.
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_orin_q4gsw_t256x128k16g22s32p", {256, 128, 16, 2, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_orin_q4gsw_t128x128k32g42s32f32p", {128, 128, 32, 4, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_orin_q4gsw_t128x128k16g22s32p", {128, 128, 16, 2, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    // <<< orin prof-q4gsw-rows
    // >>> orin q4-rows
    // Jetson Orin 4w sweep variants of the shipped tiles (glsl/sarc_dev/sarc_linear_q4gsw_coopmat_orin.yaml).
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t256x128k16g22s32", {256, 128, 16, 2, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t256x128k16g22s32bt", {256, 128, 16, 2, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t256x128k16g22s32c", {256, 128, 16, 2, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t256x128k16g22s32cbt", {256, 128, 16, 2, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t128x128k16g22s32bt", {128, 128, 16, 2, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t256x128k16g42s32", {256, 128, 16, 4, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t256x128k16g42s32bt", {256, 128, 16, 4, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t256x128k16g24s32", {256, 128, 16, 2, 4, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t256x128k16g24s32bt", {256, 128, 16, 2, 4, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t128x128k32g42s32f32bt", {128, 128, 32, 4, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t256x128k16g44s32bt", {256, 128, 16, 4, 4, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_orin_t128x128k32g22s32bt", {128, 128, 32, 2, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    // <<< orin q4-rows
    // >>> 4070ti prof-q4gsw-rows
    // RTX 4070 Ti SUPER phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_4070ti_q4gsw.yaml):
    // the shipped `ga` tiles with shader-clock phase counters written over their output.
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_4070ti_q4gsw_t256x128k16g42s32gap", {256, 128, 16, 4, 2, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_4070ti_q4gsw_t128x128k16g24s32gap", {128, 128, 16, 2, 4, 32, 16, false},
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_4070ti_q4gsw_t128x256k16g42s32gap", {128, 256, 16, 4, 2, 32, 16, false},
     kBufTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_4070ti_q4gsw_t128x128k16g42s32gap", {128, 128, 16, 4, 2, 32, 16, false},
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},
    // <<< 4070ti prof-q4gsw-rows
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
    // >>> 4070ti lin-dq8ca-rows
    // RTX 4070 Ti SUPER 8da4w zpgtr sweep tiles (glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpgtr_4070ti.yaml).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t128x128k64g42s32mk32ra", {128, 128, 64, 4, 2, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t128x128k64g24s32mk32ra", {128, 128, 64, 2, 4, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t256x128k32g44s32mk32ra", {256, 128, 32, 4, 4, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t256x128k32g42s32mk32ra", {256, 128, 32, 4, 2, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t128x256k32g81s32mk32ra", {128, 256, 32, 8, 1, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t128x128k32g42s32mk32ra", {128, 128, 32, 4, 2, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t128x64k64g44s32mk32ra", {128, 64, 64, 4, 4, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t128x64k64g42s32mk32ra", {128, 64, 64, 4, 2, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t64x128k64g42s32mk32ra", {64, 128, 64, 4, 2, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t256x64k64g44s32mk32ra", {256, 64, 64, 4, 4, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t256x64k64g42s32mk32ra", {256, 64, 64, 4, 2, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t256x128k32g24s32mk32ra", {256, 128, 32, 2, 4, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_t128x128k64g22s32mk32ra", {128, 128, 64, 2, 2, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    // <<< 4070ti lin-dq8ca-rows
    // >>> 4070ti bh-dq8ca-rows
    // RTX 4070 Ti SUPER zpgtr with half-texel weight staging (glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpgtr_4070ti_bh.yaml).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_bh_t128x128k64g44s32mk32ra", {128, 128, 64, 4, 4, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_bh_t128x128k64g42s32mk32ra", {128, 128, 64, 4, 2, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_bh_t128x128k64g24s32mk32ra", {128, 128, 64, 2, 4, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_4070ti_bh_t256x128k32g42s32mk32ra", {256, 128, 32, 4, 2, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    // <<< 4070ti bh-dq8ca-rows
    // >>> orin bf-dq8ca-rows
    // Jetson Orin zpgtr with whole-texel weight staging (glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpgtr_orin_bf.yaml).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g44s32mk32ra", {128, 128, 64, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g42s32mk32ra", {128, 128, 64, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g24s32mk32ra", {128, 128, 64, 2, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t256x128k32g44s32mk32ra", {256, 128, 32, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf1_t128x128k128g44s32mk32ra", {128, 128, 128, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf1_t128x128k128g42s32mk32ra", {128, 128, 128, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf1_t128x128k128g24s32mk32ra", {128, 128, 128, 2, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf1_t256x128k64g44s32mk32ra", {256, 128, 64, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf1_t128x128k64g42s32mk32ra", {128, 128, 64, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t64x128k64g42s32mk32ra", {64, 128, 64, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t64x128k64g22s32mk32ra", {64, 128, 64, 2, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t64x64k128g24s32mk32ra", {64, 64, 128, 2, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t64x64k128g42s32mk32ra", {64, 64, 128, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t64x64k64g22s32mk32ra", {64, 64, 64, 2, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g22s32mk32ra", {128, 128, 64, 2, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    // <<< orin bf-dq8ca-rows
    // >>> orin prof-dq8ca-rows
    // Jetson Orin phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_orin_dq8ca_bf.yaml).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t128x128k64g44s32mk32rap", {128, 128, 64, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t128x128k64g42s32mk32rap", {128, 128, 64, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t128x128k64g24s32mk32rap", {128, 128, 64, 2, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t256x128k32g44s32mk32rap", {256, 128, 32, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf1_t128x128k128g44s32mk32rap", {128, 128, 128, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf1_t128x128k128g42s32mk32rap", {128, 128, 128, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf1_t128x128k128g24s32mk32rap", {128, 128, 128, 2, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf1_t256x128k64g44s32mk32rap", {256, 128, 64, 4, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf1_t128x128k64g42s32mk32rap", {128, 128, 64, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t64x128k64g42s32mk32rap", {64, 128, 64, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t64x128k64g22s32mk32rap", {64, 128, 64, 2, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t64x64k128g24s32mk32rap", {64, 64, 128, 2, 4, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t64x64k128g42s32mk32rap", {64, 64, 128, 4, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t64x64k64g22s32mk32rap", {64, 64, 64, 2, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_orin_dq8ca_bf_orin_bf_t128x128k64g22s32mk32rap", {128, 128, 64, 2, 2, 32, 16, true},
     kTex3dTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    // <<< orin prof-dq8ca-rows
    // >>> 4070ti prof-dq8ca-rows
    // RTX 4070 Ti SUPER phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_4070ti_dq8ca_zpgtr.yaml).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_4070ti_dq8ca_zpgtr_t128x128k64g44s32mk32rap", {128, 128, 64, 4, 4, 32, 16, true},
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    // <<< 4070ti prof-dq8ca-rows
    // >>> 4070ti sdpa-rows
    // RTX 4070 Ti SUPER prefill refine: glsl/sarc_dev/sarc_sdpa_{qk,av}_coopmat_4070ti*.yaml.
    // Selected only through ET_VK_SARC_DEV_PROFILE=4070ti-*.
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_t128x64k32g42s32nf",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_t128x64k32g42s32",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_t128x64k32g24s32nf",
     {128, 64, 32, 2, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_t128x64k32g22s32nf",
     {128, 64, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t128x64k32g42s32nf",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t128x64k64g42s32nf",
     {128, 64, 64, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t128x64k32g24s32nf",
     {128, 64, 32, 2, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t64x64k32g22s32nf",
     {64, 64, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t64x64k32g42s32nf",
     {64, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t64x128k32g42s32nf",
     {64, 128, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t128x64k32g44s32nf",
     {128, 64, 32, 4, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t64x64k32g44s32nf",
     {64, 64, 32, 4, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t64x64k32g21s32nf",
     {64, 64, 32, 2, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_pk_t32x64k32g42s32nf",
     {32, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_t64x64k32g42s32",
     {64, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_t64x64k32g24s32",
     {64, 64, 32, 2, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t64x128k32g42s32",
     {64, 128, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t128x128k32g42s32",
     {128, 128, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t128x64k32g42s32",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t128x64k32g44s32",
     {128, 64, 32, 4, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t256x64k32g42s32",
     {256, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t128x64k32g24s32",
     {128, 64, 32, 2, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t64x128k32g44s32",
     {64, 128, 32, 4, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t128x128k32g44s32",
     {128, 128, 32, 4, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t32x64k32g42s32",
     {32, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_ml_t256x64k32g44s32",
     {256, 64, 32, 4, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    // <<< 4070ti sdpa-rows
    // >>> orin qk-rows
    // Jetson Orin: packed-staging QK^T, more tiles (glsl/sarc_dev/sarc_sdpa_qk_coopmat_orin_pk.yaml).
    // Selected only through ET_VK_SARC_DEV_PROFILE=orin-*.
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t64x64k64g22s32nf",
     {64, 64, 64, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t64x64k64g21s32nf",
     {64, 64, 64, 2, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t64x64k64g42s32nf",
     {64, 64, 64, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t128x64k64g24s32nf",
     {128, 64, 64, 2, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t128x64k64g44s32nf",
     {128, 64, 64, 4, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t128x64k64g22s32nf",
     {128, 64, 64, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t64x128k64g42s32nf",
     {64, 128, 64, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t32x64k64g42s32nf",
     {32, 64, 64, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t64x64k128g22s32nf",
     {64, 64, 128, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t64x64k128g42s32nf",
     {64, 64, 128, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t64x64k128g21s32nf",
     {64, 64, 128, 2, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_orin_pk_t64x64k128g44s32nf",
     {64, 64, 128, 4, 4, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    // <<< orin qk-rows
    // >>> 4070ti df-rows
    // RTX 4070 Ti SUPER direct-feed SDPA kernels: glsl/sarc_dev/sarc_sdpa_{qk,av}_coopmat_4070ti_df.yaml.
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_df_t64x64k32g11s32nf",
     {64, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_df_t32x64k32g11s32nf",
     {32, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_df_t64x32k32g11s32nf",
     {64, 32, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_df_t32x32k32g11s32nf",
     {32, 32, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_df_t16x64k32g11s32nf",
     {16, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_df_t128x64k32g22s32nf",
     {128, 64, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_df_t128x64k32g42s32nf",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_df_t64x64k32g22s32nf",
     {64, 64, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_df_t128x128k32g22s32nf",
     {128, 128, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_dfg_t64x64k32g11s32nf",
     {64, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_dfg_t32x64k32g11s32nf",
     {32, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_dfh_t64x64k32g11s32nf",
     {64, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_dfh_t32x64k32g11s32nf",
     {32, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_dfg_t64x64k32g22s32nf",
     {64, 64, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_dfh_t64x64k32g22s32nf",
     {64, 64, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t16x64k32g11s32",
     {16, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t32x64k32g11s32",
     {32, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t64x64k32g11s32",
     {64, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t64x64k32g12s32",
     {64, 64, 32, 1, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t32x64k32g21s32",
     {32, 64, 32, 2, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t16x128k32g11s32",
     {16, 128, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t32x128k32g11s32",
     {32, 128, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t32x128k32g21s32",
     {32, 128, 32, 2, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t64x128k32g21s32",
     {64, 128, 32, 2, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_df_t64x128k32g22s32",
     {64, 128, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_dfg_t32x64k32g11s32",
     {32, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_dfg_t32x128k32g11s32",
     {32, 128, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_dfg_t32x128k32g21s32",
     {32, 128, 32, 2, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_dfh_t32x64k32g11s32",
     {32, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_dfh_t32x128k32g11s32",
     {32, 128, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_dfh_t32x128k32g21s32",
     {32, 128, 32, 2, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_dfg_t64x64k32g11s32",
     {64, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_dfh_t64x64k32g11s32",
     {64, 64, 32, 1, 1, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    // <<< 4070ti df-rows
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
// >>> 4070ti sdpa-preferences
// RTX 4070 Ti SUPER: single-kernel SDPA screening profiles and 4070ti-refine1.
const Preference k4070ti_qk_t128x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_t128x64k32g42s32nf", nullptr}};
const Preference k4070ti_qk_t128x64k32g42s32[] = {{Op::kSdpaQk, "4070ti_t128x64k32g42s32", nullptr}};
const Preference k4070ti_qk_t128x64k32g24s32nf[] = {{Op::kSdpaQk, "4070ti_t128x64k32g24s32nf", nullptr}};
const Preference k4070ti_qk_t128x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_t128x64k32g22s32nf", nullptr}};
const Preference k4070ti_qk_pk_t128x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t128x64k32g42s32nf", nullptr}};
const Preference k4070ti_qk_pk_t128x64k64g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t128x64k64g42s32nf", nullptr}};
const Preference k4070ti_qk_pk_t128x64k32g24s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t128x64k32g24s32nf", nullptr}};
const Preference k4070ti_qk_pk_t64x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x64k32g22s32nf", nullptr}};
const Preference k4070ti_qk_pk_t64x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x64k32g42s32nf", nullptr}};
const Preference k4070ti_qk_pk_t64x128k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x128k32g42s32nf", nullptr}};
const Preference k4070ti_qk_pk_t128x64k32g44s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t128x64k32g44s32nf", nullptr}};
const Preference k4070ti_qk_pk_t64x64k32g44s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x64k32g44s32nf", nullptr}};
const Preference k4070ti_qk_pk_t64x64k32g21s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x64k32g21s32nf", nullptr}};
const Preference k4070ti_qk_pk_t32x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t32x64k32g42s32nf", nullptr}};
const Preference k4070ti_av_t64x64k32g42s32[] = {{Op::kSdpaAv, "4070ti_t64x64k32g42s32", nullptr}};
const Preference k4070ti_av_t64x64k32g24s32[] = {{Op::kSdpaAv, "4070ti_t64x64k32g24s32", nullptr}};
const Preference k4070ti_av_ml_t64x128k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr}};
const Preference k4070ti_av_ml_t128x128k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x128k32g42s32", nullptr}};
const Preference k4070ti_av_ml_t128x64k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x64k32g42s32", nullptr}};
const Preference k4070ti_av_ml_t128x64k32g44s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x64k32g44s32", nullptr}};
const Preference k4070ti_av_ml_t256x64k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t256x64k32g42s32", nullptr}};
const Preference k4070ti_av_ml_t128x64k32g24s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x64k32g24s32", nullptr}};
const Preference k4070ti_av_ml_t64x128k32g44s32[] = {{Op::kSdpaAv, "4070ti_ml_t64x128k32g44s32", nullptr}};
const Preference k4070ti_av_ml_t128x128k32g44s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x128k32g44s32", nullptr}};
const Preference k4070ti_av_ml_t32x64k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t32x64k32g42s32", nullptr}};
const Preference k4070ti_av_ml_t256x64k32g44s32[] = {{Op::kSdpaAv, "4070ti_ml_t256x64k32g44s32", nullptr}};
// Candidate 1 (SDPA prefill kernels for this device), the best tile per head_dim of screen 2
// (results/4070ti/screens/sdpa-screen2.csv), accumulation unchanged (fp32):
//   QK^T    head_dim 64: direct feed t64x64 (one subgroup per workgroup); 128: packed staging t64x128
//   attn*V  head_dim 64: multi-pass staging t32x64; 128: t64x128 (does not fit head_dim 64)
bool head_dim_64_4070ti(const ShapeInfo& s) {
  return (s.op == Op::kSdpaQk ? s.K : s.N) == 64;
}
const Preference k4070tiRefine1[] = {
    {Op::kSdpaQk, "4070ti_df_t64x64k32g11s32nf", head_dim_64_4070ti},
    {Op::kSdpaQk, "4070ti_pk_t64x128k32g42s32nf", nullptr},
    {Op::kSdpaAv, "4070ti_ml_t32x64k32g42s32", head_dim_64_4070ti},
    {Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr},
};
// <<< 4070ti sdpa-preferences
// >>> 4070ti df-preferences
// RTX 4070 Ti SUPER: single-kernel screening profiles of the direct-feed SDPA kernels.
const Preference k4070ti_qk_df_t64x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t64x64k32g11s32nf", nullptr}};
const Preference k4070ti_qk_df_t32x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t32x64k32g11s32nf", nullptr}};
const Preference k4070ti_qk_df_t64x32k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t64x32k32g11s32nf", nullptr}};
const Preference k4070ti_qk_df_t32x32k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t32x32k32g11s32nf", nullptr}};
const Preference k4070ti_qk_df_t16x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t16x64k32g11s32nf", nullptr}};
const Preference k4070ti_qk_df_t128x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_df_t128x64k32g22s32nf", nullptr}};
const Preference k4070ti_qk_df_t128x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_df_t128x64k32g42s32nf", nullptr}};
const Preference k4070ti_qk_df_t64x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_df_t64x64k32g22s32nf", nullptr}};
const Preference k4070ti_qk_df_t128x128k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_df_t128x128k32g22s32nf", nullptr}};
const Preference k4070ti_qk_dfg_t64x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_dfg_t64x64k32g11s32nf", nullptr}};
const Preference k4070ti_qk_dfg_t32x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_dfg_t32x64k32g11s32nf", nullptr}};
const Preference k4070ti_qk_dfh_t64x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_dfh_t64x64k32g11s32nf", nullptr}};
const Preference k4070ti_qk_dfh_t32x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_dfh_t32x64k32g11s32nf", nullptr}};
const Preference k4070ti_qk_dfg_t64x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_dfg_t64x64k32g22s32nf", nullptr}};
const Preference k4070ti_qk_dfh_t64x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_dfh_t64x64k32g22s32nf", nullptr}};
const Preference k4070ti_av_df_t16x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t16x64k32g11s32", nullptr}};
const Preference k4070ti_av_df_t32x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t32x64k32g11s32", nullptr}};
const Preference k4070ti_av_df_t64x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t64x64k32g11s32", nullptr}};
const Preference k4070ti_av_df_t64x64k32g12s32[] = {{Op::kSdpaAv, "4070ti_df_t64x64k32g12s32", nullptr}};
const Preference k4070ti_av_df_t32x64k32g21s32[] = {{Op::kSdpaAv, "4070ti_df_t32x64k32g21s32", nullptr}};
const Preference k4070ti_av_df_t16x128k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t16x128k32g11s32", nullptr}};
const Preference k4070ti_av_df_t32x128k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t32x128k32g11s32", nullptr}};
const Preference k4070ti_av_df_t32x128k32g21s32[] = {{Op::kSdpaAv, "4070ti_df_t32x128k32g21s32", nullptr}};
const Preference k4070ti_av_df_t64x128k32g21s32[] = {{Op::kSdpaAv, "4070ti_df_t64x128k32g21s32", nullptr}};
const Preference k4070ti_av_df_t64x128k32g22s32[] = {{Op::kSdpaAv, "4070ti_df_t64x128k32g22s32", nullptr}};
const Preference k4070ti_av_dfg_t32x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfg_t32x64k32g11s32", nullptr}};
const Preference k4070ti_av_dfg_t32x128k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfg_t32x128k32g11s32", nullptr}};
const Preference k4070ti_av_dfg_t32x128k32g21s32[] = {{Op::kSdpaAv, "4070ti_dfg_t32x128k32g21s32", nullptr}};
const Preference k4070ti_av_dfh_t32x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfh_t32x64k32g11s32", nullptr}};
const Preference k4070ti_av_dfh_t32x128k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfh_t32x128k32g11s32", nullptr}};
const Preference k4070ti_av_dfh_t32x128k32g21s32[] = {{Op::kSdpaAv, "4070ti_dfh_t32x128k32g21s32", nullptr}};
const Preference k4070ti_av_dfg_t64x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfg_t64x64k32g11s32", nullptr}};
const Preference k4070ti_av_dfh_t64x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfh_t64x64k32g11s32", nullptr}};
// <<< 4070ti df-preferences
// >>> 4070ti lin-preferences
// RTX 4070 Ti SUPER linear profiles (tools/gen_4070ti_profiles.py).
bool n_above_512_4070ti(const ShapeInfo& s) {
  return s.N > 512;
}
const Preference k4070tiRefine2[] = {
    {Op::kDq8caLinear, "bh_t128x128k64g44s32mk32ra", nullptr},
};
const Preference k4070tiRefine3[] = {
    {Op::kDq8caLinear, "bh_t128x128k64g44s32mk32ra", nullptr},
    {Op::kQ4gswLinear, "4070ti_t256x128k16g42s32gac", n_above_512_4070ti},
};
const Preference k4070tiRefine4[] = {
    {Op::kSdpaQk, "4070ti_df_t64x64k32g11s32nf", head_dim_64_4070ti},
    {Op::kSdpaQk, "4070ti_pk_t64x128k32g42s32nf", nullptr},
    {Op::kSdpaAv, "4070ti_ml_t32x64k32g42s32", head_dim_64_4070ti},
    {Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr},
    {Op::kDq8caLinear, "bh_t128x128k64g44s32mk32ra", nullptr},
};
const Preference k4070tiRefine5[] = {
    {Op::kSdpaQk, "4070ti_df_t64x64k32g11s32nf", head_dim_64_4070ti},
    {Op::kSdpaQk, "4070ti_pk_t64x128k32g42s32nf", nullptr},
    {Op::kSdpaAv, "4070ti_ml_t32x64k32g42s32", head_dim_64_4070ti},
    {Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr},
    {Op::kDq8caLinear, "bh_t128x128k64g44s32mk32ra", nullptr},
    {Op::kQ4gswLinear, "4070ti_t256x128k16g42s32gac", n_above_512_4070ti},
};
// <<< 4070ti lin-preferences
// >>> orin sdpa-preferences
// Jetson Orin (tools/gen_orin_sdpa.py): single-kernel SDPA screening profiles and orin-refineN.
bool k_above_8192_orin(const ShapeInfo& s) {
  return s.K > 8192;
}
bool orin_256_shape(const ShapeInfo& s) {
  return s.K <= 8192 && (s.M != 256 || s.N % 2048 == 0);
}
const Preference kOrin_av_df_t16x128k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t16x128k32g11s32", nullptr}};
const Preference kOrin_av_df_t16x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t16x64k32g11s32", nullptr}};
const Preference kOrin_av_df_t32x128k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t32x128k32g11s32", nullptr}};
const Preference kOrin_av_df_t32x128k32g21s32[] = {{Op::kSdpaAv, "4070ti_df_t32x128k32g21s32", nullptr}};
const Preference kOrin_av_df_t32x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t32x64k32g11s32", nullptr}};
const Preference kOrin_av_df_t32x64k32g21s32[] = {{Op::kSdpaAv, "4070ti_df_t32x64k32g21s32", nullptr}};
const Preference kOrin_av_df_t64x128k32g21s32[] = {{Op::kSdpaAv, "4070ti_df_t64x128k32g21s32", nullptr}};
const Preference kOrin_av_df_t64x128k32g22s32[] = {{Op::kSdpaAv, "4070ti_df_t64x128k32g22s32", nullptr}};
const Preference kOrin_av_df_t64x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_df_t64x64k32g11s32", nullptr}};
const Preference kOrin_av_df_t64x64k32g12s32[] = {{Op::kSdpaAv, "4070ti_df_t64x64k32g12s32", nullptr}};
const Preference kOrin_av_dfg_t32x128k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfg_t32x128k32g11s32", nullptr}};
const Preference kOrin_av_dfg_t32x128k32g21s32[] = {{Op::kSdpaAv, "4070ti_dfg_t32x128k32g21s32", nullptr}};
const Preference kOrin_av_dfg_t32x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfg_t32x64k32g11s32", nullptr}};
const Preference kOrin_av_dfg_t64x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfg_t64x64k32g11s32", nullptr}};
const Preference kOrin_av_dfh_t32x128k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfh_t32x128k32g11s32", nullptr}};
const Preference kOrin_av_dfh_t32x128k32g21s32[] = {{Op::kSdpaAv, "4070ti_dfh_t32x128k32g21s32", nullptr}};
const Preference kOrin_av_dfh_t32x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfh_t32x64k32g11s32", nullptr}};
const Preference kOrin_av_dfh_t64x64k32g11s32[] = {{Op::kSdpaAv, "4070ti_dfh_t64x64k32g11s32", nullptr}};
const Preference kOrin_av_ml_t128x128k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x128k32g42s32", nullptr}};
const Preference kOrin_av_ml_t128x128k32g44s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x128k32g44s32", nullptr}};
const Preference kOrin_av_ml_t128x64k32g24s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x64k32g24s32", nullptr}};
const Preference kOrin_av_ml_t128x64k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x64k32g42s32", nullptr}};
const Preference kOrin_av_ml_t128x64k32g44s32[] = {{Op::kSdpaAv, "4070ti_ml_t128x64k32g44s32", nullptr}};
const Preference kOrin_av_ml_t256x64k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t256x64k32g42s32", nullptr}};
const Preference kOrin_av_ml_t256x64k32g44s32[] = {{Op::kSdpaAv, "4070ti_ml_t256x64k32g44s32", nullptr}};
const Preference kOrin_av_ml_t32x64k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t32x64k32g42s32", nullptr}};
const Preference kOrin_av_ml_t64x128k32g42s32[] = {{Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr}};
const Preference kOrin_av_ml_t64x128k32g44s32[] = {{Op::kSdpaAv, "4070ti_ml_t64x128k32g44s32", nullptr}};
const Preference kOrin_av_t64x64k32g24s32[] = {{Op::kSdpaAv, "4070ti_t64x64k32g24s32", nullptr}};
const Preference kOrin_av_t64x64k32g42s32[] = {{Op::kSdpaAv, "4070ti_t64x64k32g42s32", nullptr}};
const Preference kOrin_qk_df_t128x128k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_df_t128x128k32g22s32nf", nullptr}};
const Preference kOrin_qk_df_t128x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_df_t128x64k32g22s32nf", nullptr}};
const Preference kOrin_qk_df_t128x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_df_t128x64k32g42s32nf", nullptr}};
const Preference kOrin_qk_df_t16x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t16x64k32g11s32nf", nullptr}};
const Preference kOrin_qk_df_t32x32k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t32x32k32g11s32nf", nullptr}};
const Preference kOrin_qk_df_t32x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t32x64k32g11s32nf", nullptr}};
const Preference kOrin_qk_df_t64x32k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t64x32k32g11s32nf", nullptr}};
const Preference kOrin_qk_df_t64x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_df_t64x64k32g11s32nf", nullptr}};
const Preference kOrin_qk_df_t64x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_df_t64x64k32g22s32nf", nullptr}};
const Preference kOrin_qk_dfg_t32x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_dfg_t32x64k32g11s32nf", nullptr}};
const Preference kOrin_qk_dfg_t64x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_dfg_t64x64k32g11s32nf", nullptr}};
const Preference kOrin_qk_dfg_t64x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_dfg_t64x64k32g22s32nf", nullptr}};
const Preference kOrin_qk_dfh_t32x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_dfh_t32x64k32g11s32nf", nullptr}};
const Preference kOrin_qk_dfh_t64x64k32g11s32nf[] = {{Op::kSdpaQk, "4070ti_dfh_t64x64k32g11s32nf", nullptr}};
const Preference kOrin_qk_dfh_t64x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_dfh_t64x64k32g22s32nf", nullptr}};
const Preference kOrin_qk_pk_t128x64k32g24s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t128x64k32g24s32nf", nullptr}};
const Preference kOrin_qk_pk_t128x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t128x64k32g42s32nf", nullptr}};
const Preference kOrin_qk_pk_t128x64k32g44s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t128x64k32g44s32nf", nullptr}};
const Preference kOrin_qk_pk_t128x64k64g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t128x64k64g42s32nf", nullptr}};
const Preference kOrin_qk_pk_t32x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t32x64k32g42s32nf", nullptr}};
const Preference kOrin_qk_pk_t64x128k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x128k32g42s32nf", nullptr}};
const Preference kOrin_qk_pk_t64x64k32g21s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x64k32g21s32nf", nullptr}};
const Preference kOrin_qk_pk_t64x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x64k32g22s32nf", nullptr}};
const Preference kOrin_qk_pk_t64x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x64k32g42s32nf", nullptr}};
const Preference kOrin_qk_pk_t64x64k32g44s32nf[] = {{Op::kSdpaQk, "4070ti_pk_t64x64k32g44s32nf", nullptr}};
const Preference kOrin_qk_t128x64k32g22s32nf[] = {{Op::kSdpaQk, "4070ti_t128x64k32g22s32nf", nullptr}};
const Preference kOrin_qk_t128x64k32g24s32nf[] = {{Op::kSdpaQk, "4070ti_t128x64k32g24s32nf", nullptr}};
const Preference kOrin_qk_t128x64k32g42s32[] = {{Op::kSdpaQk, "4070ti_t128x64k32g42s32", nullptr}};
const Preference kOrin_qk_t128x64k32g42s32nf[] = {{Op::kSdpaQk, "4070ti_t128x64k32g42s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t128x64k64g22s32nf[] = {{Op::kSdpaQk, "orin_pk_t128x64k64g22s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t128x64k64g24s32nf[] = {{Op::kSdpaQk, "orin_pk_t128x64k64g24s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t128x64k64g44s32nf[] = {{Op::kSdpaQk, "orin_pk_t128x64k64g44s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t32x64k64g42s32nf[] = {{Op::kSdpaQk, "orin_pk_t32x64k64g42s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t64x128k64g42s32nf[] = {{Op::kSdpaQk, "orin_pk_t64x128k64g42s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t64x64k128g21s32nf[] = {{Op::kSdpaQk, "orin_pk_t64x64k128g21s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t64x64k128g22s32nf[] = {{Op::kSdpaQk, "orin_pk_t64x64k128g22s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t64x64k128g42s32nf[] = {{Op::kSdpaQk, "orin_pk_t64x64k128g42s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t64x64k128g44s32nf[] = {{Op::kSdpaQk, "orin_pk_t64x64k128g44s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t64x64k64g21s32nf[] = {{Op::kSdpaQk, "orin_pk_t64x64k64g21s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t64x64k64g22s32nf[] = {{Op::kSdpaQk, "orin_pk_t64x64k64g22s32nf", nullptr}};
const Preference kOrin_qk_orin_pk_t64x64k64g42s32nf[] = {{Op::kSdpaQk, "orin_pk_t64x64k64g42s32nf", nullptr}};
const Preference kOrinRefine1[] = {
    {Op::kSdpaQk, "4070ti_pk_t128x64k64g42s32nf", nullptr},
    {Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr},
    {Op::kSdpaAv, "4070ti_t64x64k32g42s32", nullptr},
};
const Preference kOrinLinRefine2[] = {
    {Op::kDq8caLinear, "orin_bf_t128x128k64g24s32mk32ra", nullptr},
};
const Preference kOrinRefine3[] = {
    {Op::kSdpaQk, "4070ti_pk_t128x64k64g42s32nf", nullptr},
    {Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr},
    {Op::kSdpaAv, "4070ti_t64x64k32g42s32", nullptr},
    {Op::kDq8caLinear, "orin_bf_t128x128k64g24s32mk32ra", nullptr},
};
const Preference kOrinLinRefine3[] = {
    {Op::kDq8caLinear, "orin_bf_t128x128k64g24s32mk32ra", nullptr},
    {Op::kQ4gswLinear, "bx_t128x128k32g42s32f32c", k_above_8192_orin},
};
const Preference kOrinRefine4[] = {
    {Op::kSdpaQk, "4070ti_pk_t128x64k64g42s32nf", nullptr},
    {Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr},
    {Op::kSdpaAv, "4070ti_t64x64k32g42s32", nullptr},
    {Op::kDq8caLinear, "orin_bf_t128x128k64g24s32mk32ra", nullptr},
    {Op::kQ4gswLinear, "bx_t128x128k32g42s32f32c", k_above_8192_orin},
};
const Preference kOrinLinRefine5[] = {
    {Op::kDq8caLinear, "orin_bf_t128x128k64g24s32mk32ra", nullptr},
    {Op::kQ4gswLinear, "bx_t128x128k32g42s32f32c", k_above_8192_orin},
    {Op::kQ4gswLinear, "orin_t256x128k16g42s32bt", orin_256_shape},
};
const Preference kOrinRefine5[] = {
    {Op::kSdpaQk, "4070ti_pk_t128x64k64g42s32nf", nullptr},
    {Op::kSdpaAv, "4070ti_ml_t64x128k32g42s32", nullptr},
    {Op::kSdpaAv, "4070ti_t64x64k32g42s32", nullptr},
    {Op::kDq8caLinear, "orin_bf_t128x128k64g24s32mk32ra", nullptr},
    {Op::kQ4gswLinear, "bx_t128x128k32g42s32f32c", k_above_8192_orin},
    {Op::kQ4gswLinear, "orin_t256x128k16g42s32bt", orin_256_shape},
};
// <<< orin sdpa-preferences
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
// 4w linear: an output of at most 512 columns (1B wk / wv).
bool xe2_narrow_output(const ShapeInfo& s) {
  return s.N <= 512;
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
const Preference kXe2_refine5[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
const Preference kXe2_refine6[] = {
    {Op::kSdpaQk, "xe2c_t64x128k32g82s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k64g44s32m8", xe2_head_dim_128},
    {Op::kSdpaAv, "xe2_t64x64k64g44s32m8", nullptr},
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
const Preference kXe2_q4_g82[] = {
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
const Preference kXe2_dq_k64[] = {
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
};
// xe2 end
// b70-fused begin: fused attention on the Arc Pro B70 (openspec/changes/sarc-1.5-b70-fused-port)
// A b70-fused profile is xe2-refine5 for every call the fused kernel does not take, plus the B580's fused
// kernel variants below, one per head_dim (shader name after sarc_dev_b580_sdpa_fused_; "" = none).
// b70-fused1 is the pair the B580 campaign selected (b580-fused1 at cea76c634): 16 rows per workgroup;
// head_dim 64 in 4 subgroups with 64-column blocks, head_dim 128 in 8 subgroups with 128-column blocks.
// b70-fused-<variant> are the single-variant profiles of the one kernel screen (the B580's screen5-select).
struct FusedB70 {
  const char* profile;
  const char* variants;
};
const FusedB70 kFusedB70[] = {
    {"b70-fused1", "d64_t16x64s16m8g4roj,d128_t16x128s16m8g8oj"},
    {"b70-fused-d64_t32x32s32m8ro", "d64_t32x32s32m8ro"},
    {"b70-fused-d64_t16x64s16m8g4roj", "d64_t16x64s16m8g4roj"},
    {"b70-fused-d64_t16x64s16m8g4oj", "d64_t16x64s16m8g4oj"},
    {"b70-fused-d128_t16x64s32m8ro", "d128_t16x64s32m8ro"},
    {"b70-fused-d128_t16x128s16m8g8oj", "d128_t16x128s16m8g8oj"},
    {"b70-fused-d128_t16x64s16m8g4oj", "d128_t16x64s16m8g4oj"},
};
// The function pointers of impl/sarc_dev/b580/SdpaB580Fused.cpp, whichever static initializer runs first.
Override& fused_b70() {
  static Override fused;
  return fused;
}
// b70-fused end
// b580 begin: Arc B580 profiles (openspec/changes/sarc-1.5-b580-prefill-refine, tools/gen_b580.py)
// Single-kernel screening profiles and the b580-refineN candidates. They take effect on the Arc B580 only
// (SDPA base rows of impl/sarc_dev/B580Sdpa.cpp, ET_VK_SARC_UNVERIFIED=1).
// attn*V: ShapeInfo::N is head_dim.
bool b580_head_dim_128(const ShapeInfo& s) {
  return s.N >= 128;
}
// QK^T: ShapeInfo::K is head_dim.
bool b580_qk_head_dim_128(const ShapeInfo& s) {
  return s.K >= 128;
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
const Preference kB580_qk_xe2c_t64x128k32g82s16m8nf[] = {{Op::kSdpaQk, "xe2c_t64x128k32g82s16m8nf", nullptr}};
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
// candidate 2 = refine1 + the shipped 4w tile with the texture3d drain staged one band at a time (release body, CSH_BAND)
const Preference kB580_refine2[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k16g44s16m8flib", nullptr},
};
// refine2 with the split-staging twin of the same tile (B580 screen 3, round 1: 1.17 to 1.22x per shape at kernel level)
const Preference kB580_refine2x[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
    {Op::kQ4gswLinear, "xe2s_t128x128k16g44s16m8flib", nullptr},
};
// candidate 3 = refine1 with the column-major fragment-layout QK^T for head_dim 128 (B580 screen 1: 7 to 8 % faster than pk on 3B / 8B, under 3 % on 1B)
const Preference kB580_refine3[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", b580_qk_head_dim_128},
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bt_t128x128k64g84s16m8", nullptr},
};
// b580 end
// b580-fused begin: fused attention on the Arc B580 (openspec/changes/sarc-1.5-b580-fused-port)
// A b580-fused profile is b580-refine3 for every call the fused kernel does not take, plus the fused
// kernel variants below, one per head_dim (shader name after sarc_dev_b580_sdpa_fused_; "" = none).
// b580-fused-<variant> are single-variant screening profiles.
struct FusedB580 {
  const char* profile;
  const char* variants;
};
// b580-fused2 = b580-fused1 with the fp32 no-tail softmax of the RTX 4070 Ti SUPER campaign
// (glsl/sarc_dev/sarc_sdpa_attn_weights_softmax_4070ti, variant 4070ti_nzf, through Override::softmax_variant)
// for the calls the fused kernel does not take; b580-refine3-nzf is that softmax without the fused kernel.
// Selected by screen5-select (3 rounds): 16 rows per workgroup; head_dim 64 in 4 subgroups with 64-column
// blocks and the Q tiles in registers, head_dim 128 in 8 subgroups with 128-column blocks.
const char kFusedB580Pair[] = "d64_t16x64s16m8g4roj,d128_t16x128s16m8g8oj";
const FusedB580 kFusedB580[] = {
    {"b580-fused1", kFusedB580Pair},
    {"b580-fused2", kFusedB580Pair},
    {"b580-fused-d64_t16x32s16m8ro", "d64_t16x32s16m8ro"},
    {"b580-fused-d64_t16x32s16m8o", "d64_t16x32s16m8o"},
    {"b580-fused-d64_t8x32s16m8ro", "d64_t8x32s16m8ro"},
    {"b580-fused-d64_t16x64s16m8ro", "d64_t16x64s16m8ro"},
    {"b580-fused-d64_t8x64s16m8ro", "d64_t8x64s16m8ro"},
    {"b580-fused-d64_t16x32s16m8r", "d64_t16x32s16m8r"},
    {"b580-fused-d64_t32x32s32m8ro", "d64_t32x32s32m8ro"},
    {"b580-fused-d128_t16x64s16m8ro", "d128_t16x64s16m8ro"},
    {"b580-fused-d128_t16x64s16m8o", "d128_t16x64s16m8o"},
    {"b580-fused-d128_t8x64s16m8ro", "d128_t8x64s16m8ro"},
    {"b580-fused-d128_t8x64s16m8o", "d128_t8x64s16m8o"},
    {"b580-fused-d128_t16x32s16m8ro", "d128_t16x32s16m8ro"},
    {"b580-fused-d128_t8x32s16m8ro", "d128_t8x32s16m8ro"},
    {"b580-fused-d128_t16x64s16m8r", "d128_t16x64s16m8r"},
    {"b580-fused-d128_t8x64s16m8r", "d128_t8x64s16m8r"},
    {"b580-fused-d128_t16x64s32m8ro", "d128_t16x64s32m8ro"},
    {"b580-fused-d64_t8x32s16m8roj", "d64_t8x32s16m8roj"},
    {"b580-fused-d64_t16x32s16m8roj", "d64_t16x32s16m8roj"},
    {"b580-fused-d64_t8x32s16m8roja", "d64_t8x32s16m8roja"},
    {"b580-fused-d64_t16x32s16m8roja", "d64_t16x32s16m8roja"},
    {"b580-fused-d64_t16x64s16m8roja", "d64_t16x64s16m8roja"},
    {"b580-fused-d64_t16x32s16m8oa", "d64_t16x32s16m8oa"},
    {"b580-fused-d128_t8x64s16m8roj", "d128_t8x64s16m8roj"},
    {"b580-fused-d128_t8x64s16m8roja", "d128_t8x64s16m8roja"},
    {"b580-fused-d128_t16x64s16m8roja", "d128_t16x64s16m8roja"},
    {"b580-fused-d128_t8x64s16m8oa", "d128_t8x64s16m8oa"},
    {"b580-fused-d128_t16x64s16m8oa", "d128_t16x64s16m8oa"},
    {"b580-fused-d128_t8x32s16m8roja", "d128_t8x32s16m8roja"},
    {"b580-fused-d128_t16x32s16m8roja", "d128_t16x32s16m8roja"},
    {"b580-fused-d128_t8x64s16m8rja", "d128_t8x64s16m8rja"},
    {"b580-fused-d128_t8x64s16m8g2oj", "d128_t8x64s16m8g2oj"},
    {"b580-fused-d128_t8x64s16m8g2j", "d128_t8x64s16m8g2j"},
    {"b580-fused-d128_t8x64s16m8g4roj", "d128_t8x64s16m8g4roj"},
    {"b580-fused-d128_t8x64s16m8g4oj", "d128_t8x64s16m8g4oj"},
    {"b580-fused-d128_t8x32s16m8g2oj", "d128_t8x32s16m8g2oj"},
    {"b580-fused-d128_t16x64s16m8g4oj", "d128_t16x64s16m8g4oj"},
    {"b580-fused-d64_t8x32s16m8g2roj", "d64_t8x32s16m8g2roj"},
    {"b580-fused-d64_t16x32s16m8g2oj", "d64_t16x32s16m8g2oj"},
    {"b580-fused-d64_t16x32s16m8g2roj", "d64_t16x32s16m8g2roj"},
    {"b580-fused-d64_t16x64s16m8g4roj", "d64_t16x64s16m8g4roj"},
    {"b580-fused-d64_t8x32s16m8oj", "d64_t8x32s16m8oj"},
    {"b580-fused-d128_t16x128s16m8g8oj", "d128_t16x128s16m8g8oj"},
    {"b580-fused-d128_t8x128s16m8g8roj", "d128_t8x128s16m8g8roj"},
    {"b580-fused-d64_t16x64s16m8g4oj", "d64_t16x64s16m8g4oj"},
    {"b580-fused-d64_t16x128s16m8g4roj", "d64_t16x128s16m8g4roj"},
    {"b580-fused-d64_t8x64s16m8g4roj", "d64_t8x64s16m8g4roj"},
};
// The function pointers of impl/sarc_dev/b580/SdpaB580Fused.cpp, whichever static initializer runs first.
Override& fused_b580() {
  static Override fused;
  return fused;
}
// b580-fused end
// xe2-order begin: workgroup-order profiles (openspec/changes/sarc-1.5-xe2-prefill-refine, tools/gen_xe2.py)
// Workgroup-order profiles (tools/gen_xe2.py, ORDER_PROFILES): the base profile with the 8da4w kernel
// replaced by its xe2bto twin (part i: remap prologue alone; n<nb_max>: blocked order;
// p<nb_max>: n<nb_max> for the large weight tensors, i for every other shape).
// 8da4w linear: a weight tensor (K x N x 4 bit) of at least 12 x 10^6 bytes.
bool xe2_order_w12(const ShapeInfo& s) {
  return s.K * s.N >= 24000000;
}
const Preference kXe2Order_i[] = {
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8i", nullptr},
};
const Preference kXe2Order_n10[] = {
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n10", nullptr},
};
const Preference kXe2Order_n13[] = {
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n13", nullptr},
};
const Preference kXe2Order_n16[] = {
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n16", nullptr},
};
const Preference kXe2Order_n21[] = {
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n21", nullptr},
};
const Preference kXe2Order_n26[] = {
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n26", nullptr},
};
const Preference kXe2Order_p13[] = {
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n13", xe2_order_w12},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8i", nullptr},
};
const Preference kB580Fused1Order_i[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", b580_qk_head_dim_128},
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8i", nullptr},
};
const Preference kB580Fused1Order_n10[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", b580_qk_head_dim_128},
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n10", nullptr},
};
const Preference kB580Fused1Order_n13[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", b580_qk_head_dim_128},
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n13", nullptr},
};
const Preference kB580Fused1Order_n16[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", b580_qk_head_dim_128},
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n16", nullptr},
};
const Preference kB580Fused1Order_n21[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", b580_qk_head_dim_128},
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n21", nullptr},
};
const Preference kB580Fused1Order_n26[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", b580_qk_head_dim_128},
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n26", nullptr},
};
const Preference kB580Fused1Order_p13[] = {
    {Op::kSdpaQk, "xe2c_t128x64k32g44s16m8nf", b580_qk_head_dim_128},
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", b580_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n13", xe2_order_w12},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8i", nullptr},
};
const Preference kB70Fused1Order_i[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8i", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
const Preference kB70Fused1Order_n10[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n10", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
const Preference kB70Fused1Order_n13[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n13", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
const Preference kB70Fused1Order_n16[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n16", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
const Preference kB70Fused1Order_n21[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n21", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
const Preference kB70Fused1Order_n26[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n26", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
// 8da4w linear: a weight tensor (K x N 4-bit weights) of at least 16 MiB.
bool xe2_order_weights_16m(const ShapeInfo& s) {
  return s.K * s.N >= 2 * (int64_t(16) << 20);
}
const Preference kB70Fused1Order_i_n26_w16m[] = {
    {Op::kSdpaQk, "pk_t128x64k32g44s16m8nf", nullptr},
    {Op::kSdpaAv, "xe2_t128x64k32g44s16m8", xe2_head_dim_128},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8n26", xe2_order_weights_16m},
    {Op::kDq8caLinear, "xe2bto_t128x128k64g84s16m8i", nullptr},
    {Op::kQ4gswLinear, "sweep_t128x128k32g84s16m8flw", xe2_narrow_output},
    {Op::kQ4gswLinear, "sweep_t128x128k16g82s16m8flib", nullptr},
};
// xe2-order end
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
    // 780M final configuration: these preferences plus what profile c11 of the
    // 780m block below adds (active_profile_780m).
    {"780m-final", k780mRefine3, sizeof(k780mRefine3) / sizeof(Preference)},
    // >>> 4070ti sdpa-profiles
    {"4070ti-qk-t128x64k32g42s32nf", k4070ti_qk_t128x64k32g42s32nf, 1},
    {"4070ti-qk-t128x64k32g42s32", k4070ti_qk_t128x64k32g42s32, 1},
    {"4070ti-qk-t128x64k32g24s32nf", k4070ti_qk_t128x64k32g24s32nf, 1},
    {"4070ti-qk-t128x64k32g22s32nf", k4070ti_qk_t128x64k32g22s32nf, 1},
    {"4070ti-qk-pk_t128x64k32g42s32nf", k4070ti_qk_pk_t128x64k32g42s32nf, 1},
    {"4070ti-qk-pk_t128x64k64g42s32nf", k4070ti_qk_pk_t128x64k64g42s32nf, 1},
    {"4070ti-qk-pk_t128x64k32g24s32nf", k4070ti_qk_pk_t128x64k32g24s32nf, 1},
    {"4070ti-qk-pk_t64x64k32g22s32nf", k4070ti_qk_pk_t64x64k32g22s32nf, 1},
    {"4070ti-qk-pk_t64x64k32g42s32nf", k4070ti_qk_pk_t64x64k32g42s32nf, 1},
    {"4070ti-qk-pk_t64x128k32g42s32nf", k4070ti_qk_pk_t64x128k32g42s32nf, 1},
    {"4070ti-qk-pk_t128x64k32g44s32nf", k4070ti_qk_pk_t128x64k32g44s32nf, 1},
    {"4070ti-qk-pk_t64x64k32g44s32nf", k4070ti_qk_pk_t64x64k32g44s32nf, 1},
    {"4070ti-qk-pk_t64x64k32g21s32nf", k4070ti_qk_pk_t64x64k32g21s32nf, 1},
    {"4070ti-qk-pk_t32x64k32g42s32nf", k4070ti_qk_pk_t32x64k32g42s32nf, 1},
    {"4070ti-av-t64x64k32g42s32", k4070ti_av_t64x64k32g42s32, 1},
    {"4070ti-av-t64x64k32g24s32", k4070ti_av_t64x64k32g24s32, 1},
    {"4070ti-av-ml_t64x128k32g42s32", k4070ti_av_ml_t64x128k32g42s32, 1},
    {"4070ti-av-ml_t128x128k32g42s32", k4070ti_av_ml_t128x128k32g42s32, 1},
    {"4070ti-av-ml_t128x64k32g42s32", k4070ti_av_ml_t128x64k32g42s32, 1},
    {"4070ti-av-ml_t128x64k32g44s32", k4070ti_av_ml_t128x64k32g44s32, 1},
    {"4070ti-av-ml_t256x64k32g42s32", k4070ti_av_ml_t256x64k32g42s32, 1},
    {"4070ti-av-ml_t128x64k32g24s32", k4070ti_av_ml_t128x64k32g24s32, 1},
    {"4070ti-av-ml_t64x128k32g44s32", k4070ti_av_ml_t64x128k32g44s32, 1},
    {"4070ti-av-ml_t128x128k32g44s32", k4070ti_av_ml_t128x128k32g44s32, 1},
    {"4070ti-av-ml_t32x64k32g42s32", k4070ti_av_ml_t32x64k32g42s32, 1},
    {"4070ti-av-ml_t256x64k32g44s32", k4070ti_av_ml_t256x64k32g44s32, 1},
    {"4070ti-refine1", k4070tiRefine1, sizeof(k4070tiRefine1) / sizeof(Preference)},
    // <<< 4070ti sdpa-profiles
    // >>> 4070ti df-profiles
    {"4070ti-qk-df_t64x64k32g11s32nf", k4070ti_qk_df_t64x64k32g11s32nf, 1},
    {"4070ti-qk-df_t32x64k32g11s32nf", k4070ti_qk_df_t32x64k32g11s32nf, 1},
    {"4070ti-qk-df_t64x32k32g11s32nf", k4070ti_qk_df_t64x32k32g11s32nf, 1},
    {"4070ti-qk-df_t32x32k32g11s32nf", k4070ti_qk_df_t32x32k32g11s32nf, 1},
    {"4070ti-qk-df_t16x64k32g11s32nf", k4070ti_qk_df_t16x64k32g11s32nf, 1},
    {"4070ti-qk-df_t128x64k32g22s32nf", k4070ti_qk_df_t128x64k32g22s32nf, 1},
    {"4070ti-qk-df_t128x64k32g42s32nf", k4070ti_qk_df_t128x64k32g42s32nf, 1},
    {"4070ti-qk-df_t64x64k32g22s32nf", k4070ti_qk_df_t64x64k32g22s32nf, 1},
    {"4070ti-qk-df_t128x128k32g22s32nf", k4070ti_qk_df_t128x128k32g22s32nf, 1},
    {"4070ti-qk-dfg_t64x64k32g11s32nf", k4070ti_qk_dfg_t64x64k32g11s32nf, 1},
    {"4070ti-qk-dfg_t32x64k32g11s32nf", k4070ti_qk_dfg_t32x64k32g11s32nf, 1},
    {"4070ti-qk-dfh_t64x64k32g11s32nf", k4070ti_qk_dfh_t64x64k32g11s32nf, 1},
    {"4070ti-qk-dfh_t32x64k32g11s32nf", k4070ti_qk_dfh_t32x64k32g11s32nf, 1},
    {"4070ti-qk-dfg_t64x64k32g22s32nf", k4070ti_qk_dfg_t64x64k32g22s32nf, 1},
    {"4070ti-qk-dfh_t64x64k32g22s32nf", k4070ti_qk_dfh_t64x64k32g22s32nf, 1},
    {"4070ti-av-df_t16x64k32g11s32", k4070ti_av_df_t16x64k32g11s32, 1},
    {"4070ti-av-df_t32x64k32g11s32", k4070ti_av_df_t32x64k32g11s32, 1},
    {"4070ti-av-df_t64x64k32g11s32", k4070ti_av_df_t64x64k32g11s32, 1},
    {"4070ti-av-df_t64x64k32g12s32", k4070ti_av_df_t64x64k32g12s32, 1},
    {"4070ti-av-df_t32x64k32g21s32", k4070ti_av_df_t32x64k32g21s32, 1},
    {"4070ti-av-df_t16x128k32g11s32", k4070ti_av_df_t16x128k32g11s32, 1},
    {"4070ti-av-df_t32x128k32g11s32", k4070ti_av_df_t32x128k32g11s32, 1},
    {"4070ti-av-df_t32x128k32g21s32", k4070ti_av_df_t32x128k32g21s32, 1},
    {"4070ti-av-df_t64x128k32g21s32", k4070ti_av_df_t64x128k32g21s32, 1},
    {"4070ti-av-df_t64x128k32g22s32", k4070ti_av_df_t64x128k32g22s32, 1},
    {"4070ti-av-dfg_t32x64k32g11s32", k4070ti_av_dfg_t32x64k32g11s32, 1},
    {"4070ti-av-dfg_t32x128k32g11s32", k4070ti_av_dfg_t32x128k32g11s32, 1},
    {"4070ti-av-dfg_t32x128k32g21s32", k4070ti_av_dfg_t32x128k32g21s32, 1},
    {"4070ti-av-dfh_t32x64k32g11s32", k4070ti_av_dfh_t32x64k32g11s32, 1},
    {"4070ti-av-dfh_t32x128k32g11s32", k4070ti_av_dfh_t32x128k32g11s32, 1},
    {"4070ti-av-dfh_t32x128k32g21s32", k4070ti_av_dfh_t32x128k32g21s32, 1},
    {"4070ti-av-dfg_t64x64k32g11s32", k4070ti_av_dfg_t64x64k32g11s32, 1},
    {"4070ti-av-dfh_t64x64k32g11s32", k4070ti_av_dfh_t64x64k32g11s32, 1},
    // <<< 4070ti df-profiles
    // >>> 4070ti lin-profiles
    {"4070ti-refine2", k4070tiRefine2, sizeof(k4070tiRefine2) / sizeof(Preference)},
    {"4070ti-refine3", k4070tiRefine3, sizeof(k4070tiRefine3) / sizeof(Preference)},
    {"4070ti-refine4", k4070tiRefine4, sizeof(k4070tiRefine4) / sizeof(Preference)},
    {"4070ti-refine5", k4070tiRefine5, sizeof(k4070tiRefine5) / sizeof(Preference)},
    // <<< 4070ti lin-profiles
    // >>> 4070ti-fused profiles
    // RTX 4070 Ti SUPER fused attention port
    // (openspec/changes/sarc-1.5-4070ti-fused-port): the kernels of
    // 4070ti-refine1 for every call the fused node does not serve.
    {"4070ti-fused1", k4070tiRefine1, sizeof(k4070tiRefine1) / sizeof(Preference)},
    // <<< 4070ti-fused profiles
    // >>> orin sdpa-profiles
    {"orin-av-df_t16x128k32g11s32", kOrin_av_df_t16x128k32g11s32, 1},
    {"orin-av-df_t16x64k32g11s32", kOrin_av_df_t16x64k32g11s32, 1},
    {"orin-av-df_t32x128k32g11s32", kOrin_av_df_t32x128k32g11s32, 1},
    {"orin-av-df_t32x128k32g21s32", kOrin_av_df_t32x128k32g21s32, 1},
    {"orin-av-df_t32x64k32g11s32", kOrin_av_df_t32x64k32g11s32, 1},
    {"orin-av-df_t32x64k32g21s32", kOrin_av_df_t32x64k32g21s32, 1},
    {"orin-av-df_t64x128k32g21s32", kOrin_av_df_t64x128k32g21s32, 1},
    {"orin-av-df_t64x128k32g22s32", kOrin_av_df_t64x128k32g22s32, 1},
    {"orin-av-df_t64x64k32g11s32", kOrin_av_df_t64x64k32g11s32, 1},
    {"orin-av-df_t64x64k32g12s32", kOrin_av_df_t64x64k32g12s32, 1},
    {"orin-av-dfg_t32x128k32g11s32", kOrin_av_dfg_t32x128k32g11s32, 1},
    {"orin-av-dfg_t32x128k32g21s32", kOrin_av_dfg_t32x128k32g21s32, 1},
    {"orin-av-dfg_t32x64k32g11s32", kOrin_av_dfg_t32x64k32g11s32, 1},
    {"orin-av-dfg_t64x64k32g11s32", kOrin_av_dfg_t64x64k32g11s32, 1},
    {"orin-av-dfh_t32x128k32g11s32", kOrin_av_dfh_t32x128k32g11s32, 1},
    {"orin-av-dfh_t32x128k32g21s32", kOrin_av_dfh_t32x128k32g21s32, 1},
    {"orin-av-dfh_t32x64k32g11s32", kOrin_av_dfh_t32x64k32g11s32, 1},
    {"orin-av-dfh_t64x64k32g11s32", kOrin_av_dfh_t64x64k32g11s32, 1},
    {"orin-av-ml_t128x128k32g42s32", kOrin_av_ml_t128x128k32g42s32, 1},
    {"orin-av-ml_t128x128k32g44s32", kOrin_av_ml_t128x128k32g44s32, 1},
    {"orin-av-ml_t128x64k32g24s32", kOrin_av_ml_t128x64k32g24s32, 1},
    {"orin-av-ml_t128x64k32g42s32", kOrin_av_ml_t128x64k32g42s32, 1},
    {"orin-av-ml_t128x64k32g44s32", kOrin_av_ml_t128x64k32g44s32, 1},
    {"orin-av-ml_t256x64k32g42s32", kOrin_av_ml_t256x64k32g42s32, 1},
    {"orin-av-ml_t256x64k32g44s32", kOrin_av_ml_t256x64k32g44s32, 1},
    {"orin-av-ml_t32x64k32g42s32", kOrin_av_ml_t32x64k32g42s32, 1},
    {"orin-av-ml_t64x128k32g42s32", kOrin_av_ml_t64x128k32g42s32, 1},
    {"orin-av-ml_t64x128k32g44s32", kOrin_av_ml_t64x128k32g44s32, 1},
    {"orin-av-t64x64k32g24s32", kOrin_av_t64x64k32g24s32, 1},
    {"orin-av-t64x64k32g42s32", kOrin_av_t64x64k32g42s32, 1},
    {"orin-qk-df_t128x128k32g22s32nf", kOrin_qk_df_t128x128k32g22s32nf, 1},
    {"orin-qk-df_t128x64k32g22s32nf", kOrin_qk_df_t128x64k32g22s32nf, 1},
    {"orin-qk-df_t128x64k32g42s32nf", kOrin_qk_df_t128x64k32g42s32nf, 1},
    {"orin-qk-df_t16x64k32g11s32nf", kOrin_qk_df_t16x64k32g11s32nf, 1},
    {"orin-qk-df_t32x32k32g11s32nf", kOrin_qk_df_t32x32k32g11s32nf, 1},
    {"orin-qk-df_t32x64k32g11s32nf", kOrin_qk_df_t32x64k32g11s32nf, 1},
    {"orin-qk-df_t64x32k32g11s32nf", kOrin_qk_df_t64x32k32g11s32nf, 1},
    {"orin-qk-df_t64x64k32g11s32nf", kOrin_qk_df_t64x64k32g11s32nf, 1},
    {"orin-qk-df_t64x64k32g22s32nf", kOrin_qk_df_t64x64k32g22s32nf, 1},
    {"orin-qk-dfg_t32x64k32g11s32nf", kOrin_qk_dfg_t32x64k32g11s32nf, 1},
    {"orin-qk-dfg_t64x64k32g11s32nf", kOrin_qk_dfg_t64x64k32g11s32nf, 1},
    {"orin-qk-dfg_t64x64k32g22s32nf", kOrin_qk_dfg_t64x64k32g22s32nf, 1},
    {"orin-qk-dfh_t32x64k32g11s32nf", kOrin_qk_dfh_t32x64k32g11s32nf, 1},
    {"orin-qk-dfh_t64x64k32g11s32nf", kOrin_qk_dfh_t64x64k32g11s32nf, 1},
    {"orin-qk-dfh_t64x64k32g22s32nf", kOrin_qk_dfh_t64x64k32g22s32nf, 1},
    {"orin-qk-pk_t128x64k32g24s32nf", kOrin_qk_pk_t128x64k32g24s32nf, 1},
    {"orin-qk-pk_t128x64k32g42s32nf", kOrin_qk_pk_t128x64k32g42s32nf, 1},
    {"orin-qk-pk_t128x64k32g44s32nf", kOrin_qk_pk_t128x64k32g44s32nf, 1},
    {"orin-qk-pk_t128x64k64g42s32nf", kOrin_qk_pk_t128x64k64g42s32nf, 1},
    {"orin-qk-pk_t32x64k32g42s32nf", kOrin_qk_pk_t32x64k32g42s32nf, 1},
    {"orin-qk-pk_t64x128k32g42s32nf", kOrin_qk_pk_t64x128k32g42s32nf, 1},
    {"orin-qk-pk_t64x64k32g21s32nf", kOrin_qk_pk_t64x64k32g21s32nf, 1},
    {"orin-qk-pk_t64x64k32g22s32nf", kOrin_qk_pk_t64x64k32g22s32nf, 1},
    {"orin-qk-pk_t64x64k32g42s32nf", kOrin_qk_pk_t64x64k32g42s32nf, 1},
    {"orin-qk-pk_t64x64k32g44s32nf", kOrin_qk_pk_t64x64k32g44s32nf, 1},
    {"orin-qk-t128x64k32g22s32nf", kOrin_qk_t128x64k32g22s32nf, 1},
    {"orin-qk-t128x64k32g24s32nf", kOrin_qk_t128x64k32g24s32nf, 1},
    {"orin-qk-t128x64k32g42s32", kOrin_qk_t128x64k32g42s32, 1},
    {"orin-qk-t128x64k32g42s32nf", kOrin_qk_t128x64k32g42s32nf, 1},
    {"orin-qk-orin_pk_t128x64k64g22s32nf", kOrin_qk_orin_pk_t128x64k64g22s32nf, 1},
    {"orin-qk-orin_pk_t128x64k64g24s32nf", kOrin_qk_orin_pk_t128x64k64g24s32nf, 1},
    {"orin-qk-orin_pk_t128x64k64g44s32nf", kOrin_qk_orin_pk_t128x64k64g44s32nf, 1},
    {"orin-qk-orin_pk_t32x64k64g42s32nf", kOrin_qk_orin_pk_t32x64k64g42s32nf, 1},
    {"orin-qk-orin_pk_t64x128k64g42s32nf", kOrin_qk_orin_pk_t64x128k64g42s32nf, 1},
    {"orin-qk-orin_pk_t64x64k128g21s32nf", kOrin_qk_orin_pk_t64x64k128g21s32nf, 1},
    {"orin-qk-orin_pk_t64x64k128g22s32nf", kOrin_qk_orin_pk_t64x64k128g22s32nf, 1},
    {"orin-qk-orin_pk_t64x64k128g42s32nf", kOrin_qk_orin_pk_t64x64k128g42s32nf, 1},
    {"orin-qk-orin_pk_t64x64k128g44s32nf", kOrin_qk_orin_pk_t64x64k128g44s32nf, 1},
    {"orin-qk-orin_pk_t64x64k64g21s32nf", kOrin_qk_orin_pk_t64x64k64g21s32nf, 1},
    {"orin-qk-orin_pk_t64x64k64g22s32nf", kOrin_qk_orin_pk_t64x64k64g22s32nf, 1},
    {"orin-qk-orin_pk_t64x64k64g42s32nf", kOrin_qk_orin_pk_t64x64k64g42s32nf, 1},
    {"orin-refine1", kOrinRefine1, sizeof(kOrinRefine1) / sizeof(Preference)},
    {"orin-lin-refine2", kOrinLinRefine2, sizeof(kOrinLinRefine2) / sizeof(Preference)},
    {"orin-refine3", kOrinRefine3, sizeof(kOrinRefine3) / sizeof(Preference)},
    {"orin-lin-refine3", kOrinLinRefine3, sizeof(kOrinLinRefine3) / sizeof(Preference)},
    {"orin-refine4", kOrinRefine4, sizeof(kOrinRefine4) / sizeof(Preference)},
    {"orin-lin-refine5", kOrinLinRefine5, sizeof(kOrinLinRefine5) / sizeof(Preference)},
    {"orin-refine5", kOrinRefine5, sizeof(kOrinRefine5) / sizeof(Preference)},
    // <<< orin sdpa-profiles
    // >>> orin-fused profiles
    // orin-refine5 + the fused attention node (impl/sarc_dev/orin/SdpaOrinFused.cpp names its kernels).
    {"orin-fused1", kOrinRefine5, sizeof(kOrinRefine5) / sizeof(Preference)},
    {"orin-fused2", kOrinRefine5, sizeof(kOrinRefine5) / sizeof(Preference)},
    // <<< orin-fused profiles
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
    {"xe2-refine5", kXe2_refine5, sizeof(kXe2_refine5) / sizeof(Preference)},
    {"xe2-refine6", kXe2_refine6, sizeof(kXe2_refine6) / sizeof(Preference)},
    {"xe2-q4-g82", kXe2_q4_g82, sizeof(kXe2_q4_g82) / sizeof(Preference)},
    {"xe2-dq-k64", kXe2_dq_k64, sizeof(kXe2_dq_k64) / sizeof(Preference)},
    // xe2 end
    // b70-fused begin
    {"b70-fused1", kXe2_refine5, sizeof(kXe2_refine5) / sizeof(Preference)},
    {"b70-fused-d64_t32x32s32m8ro", kXe2_refine5, sizeof(kXe2_refine5) / sizeof(Preference)},
    {"b70-fused-d64_t16x64s16m8g4roj", kXe2_refine5, sizeof(kXe2_refine5) / sizeof(Preference)},
    {"b70-fused-d64_t16x64s16m8g4oj", kXe2_refine5, sizeof(kXe2_refine5) / sizeof(Preference)},
    {"b70-fused-d128_t16x64s32m8ro", kXe2_refine5, sizeof(kXe2_refine5) / sizeof(Preference)},
    {"b70-fused-d128_t16x128s16m8g8oj", kXe2_refine5, sizeof(kXe2_refine5) / sizeof(Preference)},
    {"b70-fused-d128_t16x64s16m8g4oj", kXe2_refine5, sizeof(kXe2_refine5) / sizeof(Preference)},
    // b70-fused end
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
    {"b580-qk-xe2c-t64x128k32g82s16m8nf", kB580_qk_xe2c_t64x128k32g82s16m8nf, 1},
    {"b580-sdpa0", nullptr, 0},
    {"b580-refine0", kB580_refine0, sizeof(kB580_refine0) / sizeof(Preference)},
    {"b580-refine1", kB580_refine1, sizeof(kB580_refine1) / sizeof(Preference)},
    {"b580-dq-k64", kB580_dq_k64, sizeof(kB580_dq_k64) / sizeof(Preference)},
    {"b580-refine2", kB580_refine2, sizeof(kB580_refine2) / sizeof(Preference)},
    {"b580-refine2x", kB580_refine2x, sizeof(kB580_refine2x) / sizeof(Preference)},
    {"b580-refine3", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    // b580 end
    // b580-fused begin
    {"b580-fused1", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused2", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-refine3-nzf", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x32s16m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x32s16m8o", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t8x32s16m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x64s16m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t8x64s16m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x32s16m8r", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t32x32s32m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x64s16m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x64s16m8o", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8o", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x32s16m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x32s16m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x64s16m8r", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8r", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x64s32m8ro", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t8x32s16m8roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x32s16m8roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t8x32s16m8roja", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x32s16m8roja", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x64s16m8roja", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x32s16m8oa", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8roja", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x64s16m8roja", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8oa", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x64s16m8oa", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x32s16m8roja", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x32s16m8roja", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8rja", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8g2oj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8g2j", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8g4roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x64s16m8g4oj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x32s16m8g2oj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x64s16m8g4oj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t8x32s16m8g2roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x32s16m8g2oj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x32s16m8g2roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x64s16m8g4roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t8x32s16m8oj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t16x128s16m8g8oj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d128_t8x128s16m8g8roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x64s16m8g4oj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t16x128s16m8g4roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    {"b580-fused-d64_t8x64s16m8g4roj", kB580_refine3, sizeof(kB580_refine3) / sizeof(Preference)},
    // b580-fused end
    // xe2-order begin: workgroup-order profiles (tools/gen_xe2.py)
    {"xe2-order-i", kXe2Order_i, sizeof(kXe2Order_i) / sizeof(Preference)},
    {"xe2-order-n10", kXe2Order_n10, sizeof(kXe2Order_n10) / sizeof(Preference)},
    {"xe2-order-n13", kXe2Order_n13, sizeof(kXe2Order_n13) / sizeof(Preference)},
    {"xe2-order-n16", kXe2Order_n16, sizeof(kXe2Order_n16) / sizeof(Preference)},
    {"xe2-order-n21", kXe2Order_n21, sizeof(kXe2Order_n21) / sizeof(Preference)},
    {"xe2-order-n26", kXe2Order_n26, sizeof(kXe2Order_n26) / sizeof(Preference)},
    {"xe2-order-p13", kXe2Order_p13, sizeof(kXe2Order_p13) / sizeof(Preference)},
    {"b580-fused1-order-i", kB580Fused1Order_i, sizeof(kB580Fused1Order_i) / sizeof(Preference)},
    {"b580-fused1-order-n10", kB580Fused1Order_n10, sizeof(kB580Fused1Order_n10) / sizeof(Preference)},
    {"b580-fused1-order-n13", kB580Fused1Order_n13, sizeof(kB580Fused1Order_n13) / sizeof(Preference)},
    {"b580-fused1-order-n16", kB580Fused1Order_n16, sizeof(kB580Fused1Order_n16) / sizeof(Preference)},
    {"b580-fused1-order-n21", kB580Fused1Order_n21, sizeof(kB580Fused1Order_n21) / sizeof(Preference)},
    {"b580-fused1-order-n26", kB580Fused1Order_n26, sizeof(kB580Fused1Order_n26) / sizeof(Preference)},
    {"b580-fused1-order-p13", kB580Fused1Order_p13, sizeof(kB580Fused1Order_p13) / sizeof(Preference)},
    {"b70-fused1-order-i", kB70Fused1Order_i, sizeof(kB70Fused1Order_i) / sizeof(Preference)},
    {"b70-fused1-order-n10", kB70Fused1Order_n10, sizeof(kB70Fused1Order_n10) / sizeof(Preference)},
    {"b70-fused1-order-n13", kB70Fused1Order_n13, sizeof(kB70Fused1Order_n13) / sizeof(Preference)},
    {"b70-fused1-order-n16", kB70Fused1Order_n16, sizeof(kB70Fused1Order_n16) / sizeof(Preference)},
    {"b70-fused1-order-n21", kB70Fused1Order_n21, sizeof(kB70Fused1Order_n21) / sizeof(Preference)},
    {"b70-fused1-order-n26", kB70Fused1Order_n26, sizeof(kB70Fused1Order_n26) / sizeof(Preference)},
    {"b70-fused1-order-i-n26-w16m", kB70Fused1Order_i_n26_w16m, sizeof(kB70Fused1Order_i_n26_w16m) / sizeof(Preference)},
    // xe2-order end
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
// b70-fused begin
const char* fused_variants_b70() {
  const Profile* p = requested_profile();
  for (const FusedB70& f : kFusedB70) {
    if (p != nullptr && std::strcmp(p->name, f.profile) == 0) {
      return f.variants;
    }
  }
  // xe2-order begin: the workgroup-order twins of b70-fused1 keep its fused attention variants (tools/gen_xe2.py)
  if (p != nullptr && std::strncmp(p->name, "b70-fused1-order-", 17) == 0) {
    for (const FusedB70& f : kFusedB70) {
      if (std::strcmp(f.profile, "b70-fused1") == 0) {
        return f.variants;
      }
    }
  }
  // xe2-order end
  return "";
}
// b70-fused end
// b580-fused begin
const char* fused_variants_b580() {
  const Profile* p = requested_profile();
  for (const FusedB580& f : kFusedB580) {
    if (p != nullptr && std::strcmp(p->name, f.profile) == 0) {
      return f.variants;
    }
  }
  // xe2-order begin: the workgroup-order twins of b580-fused1 keep its fused attention pair (tools/gen_xe2.py)
  if (p != nullptr && std::strncmp(p->name, "b580-fused1-order-", 18) == 0) {
    return kFusedB580Pair;
  }
  // xe2-order end
  return "";
}
const char* softmax_variant_b580() {
  const Profile* p = requested_profile();
  if (p != nullptr && (std::strcmp(p->name, "b580-fused2") == 0 ||
                       std::strcmp(p->name, "b580-refine3-nzf") == 0)) {
    return "4070ti_nzf";
  }
  return nullptr;
}
// b580-fused end
// >>> 4070ti softmax-variant
// RTX 4070 Ti SUPER: the refine profiles that carry the SDPA kernels run the
// fp32 softmax without the full zero tail
// (glsl/sarc_dev/sarc_sdpa_attn_weights_softmax_4070ti.yaml), as candidates 4
// to 6 were measured. 4070ti-refine1 is candidate 4.
const char* softmax_variant_4070ti() {
  const Profile* p = requested_profile();
  if (p == nullptr) {
    return nullptr;
  }
  for (const char* name :
       {"4070ti-refine1", "4070ti-refine4", "4070ti-refine5"}) {
    if (std::strcmp(p->name, name) == 0) {
      return "4070ti_nzf";
    }
  }
  return nullptr;
}
// <<< 4070ti softmax-variant
// >>> 4070ti-fused variants
// RTX 4070 Ti SUPER fused attention port: the fused kernels of the requested
// profile, one per head_dim (the shader name after sarc_dev_4070ti_sdpa_), or
// null. The node lives in 4070ti/Sdpa4070tiFused.cpp, which needs the graph
// headers this file (GPU-free, test_sarc_select) must not include; it hands its
// Override::sdpa_fused_* functions over from its own static initializer
// (register_sdpa_fused_4070ti, below). The order of the two files'
// initializers is not defined: the pair is kept here and applied by whichever
// runs last, and only while a fused profile is requested.
const char* fused_variants_4070ti() {
  const Profile* p = requested_profile();
  if (p != nullptr && std::strcmp(p->name, "4070ti-fused1") == 0) {
    return "fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko";
  }
  return nullptr;
}
Override& fused_4070ti() {
  static Override fused;
  return fused;
}
// <<< 4070ti-fused variants

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
    // b70-fused begin
    if (*fused_variants_b70() != 0) {
      o.sdpa_fused_add = fused_b70().sdpa_fused_add;
      o.sdpa_fused_serves = fused_b70().sdpa_fused_serves;
    }
    // b70-fused end
    // >>> 4070ti softmax-override
    o.softmax_variant = softmax_variant_4070ti();
    // <<< 4070ti softmax-override
    // >>> 4070ti-fused override
    if (fused_variants_4070ti() != nullptr) {
      o.softmax_variant = "4070ti_nzf";
      o.sdpa_fused_add = fused_4070ti().sdpa_fused_add;
      o.sdpa_fused_serves = fused_4070ti().sdpa_fused_serves;
    }
    // <<< 4070ti-fused override
    // b580-fused begin
    // (Merge: assigned only when a B580 profile names one; the 4070ti block above assigns unconditionally.)
    if (softmax_variant_b580() != nullptr) {
      o.softmax_variant = softmax_variant_b580();
    }
    if (*fused_variants_b580() != 0) {
      o.sdpa_fused_add = fused_b580().sdpa_fused_add;
      o.sdpa_fused_serves = fused_b580().sdpa_fused_serves;
    }
    // b580-fused end
    // >>> orin softmax-variant
    // ET_VK_SARC_SOFTMAX_VARIANT=<suffix>: a dev-zone variant of the SARC softmax, e.g. 4070ti_nzf
    // (fp32 reduction, no zero tail). Through Override::softmax_variant, the release hook of the owner
    // decision of 2026-10-05; unset, the release softmax is used.
    if (const char* v = std::getenv("ET_VK_SARC_SOFTMAX_VARIANT")) {
      if (*v != 0) {
        o.softmax_variant = v;
        std::cerr << "[sarc_dev] softmax variant: " << v << std::endl;
      }
    }
    // <<< orin softmax-variant
    // >>> orin-fused override
    // An orin-fused* profile is the whole stack: the calls its fused node does not serve use the softmax orin_g64
    // unless the variable above names another. The fused node (impl/sarc_dev/orin/SdpaOrinFused.cpp) sets its
    // entry points from its own static initializer, which may have run already: keep them.
    if (o.softmax_variant == nullptr && requested_profile() != nullptr &&
        std::strncmp(requested_profile()->name, "orin-fused", 10) == 0) {
      o.softmax_variant = "orin_g64";
      std::cerr << "[sarc_dev] softmax variant: " << o.softmax_variant << std::endl;
    }
    // (Merge: unless a profile above set them, as the 4070ti-fused block does.)
    if (o.sdpa_fused_add == nullptr) {
      o.sdpa_fused_add = get_override().sdpa_fused_add;
      o.sdpa_fused_serves = get_override().sdpa_fused_serves;
    }
    // <<< orin-fused override
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

// b580-fused, b70-fused begin: called by impl/sarc_dev/b580/SdpaB580Fused.cpp, the node both cards use.
// (Merge of the two branches' definitions: a profile is of one card, so at most one list is non-empty.)
const char* sdpa_fused_variants_b580() {
  const char* v = fused_variants_b580();
  return *v != 0 ? v : fused_variants_b70();
}
void register_sdpa_fused_b580(
    void (*add)(ComputeGraph&, const std::vector<int32_t>&),
    bool (*serves)(ComputeGraph*, const std::vector<int32_t>&)) {
  fused_b580().sdpa_fused_add = add;
  fused_b580().sdpa_fused_serves = serves;
  fused_b70().sdpa_fused_add = add;
  fused_b70().sdpa_fused_serves = serves;
  if (*sdpa_fused_variants_b580() != 0) {
    Override o = get_override();
    o.sdpa_fused_add = add;
    o.sdpa_fused_serves = serves;
    set_override(o);
  }
}
// b580-fused, b70-fused end
// >>> 4070ti-fused entry
const char* sdpa_fused_variants_4070ti() {
  const char* v = fused_variants_4070ti();
  return v != nullptr ? v : "";
}
void register_sdpa_fused_4070ti(
    void (*add)(ComputeGraph&, const std::vector<int32_t>&),
    bool (*serves)(ComputeGraph*, const std::vector<int32_t>&)) {
  fused_4070ti().sdpa_fused_add = add;
  fused_4070ti().sdpa_fused_serves = serves;
  if (fused_variants_4070ti() != nullptr) {
    Override o = get_override();
    o.sdpa_fused_add = add;
    o.sdpa_fused_serves = serves;
    set_override(o);
  }
}
// <<< 4070ti-fused entry
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
// fused3sb: fused3 with subgroupBarrier() after every memoryBarrierShared()
// (round 3, 2026-10-08). c8 to c10 stay on the fused3 names they were measured
// with.
const char kFused780mOnePassSb[] =
    "fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko";
const Profile780m k780mProfiles[] = {
    {"refine9", k780mRefine9, sizeof(k780mRefine9) / sizeof(Pick780m), nullptr, ""},
    {"refine10", k780mRefine10, sizeof(k780mRefine10) / sizeof(Pick780m), nullptr, ""},
    {"softmax-r1", nullptr, 0, "780m_r1", ""},
    {"c7", nullptr, 0, "780m_r3", ""},
    {"c8", nullptr, 0, "780m_r3", kFused780mTwoPass},
    {"c9", k780mRefine9, sizeof(k780mRefine9) / sizeof(Pick780m), "780m_r3", kFused780mOnePass},
    {"c10", k780mRefine10, sizeof(k780mRefine10) / sizeof(Pick780m), "780m_r3", kFused780mOnePass},
    {"c11", k780mRefine11, sizeof(k780mRefine11) / sizeof(Pick780m), "780m_r3", kFused780mOnePassSb},
};
const Profile780m* active_profile_780m() {
  static const Profile780m* const active = []() -> const Profile780m* {
    const char* e = std::getenv("ET_VK_SARC_780M_PROFILE");
    // ET_VK_SARC_DEV_PROFILE=780m-final alone is the final configuration:
    // 780m-refine3 (kProfiles above) with c11.
    const char* dev = std::getenv("ET_VK_SARC_DEV_PROFILE");
    if (e == nullptr && dev != nullptr && std::string(dev) == "780m-final") {
      e = "c11";
    }
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

// The 780M family's node (780m/Sdpa780mFused.cpp, also used by the RX 7600 and
// RX 7900 XTX) is installed only when it has variants to run: a 780m profile with
// a fused entry or ET_VK_SARC_780M_SDPA_FUSED. Otherwise the override keeps the
// node another device's profile installed. (Merge of the campaigns: installed
// unconditionally, it replaced the RTX 4070 Ti SUPER's node whenever its static
// initializer ran later, so 4070ti-fused1 ran the split SDPA path.)
bool fused_780m_wanted() {
  return std::getenv("ET_VK_SARC_780M_SDPA_FUSED") != nullptr ||
      *sdpa_fused_variants_780m() != 0;
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
    if (fused_780m_wanted()) {
      o.sdpa_fused_add = fused_780m().sdpa_fused_add;
      o.sdpa_fused_serves = fused_780m().sdpa_fused_serves;
    }
    set_override(o);
  }
} registrar_780m;

} // namespace

void register_sdpa_fused_780m(
    void (*add)(ComputeGraph&, const std::vector<int32_t>&),
    bool (*serves)(ComputeGraph*, const std::vector<int32_t>&)) {
  fused_780m().sdpa_fused_add = add;
  fused_780m().sdpa_fused_serves = serves;
  if (fused_780m_wanted()) {
    Override o = get_override();
    o.sdpa_fused_add = add;
    o.sdpa_fused_serves = serves;
    set_override(o);
  }
}

const char* sdpa_fused_variants_780m() {
  const Profile780m* active = active_profile_780m();
  return active != nullptr ? active->fused : "";
}
} // namespace sarc
} // namespace vkcompute
// --- 780m end ---

// --- rx7600 begin (openspec/changes/sarc-1.5-rx7600-prefill-refine, 2026-10-06) ---
// ET_VK_SARC_RX7600_PROFILE=<name>: the RX 7600 campaign's candidates, on top of
// the selection above: a linear kernel per layer shape among the registered
// candidates, and the softmax variant. A shape whose entry does not apply or
// whose kernel does not fit keeps the earlier selection. The fused attention
// kernels are chosen with ET_VK_SARC_780M_SDPA_FUSED (780m/Sdpa780mFused.cpp).
namespace vkcompute {
namespace sarc {
namespace {

struct PickRx7600 {
  Op op;
  const char* kernel_base;
  bool (*shape_ok)(const ShapeInfo&);
};
struct ProfileRx7600 {
  const char* name;
  const PickRx7600* picks;
  size_t count;
  const char* softmax_variant;
};
// rx7600-refine2 (candidate 3): a linear kernel per layer shape, from the complete
// 3-round kernel screens of 2026-10-07 (results/rx7600/screens/): a kernel
// replaces the table kernel of a shape only if at least 3 % faster in every
// round. The shapes are the twelve real prefill shapes (M = 2048); N and K
// identify a layer of one model. 4w: the 256 x 128 tile with column-major B
// (cbt) on the w2 shapes and 8B w1_w3, the same tile with row-major B staging
// (bbt) on 1B wk_wv and 3B w1_w3; the other 4w shapes keep the table kernel.
// 8da4w: the 780M's 256 x 64 tile (K step 64) on all twelve shapes. The
// texel-wise family (zpg_bt) is candidate 4's, screened against these picks.
bool q4_cbt_shape(const ShapeInfo& s) {
  return !s.gemv &&
      ((s.N == 14336 && s.K == 4096) || (s.N == 4096 && s.K == 14336) ||
       (s.N == 2048 && s.K == 8192) || (s.N == 3072 && s.K == 8192));
}
bool q4_bbt_shape(const ShapeInfo& s) {
  return !s.gemv &&
      ((s.N == 512 && s.K == 2048) || (s.N == 8192 && s.K == 3072));
}
bool dq_prefill_shape(const ShapeInfo& s) {
  static const int64_t nk[][2] = {
      {4096, 4096}, {1024, 4096}, {14336, 4096}, {4096, 14336},
      {2048, 2048}, {512, 2048},  {8192, 2048},  {2048, 8192},
      {3072, 3072}, {1024, 3072}, {8192, 3072},  {3072, 8192}};
  if (s.gemv) {
    return false;
  }
  for (const auto& e : nk) {
    if (s.N == e[0] && s.K == e[1]) {
      return true;
    }
  }
  return false;
}
const PickRx7600 kRx7600Refine2[] = {
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbt",
     q4_cbt_shape},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32bbt",
     q4_bbt_shape},
    {Op::kDq8caLinear,
     "sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1",
     dq_prefill_shape},
};
// rx7600-refine3 (candidate 4): refine2 with the whole-texel weight staging
// family (zpg_bt) on every 8da4w shape instead of the 256 x 64 tile; the 4w
// picks are refine2's. The screen has it within 1.5 % of refine2's kernel on
// every shape (not the 3 % that replaces it), so this is measured as a whole.
const PickRx7600 kRx7600Refine3[] = {
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbt",
     q4_cbt_shape},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32bbt",
     q4_bbt_shape},
    {Op::kDq8caLinear,
     "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k64g42s32",
     dq_prefill_shape},
};
// rx7600-refine4 (round 2, candidate 1): refine2 with the 8da4w kernel of glsl/sarc_dev/sarc_dev_rx7600_x_linear_dq8ca_coopmat_zpg.yaml
// whose A staging rows are 24 bytes apart in shared memory (pa6) instead of 16, on every 8da4w prefill shape of refine2; the 4w picks are
// refine2's. Same arithmetic, same tile, K step and thread maps as the refine2 kernel (afmb1); the screen (results/rx7600/round2/) has it
// at least 3 % faster on all twelve shapes in every round.
const PickRx7600 kRx7600Refine4[] = {
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbt",
     q4_cbt_shape},
    {Op::kQ4gswLinear,
     "sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32bbt",
     q4_bbt_shape},
    {Op::kDq8caLinear,
     "sarc_dev_rx7600_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa6pb4csha",
     dq_prefill_shape},
};
// rx7600-refine5 (round 2, candidate 2): refine4 plus the 4w kernel of glsl/sarc_dev/sarc_dev_rx7600_x_linear_q4gsw_coopmat.yaml with the
// 256 x 128 tile (16 waves), the A staging typed uvec2 with a row pitch of 72 bytes and the column-major B staging with 88 bytes (ap4bp12), on
// every 4w prefill shape: 12 of 12 shapes at least 3 % faster than the round-1 pick in both rounds of the kernel screen
// (results/rx7600/round2/). Same arithmetic, tile and K step as the refine2 / refine4 picks of the shapes it replaces.
const PickRx7600 kRx7600Refine5[] = {
    {Op::kQ4gswLinear,
     "sarc_dev_rx7600_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap4bp12",
     dq_prefill_shape},
    {Op::kDq8caLinear,
     "sarc_dev_rx7600_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa6pb4csha",
     dq_prefill_shape},
};
const ProfileRx7600 kRx7600Profiles[] = {
    {"rx7600-refine1", nullptr, 0, "780m_r3"},
    {"rx7600-refine2",
     kRx7600Refine2,
     sizeof(kRx7600Refine2) / sizeof(PickRx7600),
     "780m_r3"},
    {"rx7600-refine3",
     kRx7600Refine3,
     sizeof(kRx7600Refine3) / sizeof(PickRx7600),
     "780m_r3"},
    {"rx7600-refine4",
     kRx7600Refine4,
     sizeof(kRx7600Refine4) / sizeof(PickRx7600),
     "780m_r3"},
    {"rx7600-refine5",
     kRx7600Refine5,
     sizeof(kRx7600Refine5) / sizeof(PickRx7600),
     "780m_r3"},
};
const ProfileRx7600* active_profile_rx7600() {
  static const ProfileRx7600* const active = []() -> const ProfileRx7600* {
    const char* e = std::getenv("ET_VK_SARC_RX7600_PROFILE");
    if (e == nullptr || *e == 0) {
      return nullptr;
    }
    for (const ProfileRx7600& p : kRx7600Profiles) {
      if (std::strcmp(p.name, e) == 0) {
        return &p;
      }
    }
    std::cerr << "[sarc_dev] unknown ET_VK_SARC_RX7600_PROFILE=" << e << std::endl;
    std::abort();
  }();
  return active;
}

std::optional<Choice> (*select_before_rx7600)(
    const DeviceInfo&,
    const ShapeInfo&,
    const std::optional<Choice>&) = nullptr;

std::optional<Choice> select_rx7600(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const std::optional<Choice>& table_choice) {
  const std::optional<Choice> before =
      select_before_rx7600(device, shape, table_choice);
  const ProfileRx7600* active = active_profile_rx7600();
  if (active == nullptr || !before.has_value()) {
    return before;
  }
  for (size_t i = 0; i < active->count; ++i) {
    const PickRx7600& pick = active->picks[i];
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

// Round 2 (2026-10-08): the 8da4w variants of glsl/sarc_dev/sarc_dev_rx7600_x_linear_dq8ca_coopmat_zpg.yaml (generated table). Reachable by exact
// name (ET_VK_SARC_780M_DQ) and by the rx7600-refine4 profile of round 2; no earlier profile picks them.
const Row kRx7600Space[] = {
#include "SpaceRx7600Dq.inc"
};
// The 4w variants of glsl/sarc_dev/sarc_dev_rx7600_x_linear_q4gsw_coopmat.yaml (generated table), reachable by exact name (ET_VK_SARC_780M_Q4).
const Row kRx7600SpaceQ4[] = {
#include "SpaceRx7600Q4.inc"
};

struct RegistrarRx7600 {
  RegistrarRx7600() {
    register_candidates(kRx7600Space, sizeof(kRx7600Space) / sizeof(kRx7600Space[0]));
    register_candidates(kRx7600SpaceQ4, sizeof(kRx7600SpaceQ4) / sizeof(kRx7600SpaceQ4[0]));
    Override o = get_override();
    select_before_rx7600 = o.select;
    o.select = select_rx7600;
    const ProfileRx7600* active = active_profile_rx7600();
    if (active != nullptr && active->softmax_variant != nullptr &&
        !env_true("ET_VK_DISABLE_COOPMAT")) {
      o.softmax_variant = active->softmax_variant;
    }
    set_override(o);
    if (active != nullptr) {
      std::cerr << "[sarc_dev] rx7600 profile active: " << active->name << std::endl;
    }
  }
} registrar_rx7600;

} // namespace
} // namespace sarc
} // namespace vkcompute
// --- rx7600 end ---

// --- m51 begin (openspec/changes/sarc-1.5-m51-prefill-refine, 2026-10-06) ---
// ET_VK_SARC_M51_PROFILE=<name>: the M51 candidates, on top of
// ET_VK_SARC_UNVERIFIED=1 (the xclipse rows). A profile names the fused
// attention kernels per head_dim (m51/SdpaM51Fused.cpp) and linear kernels per
// shape; shapes and calls they do not cover keep what the selection above
// returns. Not meant together with
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
    // Texel-wise B staging (glsl/sarc_dev/sarc_dev_linear_q4gsw_coopmat_bx.yaml, m51 block).
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_linear_q4gsw_coopmat_bx_t128x128k16g22s32f32xp", xp_tile(128, 128, 16, 2, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g22s32f32xp", xp_tile(128, 128, 32, 2, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32xp", xp_tile(128, 128, 32, 4, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
};

// 8da4w, second round (glsl/sarc_dev/sarc_dev_m51_linear_dq8ca_coopmat_zpgd.yaml): zpg_bt's texel-wise B
// staging with the A operand loaded by coopMatLoad straight from the row-major activations (rowmajor_a, so the
// graph quantizes row-major and an unaligned prompt keeps the table's zpgtr coopmat kernel). Selected with
// ET_VK_SARC_DQ8CA_VARIANT=zpgd_<tile> or by profile c6.
const Row kM51Dq[] = {
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_m51_linear_dq8ca_coopmat_zpgd_t128x64k32g22s32",
     {128, 64, 32, 2, 2, 32, 16, false}, kTex3dTex2d | kBufTex2d, nullptr,
     Status::kUnverified, true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_m51_linear_dq8ca_coopmat_zpgd_t128x64k64g22s32",
     {128, 64, 64, 2, 2, 32, 16, false}, kTex3dTex2d | kBufTex2d, nullptr,
     Status::kUnverified, true},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_m51_linear_dq8ca_coopmat_zpgd_t128x128k32g42s32",
     {128, 128, 32, 4, 2, 32, 16, false}, kTex3dTex2d | kBufTex2d, nullptr,
     Status::kUnverified, true},
};

// 8da4w, second round: smaller subgroup tiles of zpg_bt (glsl/sarc_dev/sarc_dev_linear_dq8ca_coopmat_zpg_bt.yaml, m51
// block), for the screen. Selected with ET_VK_SARC_DQ8CA_VARIANT=bt_<tile>.
const Row kM51DqBt[] = {
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t64x64k32g22s32",
     {64, 64, 32, 2, 2, 32, 16, false}, kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t64x128k32g22s32",
     {64, 128, 32, 2, 2, 32, 16, false}, kTex3dTex2d, nullptr, Status::kUnverified},
};

// 8da4w, second round (glsl/sarc_dev/sarc_dev_m51_linear_dq8ca_coopmat_zpgf.yaml): zpg_bt with every fragment of a
// chunk loaded before its MMAs. Selected with ET_VK_SARC_DQ8CA_VARIANT=zpgf_<tile> or by profile c7.
const Row kM51DqF[] = {
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_m51_linear_dq8ca_coopmat_zpgf_t128x64k32g22s32",
     {128, 64, 32, 2, 2, 32, 16, false}, kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_m51_linear_dq8ca_coopmat_zpgf_t128x64k64g22s32",
     {128, 64, 64, 2, 2, 32, 16, false}, kTex3dTex2d, nullptr, Status::kUnverified},
};

// 4w, second round (glsl/sarc_dev/sarc_dev_m51_linear_q4gsw_coopmat_bz.yaml): the texel-wise bx staging with the B operand
// staged N-major inside the shared pool (B_COLMAJOR + CSH_POOL), so the matrix unit's B loads are wide. Selected with
// ET_VK_SARC_Q4GSW_VARIANT=bz_<tile> or by profile c8.
const Row kM51Bz[] = {
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_m51_linear_q4gsw_coopmat_bz_t128x128k16g22s32f32xp", xp_tile(128, 128, 16, 2, 2, 32),
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_m51_linear_q4gsw_coopmat_bz_t128x128k32g22s32f32xp", xp_tile(128, 128, 32, 2, 2, 32),
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_m51_linear_q4gsw_coopmat_bz_t128x128k32g42s32f32xp", xp_tile(128, 128, 32, 4, 2, 32),
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_m51_linear_q4gsw_coopmat_bz_t128x128k16g42s32f32xp", xp_tile(128, 128, 16, 4, 2, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_m51_linear_q4gsw_coopmat_bz_t128x128k32g24s32f32xp", xp_tile(128, 128, 32, 2, 4, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_m51_linear_q4gsw_coopmat_bz_t128x128k16g24s32f32xp", xp_tile(128, 128, 16, 2, 4, 32),
     kTex3dTex2d, nullptr, Status::kUnverified},
};

// 4w, second round (glsl/sarc_dev/sarc_dev_m51_linear_q4gsw_coopmat_bw.yaml): bz with packed fp16 dequantisation of the
// weights (bit-identical result). Selected with ET_VK_SARC_Q4GSW_VARIANT=bw_<tile> or by profile c9.
const Row kM51Bw[] = {
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_m51_linear_q4gsw_coopmat_bw_t128x128k32g42s32f32xp", xp_tile(128, 128, 32, 4, 2, 32),
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
};

// A pick sends the shapes its predicate accepts to one candidate kernel (exact
// name) when the kernel fits; every other shape keeps the selection above.
struct PickM51 {
  Op op;
  const char* kernel_base;
  bool (*shape_ok)(const ShapeInfo&);
};
struct ProfileM51 {
  const char* name;
  const char* fused;
  const PickM51* picks;
  size_t count;
};
// Screen of 2026-10-06 (texture3d, 1B/3B/8B shapes, 3 rounds; selected = beats
// the xclipse row by the RULES R8 margin in every round): the 780M's texel-wise
// 8da4w staging (zpg_bt) on every 8da4w shape; on the 4w shapes with K = 4096
// (8B w1_w3, wk_wv, wq_wo) t128x128k32g42s32f32xp and, slightly faster, the
// same tile with texel-wise B staging (bx); no other tile on any shape.
bool any_shape(const ShapeInfo&) {
  return true;
}
bool k_is_4096(const ShapeInfo& s) {
  return s.K == 4096;
}
const PickM51 kM51PicksC2[] = {
    {Op::kDq8caLinear, "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32", any_shape},
};
const PickM51 kM51PicksC3[] = {
    {Op::kDq8caLinear, "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32", any_shape},
    {Op::kQ4gswLinear, "sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32xp", k_is_4096},
};
// c6: c5 with the 8da4w pick replaced by the direct-A kernel (second round). Default off.
const PickM51 kM51PicksC6[] = {
    {Op::kDq8caLinear, "sarc_dev_m51_linear_dq8ca_coopmat_zpgd_t128x64k32g22s32", any_shape},
    {Op::kQ4gswLinear, "sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32xp", k_is_4096},
};
// c7: c5 with the fragments-first 8da4w kernel (second round). Default off.
const PickM51 kM51PicksC7[] = {
    {Op::kDq8caLinear, "sarc_dev_m51_linear_dq8ca_coopmat_zpgf_t128x64k32g22s32", any_shape},
    {Op::kQ4gswLinear, "sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32xp", k_is_4096},
};
// c8: c5 with the 4w kernel bz_t128x128k32g42s32f32xp on every 4w shape (second round; the K = 4096 pick of c3 is replaced by it).
const PickM51 kM51PicksC8[] = {
    {Op::kDq8caLinear, "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32", any_shape},
    {Op::kQ4gswLinear, "sarc_dev_m51_linear_q4gsw_coopmat_bz_t128x128k32g42s32f32xp", any_shape},
};
// c9: c8 with the packed-dequantisation 4w kernel (second round).
const PickM51 kM51PicksC9[] = {
    {Op::kDq8caLinear, "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32", any_shape},
    {Op::kQ4gswLinear, "sarc_dev_m51_linear_q4gsw_coopmat_bw_t128x128k32g42s32f32xp", any_shape},
};
const char kFusedM51[] = "fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk";
// Fused-variant screen (ETDump of the fused kernel, 3 rounds, RULES R8 margin): for head_dim 64 the
// 16 x 64 tile on a 64-wide subgroup beat the 780M's choice in every round; for head_dim 128 no variant did.
const char kFusedM51B[] = "fused3_d64_t16x64g11s64rk,fused3_d128_t16x64g11s32rk";
const ProfileM51 kM51Profiles[] = {
    {"c1", kFusedM51, nullptr, 0},
    {"c2", kFusedM51, kM51PicksC2, sizeof(kM51PicksC2) / sizeof(PickM51)},
    {"c3", kFusedM51, kM51PicksC3, sizeof(kM51PicksC3) / sizeof(PickM51)},
    // c4: c2 plus the head_dim 64 fused variant above (c3, the 4w pick for K = 4096, acts on the 8B model only).
    {"c4", kFusedM51B, kM51PicksC2, sizeof(kM51PicksC2) / sizeof(PickM51)},
    // c5: c4 plus the 4w pick of c3 (K = 4096, 8B model only), one name for the stack that the 8B round gates.
    {"c5", kFusedM51B, kM51PicksC3, sizeof(kM51PicksC3) / sizeof(PickM51)},
    // c6: c5 with the direct-A 8da4w kernel (second round of the M51 campaign).
    {"c6", kFusedM51B, kM51PicksC6, sizeof(kM51PicksC6) / sizeof(PickM51)},
    // c7: c5 with the fragments-first 8da4w kernel (second round).
    {"c7", kFusedM51B, kM51PicksC7, sizeof(kM51PicksC7) / sizeof(PickM51)},
    // c8: c5 with the N-major-B 4w kernel on every 4w shape (second round of the M51 campaign).
    {"c8", kFusedM51B, kM51PicksC8, sizeof(kM51PicksC8) / sizeof(PickM51)},
    // c9: c8 with the packed-dequantisation 4w kernel (second round).
    {"c9", kFusedM51B, kM51PicksC9, sizeof(kM51PicksC9) / sizeof(PickM51)},
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
  const std::optional<Choice> before =
      select_before_m51(device, shape, table_choice);
  const ProfileM51* active = active_profile_m51();
  if (active == nullptr) {
    return before;
  }
  std::optional<Choice> served = before;
  if (!served.has_value() && shape.op == Op::kDq8caLinear &&
      shape.int8_layout != Int8Layout::kAny) {
    // The activation layout was fixed by the kernel chosen for the build-time
    // shape, which may be a pick of the other layout than the table row's:
    // ask again with the layout open. The fit check below keeps the layout.
    ShapeInfo any_layout = shape;
    any_layout.int8_layout = Int8Layout::kAny;
    served = select_before_m51(device, any_layout, select_table(device, any_layout));
  }
  if (!served.has_value()) {
    return before;
  }
  for (size_t i = 0; i < active->count; ++i) {
    const PickM51& pick = active->picks[i];
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

struct RegistrarM51 {
  RegistrarM51() {
    register_candidates(kM51Q4, sizeof(kM51Q4) / sizeof(kM51Q4[0]));
    register_candidates(kM51Dq, sizeof(kM51Dq) / sizeof(kM51Dq[0]));
    register_candidates(kM51DqBt, sizeof(kM51DqBt) / sizeof(kM51DqBt[0]));
    register_candidates(kM51DqF, sizeof(kM51DqF) / sizeof(kM51DqF[0]));
    register_candidates(kM51Bz, sizeof(kM51Bz) / sizeof(kM51Bz[0]));
    register_candidates(kM51Bw, sizeof(kM51Bw) / sizeof(kM51Bw[0]));
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

// --- 7900xtx begin (openspec/changes/sarc-1.5-7900xtx-prefill-refine, 2026-10-08) ---
// ET_VK_SARC_7900XTX_PROFILE=<name>: the RX 7900 XTX campaign's candidates, on top of the selection above: a linear
// kernel per layer shape among the registered candidates, and the softmax variant. A shape whose entry does not apply
// or whose kernel does not fit keeps the earlier selection. The shapes are the twelve real prefill shapes (M = 2048);
// N and K identify a layer of one model. The picks are the complete 3-round kernel screens of 2026-10-08
// (results/7900xtx/screens/): a kernel replaces the table kernel of a shape only if at least 3 % faster in every round.
//   7900xtx-refine1: the softmax variant only (the 780m_r3 of ET_VK_SARC_780M_PROFILE=c7)
//   7900xtx-refine2: refine1 and the screen's picks, texel-wise (zpg_bt) family excluded (candidate 3); 4w only on 1B
//   7900xtx-refine3: refine2 with the screen's picks over all kernels on the 8da4w shapes, i.e. the texel-wise
//                    weight staging family where it is the best (candidate 4)
//   7900xtx-refine5: refine4 and, for head dimension 64 only (1B), the attn*V sweep tile t32x32k32g22s32 (second attention screen of
//                    2026-10-09, results/7900xtx/sdpa-screen2/: 1.16x in every round on that op; every other tile of the screen was slower)
//   7900xtx-refine6: refine5 and, for 8B w2 (N = 4096, K = 14336), the sweep 128 x 64 tile t128x64k32g22s32 instead of the release row: the one pick of the
//                    first second-set screen (1.034 in the worst round; the same pair read 1.026 and 1.024 in two other screens, so it is gated as candidate 7)
//   ET_VK_SARC_7900XTX_QK / ET_VK_SARC_7900XTX_AV=<kernel_base>: a registered attention candidate by exact name (screening), whatever the profile
//   7900xtx-refine4: refine2 and the unfused attention kernels of the screen of 2026-10-09 (results/7900xtx/sdpa-screen/):
//                    QK^T with packed K staging and no mask fill (pk_t128x128k32g42s32nf, valid with the truncated SARC
//                    softmax only) and the sweep attn*V tile t64x64k32g42s32, on every prefill call that fits (candidate 5)
namespace vkcompute {
namespace sarc {
namespace {

struct NK7900xtx {
  int64_t N, K;
};
struct Pick7900xtx {
  Op op;
  const char* kernel_base;
  const NK7900xtx* shapes;
  size_t count;
};
struct Profile7900xtx {
  const char* name;
  const Pick7900xtx* picks;
  size_t count;
  const char* softmax_variant;
};
#define SARC_7900XTX_N(a) (sizeof(a) / sizeof(a[0]))

// 4w (fp32 accumulate, column-major B staging, 128 x 128 tile): the 1B shapes wq_wo, wk_wv and w2.
const NK7900xtx kQ4Sweep128[] = {{2048, 2048}, {512, 2048}, {2048, 8192}};
// 8da4w, zpg sweep 256 x 64 tile (K step 32): 1B w1_w3 / wq_wo, 3B w1_w3 / w2 / wq_wo, 8B w1_w3.
const NK7900xtx kDqSweep256x64[] = {
    {8192, 2048}, {2048, 2048}, {8192, 3072}, {3072, 8192}, {3072, 3072}, {14336, 4096}};
// 8da4w, zpg sweep 128 x 64 tile (K step 32): 8B wk_wv and wq_wo.
const NK7900xtx kDqSweep128x64[] = {{1024, 4096}, {4096, 4096}};
// 8da4w, the 780M's 256 x 64 tile with K step 64: 1B w2.
const NK7900xtx kDqAfmb1[] = {{2048, 8192}};
// 8da4w, zpg sweep 64 x 64 tile: 1B wk_wv.
const NK7900xtx kDqSweep64x64[] = {{512, 2048}};
// refine3: the texel-wise family (zpg_bt) where it is the best kernel of the screen.
const NK7900xtx kDqBtK32[] = {{14336, 4096}, {4096, 14336}, {8192, 2048}, {8192, 3072}};
const NK7900xtx kDqBtK64[] = {{1024, 4096}, {4096, 4096}, {2048, 8192}, {2048, 2048}, {3072, 8192}, {3072, 3072}};
const NK7900xtx kDqBtAfmb2[] = {{1024, 3072}};
// refine5: attn*V of head dimension 64 (N = head_dim, any context length: K < 0 matches every K).
const NK7900xtx kAvHeadDim64[] = {{64, -1}};
// refine6: the two 8B shapes that refine2 already gives this tile, and 8B w2.
const NK7900xtx kDqSweep128x64W2[] = {{1024, 4096}, {4096, 4096}, {4096, 14336}};

const Pick7900xtx kRefine2[] = {
    {Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt",
     kQ4Sweep128,
     SARC_7900XTX_N(kQ4Sweep128)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g24s32",
     kDqSweep256x64,
     SARC_7900XTX_N(kDqSweep256x64)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g22s32",
     kDqSweep128x64,
     SARC_7900XTX_N(kDqSweep128x64)},
    {Op::kDq8caLinear,
     "sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1",
     kDqAfmb1,
     SARC_7900XTX_N(kDqAfmb1)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g22s32",
     kDqSweep64x64,
     SARC_7900XTX_N(kDqSweep64x64)},
};
const Pick7900xtx kRefine3[] = {
    {Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt",
     kQ4Sweep128,
     SARC_7900XTX_N(kQ4Sweep128)},
    {Op::kDq8caLinear,
     "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32",
     kDqBtK32,
     SARC_7900XTX_N(kDqBtK32)},
    {Op::kDq8caLinear,
     "sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k64g22s32",
     kDqBtK64,
     SARC_7900XTX_N(kDqBtK64)},
    {Op::kDq8caLinear,
     "sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32afmb2",
     kDqBtAfmb2,
     SARC_7900XTX_N(kDqBtAfmb2)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g22s32",
     kDqSweep64x64,
     SARC_7900XTX_N(kDqSweep64x64)},
};
// refine4: refine2 and the two attention kernels; a pick without a shape list applies to every prefill call of its op.
const Pick7900xtx kRefine4[] = {
    {Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt",
     kQ4Sweep128,
     SARC_7900XTX_N(kQ4Sweep128)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g24s32",
     kDqSweep256x64,
     SARC_7900XTX_N(kDqSweep256x64)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g22s32",
     kDqSweep128x64,
     SARC_7900XTX_N(kDqSweep128x64)},
    {Op::kDq8caLinear,
     "sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1",
     kDqAfmb1,
     SARC_7900XTX_N(kDqAfmb1)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g22s32",
     kDqSweep64x64,
     SARC_7900XTX_N(kDqSweep64x64)},
    {Op::kSdpaQk, "sarc_sdpa_qk_coopmat_pk_t128x128k32g42s32nf", nullptr, 0},
    {Op::kSdpaAv, "sarc_sdpa_av_coopmat_sweep_t64x64k32g42s32", nullptr, 0},
};
// Attention kernel variants of the 7900 XTX screen (glsl/sarc_dev/sarc_sdpa_{qk_coopmat_pk,av_coopmat_sweep}.yaml, 7900xtx blocks); selected
// only by name: ET_VK_SARC_7900XTX_QK / ET_VK_SARC_7900XTX_AV = <kernel_base> (screening), or by a profile pick.
const Row k7900xtxSdpaSpace[] = {
    {"", nullptr, Op::kSdpaQk, "sarc_sdpa_qk_coopmat_pk_t128x128k32g24s32nf", {128, 128, 32, 2, 4, 32, 16, false}, kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kSdpaQk, "sarc_sdpa_qk_coopmat_pk_t128x128k32g22s64nf", {128, 128, 32, 2, 2, 64, 16, false}, kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kSdpaQk, "sarc_sdpa_qk_coopmat_pk_t128x128k32g44s32nf", {128, 128, 32, 4, 4, 32, 16, false}, kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kSdpaQk, "sarc_sdpa_qk_coopmat_pk_t128x128k32g22s32nf", {128, 128, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kSdpaQk, "sarc_sdpa_qk_coopmat_pk_t64x128k32g42s32nf", {64, 128, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kSdpaAv, "sarc_sdpa_av_coopmat_sweep_t128x128k32g44s32", {128, 128, 32, 4, 4, 32, 16, false}, kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kSdpaAv, "sarc_sdpa_av_coopmat_sweep_t128x128k32g82s32", {128, 128, 32, 8, 2, 32, 16, false}, kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kSdpaAv, "sarc_sdpa_av_coopmat_sweep_t128x128k32g42s64", {128, 128, 32, 4, 2, 64, 16, false}, kBufBuf, nullptr, Status::kUnverified},
    {"", nullptr, Op::kSdpaAv, "sarc_sdpa_av_coopmat_sweep_t32x32k32g22s32", {32, 32, 32, 2, 2, 32, 16, false}, kBufBuf, nullptr, Status::kUnverified},
};
// Second set (items 2 and 3, glsl/sarc_dev/sarc_dev_7900xtx_x_linear_{dq8ca_coopmat_zpg,q4gsw_coopmat}.yaml, generated by tools/gen_7900xtx_{dq,q4}.py):
// the release linear bodies with staging row pitches, selected by name (screens) or by a profile pick. The last row is the phase-timing twin of the 4w table kernel
// (MEASUREMENT ONLY, glsl/sarc_dev/sarc_dev_prof_q4gsw.yaml, 7900xtx block).
const Row k7900xtxLinearSpace[] = {
#include "Space7900xtxDq.inc"
#include "Space7900xtxQ4.inc"
    {"", nullptr, Op::kQ4gswLinear, "sarc_dev_prof_q4gsw_t256x128k32g24s32f32cbtp", {256, 128, 32, 2, 4, 32, 16, true, false, false}, kTex3dTex2d, nullptr, Status::kUnverified},
};
const char* exact_name_7900xtx(Op op) {
  static const char* const qk = std::getenv("ET_VK_SARC_7900XTX_QK");
  static const char* const av = std::getenv("ET_VK_SARC_7900XTX_AV");
  return op == Op::kSdpaQk ? qk : (op == Op::kSdpaAv ? av : nullptr);
}
const Pick7900xtx kRefine5[] = {
    {Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt",
     kQ4Sweep128,
     SARC_7900XTX_N(kQ4Sweep128)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g24s32",
     kDqSweep256x64,
     SARC_7900XTX_N(kDqSweep256x64)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g22s32",
     kDqSweep128x64,
     SARC_7900XTX_N(kDqSweep128x64)},
    {Op::kDq8caLinear,
     "sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1",
     kDqAfmb1,
     SARC_7900XTX_N(kDqAfmb1)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g22s32",
     kDqSweep64x64,
     SARC_7900XTX_N(kDqSweep64x64)},
    {Op::kSdpaQk, "sarc_sdpa_qk_coopmat_pk_t128x128k32g42s32nf", nullptr, 0},
    {Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_sweep_t32x32k32g22s32",
     kAvHeadDim64,
     SARC_7900XTX_N(kAvHeadDim64)},
    {Op::kSdpaAv, "sarc_sdpa_av_coopmat_sweep_t64x64k32g42s32", nullptr, 0},
};
const Pick7900xtx kRefine6[] = {
    {Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt",
     kQ4Sweep128,
     SARC_7900XTX_N(kQ4Sweep128)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g24s32",
     kDqSweep256x64,
     SARC_7900XTX_N(kDqSweep256x64)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g22s32",
     kDqSweep128x64W2,
     SARC_7900XTX_N(kDqSweep128x64W2)},
    {Op::kDq8caLinear,
     "sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1",
     kDqAfmb1,
     SARC_7900XTX_N(kDqAfmb1)},
    {Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g22s32",
     kDqSweep64x64,
     SARC_7900XTX_N(kDqSweep64x64)},
    {Op::kSdpaQk, "sarc_sdpa_qk_coopmat_pk_t128x128k32g42s32nf", nullptr, 0},
    {Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_sweep_t32x32k32g22s32",
     kAvHeadDim64,
     SARC_7900XTX_N(kAvHeadDim64)},
    {Op::kSdpaAv, "sarc_sdpa_av_coopmat_sweep_t64x64k32g42s32", nullptr, 0},
};
const Profile7900xtx k7900xtxProfiles[] = {
    {"7900xtx-refine1", nullptr, 0, "780m_r3"},
    {"7900xtx-refine2", kRefine2, SARC_7900XTX_N(kRefine2), "780m_r3"},
    {"7900xtx-refine3", kRefine3, SARC_7900XTX_N(kRefine3), "780m_r3"},
    {"7900xtx-refine4", kRefine4, SARC_7900XTX_N(kRefine4), "780m_r3"},
    {"7900xtx-refine5", kRefine5, SARC_7900XTX_N(kRefine5), "780m_r3"},
    {"7900xtx-refine6", kRefine6, SARC_7900XTX_N(kRefine6), "780m_r3"},
};
const Profile7900xtx* active_profile_7900xtx() {
  static const Profile7900xtx* const active = []() -> const Profile7900xtx* {
    const char* e = std::getenv("ET_VK_SARC_7900XTX_PROFILE");
    if (e == nullptr || *e == 0) {
      return nullptr;
    }
    for (const Profile7900xtx& p : k7900xtxProfiles) {
      if (std::strcmp(p.name, e) == 0) {
        return &p;
      }
    }
    std::cerr << "[sarc_dev] unknown ET_VK_SARC_7900XTX_PROFILE=" << e << std::endl;
    std::abort();
  }();
  return active;
}

std::optional<Choice> (*select_before_7900xtx)(
    const DeviceInfo&,
    const ShapeInfo&,
    const std::optional<Choice>&) = nullptr;

std::optional<Choice> select_7900xtx(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const std::optional<Choice>& table_choice) {
  const std::optional<Choice> before =
      select_before_7900xtx(device, shape, table_choice);
  const Profile7900xtx* active = active_profile_7900xtx();
  if (!before.has_value() || shape.gemv) {
    return before;
  }
  if (const char* want = exact_name_7900xtx(shape.op)) {
    if (*want != 0) {
      for (const Row& row : candidates()) {
        if (row.op == shape.op && row.kernel_base == std::string(want) &&
            q4gsw_coopmat_fits(device, shape, row)) {
          return Choice{row.kernel_base, row.dims, row.rowmajor_a};
        }
      }
      return before;
    }
  }
  if (active == nullptr) {
    return before;
  }
  for (size_t i = 0; i < active->count; ++i) {
    const Pick7900xtx& pick = active->picks[i];
    if (pick.op != shape.op) {
      continue;
    }
    bool listed = pick.shapes == nullptr;
    for (size_t j = 0; j < pick.count; ++j) {
      listed = listed ||
          (shape.N == pick.shapes[j].N && (pick.shapes[j].K < 0 || shape.K == pick.shapes[j].K));
    }
    if (!listed) {
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

struct Registrar7900xtx {
  Registrar7900xtx() {
    Override o = get_override();
    register_candidates(k7900xtxSdpaSpace, sizeof(k7900xtxSdpaSpace) / sizeof(k7900xtxSdpaSpace[0]));
    register_candidates(k7900xtxLinearSpace, sizeof(k7900xtxLinearSpace) / sizeof(k7900xtxLinearSpace[0]));
    select_before_7900xtx = o.select;
    o.select = select_7900xtx;
    const Profile7900xtx* active = active_profile_7900xtx();
    if (active != nullptr && active->softmax_variant != nullptr &&
        !env_true("ET_VK_DISABLE_COOPMAT")) {
      o.softmax_variant = active->softmax_variant;
    }
    set_override(o);
    if (active != nullptr) {
      std::cerr << "[sarc_dev] 7900xtx profile active: " << active->name << std::endl;
    }
  }
} registrar_7900xtx;

} // namespace
} // namespace sarc
} // namespace vkcompute
// --- 7900xtx end ---

// --- rdna3-780m WMMA matrix begin ---
namespace vkcompute {
namespace sarc {
namespace {

struct Wmma780mProfile {
  const char* name;
  const char* kernel_base;
  const char* alias;
  Op op;
  Storage io;
  Storage weight;
  unsigned sources;
  uint32_t shared_bytes;
};
const Wmma780mProfile kWmma780mProfiles[] = {
#include "Wmma780mProfiles.inc"
};

std::optional<Choice> registered_wmma780m_choice(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const char* name) {
  for (const auto* store : {&candidates(), &rows()}) {
    for (const Row& row : *store) {
      if (row.op == shape.op && std::strcmp(row.kernel_base, name) == 0 &&
          q4gsw_coopmat_fits(device, shape, row)) {
        return Choice{row.kernel_base, row.dims, row.rowmajor_a};
      }
    }
  }
  return std::nullopt;
}

// Replay source predicates with the queried target limits, not historical device limits.
std::optional<Choice> source_wmma780m_choice(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    unsigned source) {
  DeviceInfo origin = device;
  origin.name = source == 2 ? "radeon rx 7600" : "radeon rx 7900 xtx";
  const auto table = select_table(origin, shape);
  if (!table.has_value()) {
    return table;
  }
  if (source == 2) {
    for (const PickRx7600& pick : kRx7600Refine5) {
      if (pick.op == shape.op && pick.shape_ok(shape)) {
        auto choice = registered_wmma780m_choice(origin, shape, pick.kernel_base);
        if (choice.has_value()) {
          return choice;
        }
      }
    }
  } else {
    for (const Pick7900xtx& pick : kRefine5) {
      if (pick.op != shape.op) {
        continue;
      }
      bool listed = pick.shapes == nullptr;
      for (size_t i = 0; i < pick.count; ++i) {
        listed = listed ||
            (shape.N == pick.shapes[i].N &&
             (pick.shapes[i].K < 0 || shape.K == pick.shapes[i].K));
      }
      if (listed) {
        auto choice = registered_wmma780m_choice(origin, shape, pick.kernel_base);
        if (choice.has_value()) {
          return choice;
        }
      }
    }
  }
  return table;
}

const Wmma780mProfile* active_wmma780m_profile() {
  static const Wmma780mProfile* const active = []() -> const Wmma780mProfile* {
    const char* name = std::getenv("ET_VK_SARC_780M_WMMA_PROFILE");
    if (name == nullptr || *name == 0 || !env_true("ET_VK_SARC_UNVERIFIED")) {
      return nullptr;
    }
    const char* control = std::getenv("ET_VK_SARC_DEV_PROFILE");
    if (control != nullptr && std::strcmp(control, "780m-final") == 0) {
      for (const Wmma780mProfile& profile : kWmma780mProfiles) {
        if (std::strcmp(profile.name, name) == 0) {
          return &profile;
        }
      }
    }
    std::cerr << "[sarc_dev] invalid 780M WMMA profile/control: " << name << std::endl;
    std::abort();
  }();
  return active;
}

std::optional<Choice> (*select_before_wmma780m)(
    const DeviceInfo&, const ShapeInfo&, const std::optional<Choice>&) = nullptr;

std::optional<Choice> select_wmma780m(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const std::optional<Choice>& table_choice) {
  const auto before = select_before_wmma780m(device, shape, table_choice);
  const Wmma780mProfile* profile = active_wmma780m_profile();
  if (profile == nullptr || !before.has_value() || !device.is_amd ||
      device.name.find("780m") == std::string::npos ||
      profile->op != shape.op || shape.M <= 1 || shape.N <= 0 || shape.K <= 0 ||
      shape.group_size <= 0 || shape.ignore_alignment ||
      shape.input != profile->io || shape.output != profile->io ||
      shape.weight != profile->weight || profile->shared_bytes > device.max_shared_bytes) {
    return before;
  }
  bool eligible = (profile->sources & 1) &&
      (before->kernel_base == profile->kernel_base ||
       (profile->alias != nullptr && before->kernel_base == profile->alias));
  for (unsigned source : {2u, 4u}) {
    if (!(profile->sources & source)) {
      continue;
    }
    const auto choice = source_wmma780m_choice(device, shape, source);
    eligible = eligible || (choice.has_value() &&
        (choice->kernel_base == profile->kernel_base ||
         (profile->alias != nullptr && choice->kernel_base == profile->alias)));
  }
  if (eligible) {
    const auto choice = registered_wmma780m_choice(device, shape, profile->kernel_base);
    if (choice.has_value()) {
      return choice;
    }
  }
  return before;
}

struct RegistrarWmma780m {
  RegistrarWmma780m() {
    Override o = get_override();
    select_before_wmma780m = o.select;
    o.select = select_wmma780m;
    set_override(o);
  }
} registrar_wmma780m;

} // namespace
} // namespace sarc
} // namespace vkcompute
// --- rdna3-780m WMMA matrix end ---
