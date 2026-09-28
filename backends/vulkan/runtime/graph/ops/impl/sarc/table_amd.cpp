/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC selection rows for AMD GPUs. First matching row wins, so list the most
// specific device first. Status kVerified requires sarc/tools/verify.sh
// evidence on that device (see sarc/README.md).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {
namespace {

// The shipped RDNA coopmat variants were built and validated for wave64
// adapters that support cooperative matrix; they force subgroup 32 per
// pipeline (SUBGROUP_SIZE in the yaml).
bool amd_wave64(const DeviceInfo& d) {
  return d.is_amd && d.subgroup_size == 64;
}

// t128x128k32g42s32 with fp32 accumulate and the drain staged in Ash.
constexpr TileDims k780mDims = {128, 128, 32, 4, 2, 32, 16, true};

const Row kAmdRows[] = {
    // Radeon 780M (gfx1103, RADV, Mesa 25.2.7). Verified on release 1.5
    // (openspec/changes/sarc-1.5-bootstrap/results/780m): 2048-token prefill
    // 1B/3B/8B 1916/683/352 tok/s vs stock 1.5 1205/421/194; kernel times
    // within -1.6..-0.2 % of the 1.4 study; next token == tiled; pdiff pass.
    {"780m",
     amd_wave64,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c",
     k780mDims,
     kTex3dTex2d | kBufTex2d | kBufBuf,
     /*shape_ok=*/nullptr,
     Status::kVerified},
    // Radeon 780M 8da4w: zpg (4h4w activations), full A map. Verified on
    // release 1.5 (sarc-1.5-8da4w-port): kernels 0.992x of the 1.4 study;
    // with the SDPA rows, prefill 1B/3B/8B 2544/1048/487 tok/s (1.4:
    // 2538/1055/486).
    {"780m",
     amd_wave64,
     Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_t128x64k32g42s32",
     {128, 64, 32, 4, 2, 32, 16, false},
     kTex3dTex2d | kBufTex2d,
     nullptr,
     Status::kVerified},

    // Samsung Xclipse (M51): 4w on the 1.4 tile with fp32 accumulate and, for
    // texture3d, the one-pass full-tile drain from a shared LDS pool (the 1.4
    // fp16-accumulate, banded-drain variant returned wrong values here).
    // 8da4w: the 1.4 default.
    // Owned by the M51 agent, not verified on release 1.5.
    {"xclipse",
     amd_wave64,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g22s32f32xp",
     {128, 128, 16, 2, 2, 32, 16, false, /*csh_full=*/true, /*csh_pool=*/true},
     kTex3dTex2d | kBufTex2d | kBufBuf,
     nullptr,
     Status::kUnverified},
    {"xclipse",
     amd_wave64,
     Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_t128x64k32g42s32",
     {128, 64, 32, 4, 2, 32, 16, false},
     kTex3dTex2d | kBufTex2d,
     nullptr,
     Status::kUnverified,
     /*rowmajor_a=*/true},

    // Radeon 780M: SDPA prefill coopmat (QK^T, attn*V; the softmax truncation comes
    // with them). Verified on release 1.5 (sarc-1.5-sdpa-port): SDPA correctness
    // 4/4 (0 mismatches); prefill back to the 1.4 level, 1B 4w 2695 vs 2702.
    {"780m",
     amd_wave64,
     Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_t128x64k32g22s64",
     {128, 64, 32, 2, 2, 64, 16, false},
     kBufBuf,
     nullptr,
     Status::kVerified},
    {"780m",
     amd_wave64,
     Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_t64x64k32g22s64",
     {64, 64, 32, 2, 2, 64, 16, false},
     kBufBuf,
     nullptr,
     Status::kVerified},
    // Samsung Xclipse (M51): SDPA prefill coopmat (QK^T, attn*V; the softmax truncation comes
    // with them). 1.4 SARC branches; on the 780M it recovers ~307 ms of the
    // 1B 2048-token prefill (.artifacts/sarc-1.5/sdpa-gap/REPORT.md).
    {"xclipse",
     amd_wave64,
     Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_t128x64k32g22s64",
     {128, 64, 32, 2, 2, 64, 16, false},
     kBufBuf,
     nullptr,
     Status::kUnverified},
    {"xclipse",
     amd_wave64,
     Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_t64x64k32g22s64",
     {64, 64, 32, 2, 2, 64, 16, false},
     kBufBuf,
     nullptr,
     Status::kUnverified},
    // RX 7900 XTX (gfx1100, AMDVLK 2025.Q2.1), not yet verified on release 1.5
    // (openspec/changes/sarc-1.5-7900xtx-4w). 4w: fp32 accumulate (the fp16-accumulate tile
    // fails the 8B production diff here), column-major B staging, texture3d IO only.
    {"7900 xtx",
     amd_wave64,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t256x128k32g24s32f32cbt",
     {256, 128, 32, 2, 4, 32, 16, true},
     kTex3dTex2d,
     nullptr,
     Status::kUnverified},
    // The 780M SDPA prefill kernels and 8da4w zpg kernel. QK^T's shared memory (Ash + Bsh +
    // Csh, 30.5 KB) fits this card's 32 KB limit.
    {"7900 xtx",
     amd_wave64,
     Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_t128x64k32g22s64",
     {128, 64, 32, 2, 2, 64, 16, false},
     kBufBuf,
     nullptr,
     Status::kUnverified},
    {"7900 xtx",
     amd_wave64,
     Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_t64x64k32g22s64",
     {64, 64, 32, 2, 2, 64, 16, false},
     kBufBuf,
     nullptr,
     Status::kUnverified},
    {"7900 xtx",
     amd_wave64,
     Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_t128x64k32g42s32",
     {128, 64, 32, 4, 2, 32, 16, false},
     kTex3dTex2d | kBufTex2d,
     nullptr,
     Status::kUnverified},
    // RX 7600 (gfx1102, Navi33, RADV Mesa 26.2.3 -- the first RADV with
    // VK_KHR_cooperative_matrix here; Mesa 23.2 has none), not yet verified on
    // release 1.5. 4w: the 7900 XTX tile won the 2026-09-28 sweep on this card
    // (fp32 accumulate: RADV's fp16-accumulate WMMA is the slower roof here).
    {"rx 7600",
     amd_wave64,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t256x128k32g24s32f32cbt",
     {256, 128, 32, 2, 4, 32, 16, true},
     kTex3dTex2d,
     nullptr,
     Status::kUnverified},
    // The 780M SDPA prefill kernels and 8da4w zpg kernel.
    {"rx 7600",
     amd_wave64,
     Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_t128x64k32g22s64",
     {128, 64, 32, 2, 2, 64, 16, false},
     kBufBuf,
     nullptr,
     Status::kUnverified},
    {"rx 7600",
     amd_wave64,
     Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_t64x64k32g22s64",
     {64, 64, 32, 2, 2, 64, 16, false},
     kBufBuf,
     nullptr,
     Status::kUnverified},
    {"rx 7600",
     amd_wave64,
     Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_t128x64k32g42s32",
     {128, 64, 32, 4, 2, 32, 16, false},
     kTex3dTex2d | kBufTex2d,
     nullptr,
     Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_rows(kAmdRows, sizeof(kAmdRows) / sizeof(kAmdRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
