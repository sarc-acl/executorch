/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <cstring>

namespace vkcompute {
namespace sarc {
namespace {

bool active_device(const DeviceInfo& device);

// Preserve the parent table's projection service boundary.
bool projection(const ShapeInfo& s) {
  const bool nk = (s.K == 2048 && (s.N == 512 || s.N == 2048 || s.N == 8192)) ||
      (s.K == 3072 && (s.N == 1024 || s.N == 3072 || s.N == 8192)) ||
      (s.K == 4096 && (s.N == 1024 || s.N == 4096 || s.N == 14336)) ||
      (s.K == 8192 && (s.N == 2048 || s.N == 3072)) ||
      (s.K == 14336 && s.N == 4096);
  return nk && s.group_size == 128 && s.M > 1 && s.M <= 2048;
}

bool wide_projection(const ShapeInfo& s) {
  return projection(s) && s.N > 512;
}

bool large_k_projection(const ShapeInfo& s) {
  return projection(s) && s.K > 8192;
}

bool short_k_projection(const ShapeInfo& s) {
  return projection(s) && s.K <= 8192;
}

bool short_k_256_projection(const ShapeInfo& s) {
  return short_k_projection(s) && (s.M != 256 || s.N % 2048 == 0);
}

struct Candidate {
  const char* profile;
  Row row;
  uint32_t shared_bytes;
};

const Candidate kCandidates[] = {
    {"orin-fused-wmma-01",
     {"tegra orin", active_device, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t256x128k16g42s32ga",
      {256, 128, 16, 4, 2, 32, 16, false}, kTex3dTex2d, wide_projection, Status::kUnverified,
      false},
     41472},
    {"orin-fused-wmma-02",
     {"tegra orin", active_device, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x128k16g24s32ga",
      {128, 128, 16, 2, 4, 32, 16, false}, kTex3dTex2d, projection, Status::kUnverified,
      false},
     37376},
    {"orin-fused-wmma-03",
     {"tegra orin", active_device, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x256k16g42s32ga",
      {128, 256, 16, 4, 2, 32, 16, false}, kBufTex2d, wide_projection, Status::kUnverified,
      false},
     29184},
    {"orin-fused-wmma-04",
     {"tegra orin", active_device, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x128k16g42s32ga",
      {128, 128, 16, 4, 2, 32, 16, false}, kBufTex2d, projection, Status::kUnverified,
      false},
     20992},
    {"orin-fused-wmma-05",
     {"tegra orin", active_device, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x128k16g42s32ga",
      {128, 128, 16, 4, 2, 32, 16, false}, kBufBuf, projection, Status::kUnverified,
      false},
     20992},
    {"orin-fused-wmma-06",
     {"tegra orin", active_device, Op::kDq8caLinear,
      "sarc_linear_dq8ca_coopmat_zpgtr_t128x128k64g44s32mk32ra",
      {128, 128, 64, 4, 4, 32, 16, true}, kTex3dTex2d, projection, Status::kUnverified,
      true},
     35328},
    {"orin-fused-wmma-07",
     {"tegra orin", active_device, Op::kDq8caLinear,
      "sarc_linear_dq8ca_coopmat_zpgtr_t128x128k64g44s32mk32ra",
      {128, 128, 64, 4, 4, 32, 16, false}, kBufTex2d, projection, Status::kUnverified,
      true},
     35328},
    {"orin-fused-wmma-08",
     {"tegra orin", active_device, Op::kQ4gswLinear,
      "sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32c",
      {128, 128, 32, 4, 2, 32, 16, true}, kTex3dTex2d, large_k_projection, Status::kUnverified,
      false},
     37888},
    {"orin-fused-wmma-09",
     {"tegra orin", active_device, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_orin_t256x128k16g42s32bt",
      {256, 128, 16, 4, 2, 32, 16, false}, kTex3dTex2d, short_k_256_projection, Status::kUnverified,
      false},
     45056},
    {"orin-fused-wmma-10",
     {"tegra orin", active_device, Op::kDq8caLinear,
      "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g24s32mk32ra",
      {128, 128, 64, 2, 4, 32, 16, true}, kTex3dTex2d, projection, Status::kUnverified,
      true},
     35328},
    {"orin-fused-wmma-11",
     {"tegra orin", active_device, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x128k16g22s32",
      {128, 128, 16, 2, 2, 32, 16, false}, kTex3dTex2d, short_k_projection, Status::kUnverified,
      false},
     29184}
};

const Candidate* active_candidate() {
  const char* name = std::getenv("ET_VK_SARC_DEV_PROFILE");
  if (name != nullptr) {
    for (const Candidate& candidate : kCandidates) {
      if (std::strcmp(name, candidate.profile) == 0) {
        return &candidate;
      }
    }
  }
  return nullptr;
}

bool active_device(const DeviceInfo& device) {
  const Candidate* candidate = active_candidate();
  return candidate != nullptr &&
      device.name.find("tegra orin") != std::string::npos &&
      candidate->shared_bytes <= device.max_shared_bytes;
}

std::optional<Choice> (*select_before_wmma)(
    const DeviceInfo&, const ShapeInfo&, const std::optional<Choice>&) = nullptr;

std::optional<Choice> select_wmma(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const std::optional<Choice>& table_choice) {
  const auto before = select_before_wmma(device, shape, table_choice);
  const Candidate* candidate = active_candidate();
  if (candidate == nullptr || !active_device(device) ||
      !get_override().allow_unverified || shape.op != candidate->row.op ||
      std::getenv("ET_VK_FORCE_TILED_LINEAR") != nullptr ||
      !q4gsw_coopmat_fits(device, shape, candidate->row)) {
    return before;
  }
  return Choice{candidate->row.kernel_base, candidate->row.dims,
                candidate->row.rowmajor_a};
}

} // namespace

// Called after the other selectors in Overrides.cpp have installed their chain.
void install_orin_wmma_linear() {
  const Candidate* candidate = active_candidate();
  if (candidate == nullptr || !get_override().allow_unverified) {
    return;
  }
  register_rows(&candidate->row, 1);
  Override override = get_override();
  select_before_wmma = override.select;
  override.select = select_wmma;
  set_override(override);
}

} // namespace sarc
} // namespace vkcompute
