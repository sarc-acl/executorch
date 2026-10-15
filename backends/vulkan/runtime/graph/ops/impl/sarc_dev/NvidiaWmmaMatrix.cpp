#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace vkcompute {
namespace sarc {
namespace {

bool wide_4070ti_wmma(const ShapeInfo& s) {
  return s.N > 512;
}

bool measured_4070ti_wmma(const ShapeInfo& s) {
  const bool projection =
      (s.K == 2048 && (s.N == 512 || s.N == 2048 || s.N == 8192)) ||
      (s.K == 3072 && (s.N == 1024 || s.N == 3072 || s.N == 8192)) ||
      (s.K == 4096 && (s.N == 1024 || s.N == 4096 || s.N == 14336)) ||
      (s.K == 8192 && (s.N == 2048 || s.N == 3072)) ||
      (s.K == 14336 && s.N == 4096);
  return projection && s.group_size == 128 && s.M <= 2048;
}

bool large_k_4070ti_wmma(const ShapeInfo& s) {
  return measured_4070ti_wmma(s) && s.K > 8192;
}

bool tile256_4070ti_wmma(const ShapeInfo& s) {
  return measured_4070ti_wmma(s) && s.K <= 8192 &&
      (s.M != 256 || s.N % 2048 == 0);
}

bool small_k_4070ti_wmma(const ShapeInfo& s) {
  return measured_4070ti_wmma(s) && s.K <= 8192;
}

struct Candidate4070tiWmma {
  const char* profile;
  Row row;
  uint32_t shared_bytes;
};

const Candidate4070tiWmma kCandidates4070tiWmma[] = {
    {"4070ti-wmma-c01",
     {"4070 ti super", nullptr, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t256x128k16g42s32ga",
      {256, 128, 16, 4, 2, 32, 16, false}, kTex3dTex2d, wide_4070ti_wmma,
      Status::kUnverified, false}, 41472},
    {"4070ti-wmma-c02",
     {"4070 ti super", nullptr, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x128k16g24s32ga",
      {128, 128, 16, 2, 4, 32, 16, false}, kTex3dTex2d, nullptr,
      Status::kUnverified, false}, 37376},
    {"4070ti-wmma-c03",
     {"4070 ti super", nullptr, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x256k16g42s32ga",
      {128, 256, 16, 4, 2, 32, 16, false}, kBufTex2d, wide_4070ti_wmma,
      Status::kUnverified, false}, 29184},
    {"4070ti-wmma-c04",
     {"4070 ti super", nullptr, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x128k16g42s32ga",
      {128, 128, 16, 4, 2, 32, 16, false}, kBufTex2d, nullptr,
      Status::kUnverified, false}, 20992},
    {"4070ti-wmma-c05",
     {"4070 ti super", nullptr, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x128k16g42s32ga",
      {128, 128, 16, 4, 2, 32, 16, false}, kBufBuf, nullptr,
      Status::kUnverified, false}, 20992},
    {"4070ti-wmma-c06",
     {"4070 ti super", nullptr, Op::kDq8caLinear,
      "sarc_linear_dq8ca_coopmat_zpgtr_t128x128k64g44s32mk32ra",
      {128, 128, 64, 4, 4, 32, 16, true}, kTex3dTex2d, nullptr,
      Status::kUnverified, true}, 35328},
    {"4070ti-wmma-c07",
     {"4070 ti super", nullptr, Op::kDq8caLinear,
      "sarc_linear_dq8ca_coopmat_zpgtr_t128x128k64g44s32mk32ra",
      {128, 128, 64, 4, 4, 32, 16, false}, kBufTex2d, nullptr,
      Status::kUnverified, true}, 35328},
    {"4070ti-wmma-c08",
     {"4070 ti super", nullptr, Op::kQ4gswLinear,
      "sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32c",
      {128, 128, 32, 4, 2, 32, 16, true}, kTex3dTex2d, large_k_4070ti_wmma,
      Status::kUnverified, false}, 37888},
    {"4070ti-wmma-c09",
     {"4070 ti super", nullptr, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_orin_t256x128k16g42s32bt",
      {256, 128, 16, 4, 2, 32, 16, false}, kTex3dTex2d, tile256_4070ti_wmma,
      Status::kUnverified, false}, 45056},
    {"4070ti-wmma-c10",
     {"4070 ti super", nullptr, Op::kDq8caLinear,
      "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g24s32mk32ra",
      {128, 128, 64, 2, 4, 32, 16, true}, kTex3dTex2d, measured_4070ti_wmma,
      Status::kUnverified, true}, 35328},
    {"4070ti-wmma-c11",
     {"4070 ti super", nullptr, Op::kQ4gswLinear,
      "sarc_linear_q4gsw_coopmat_t128x128k16g22s32",
      {128, 128, 16, 2, 2, 32, 16, false}, kTex3dTex2d, small_k_4070ti_wmma,
      Status::kUnverified, false}, 29184},
};

const Candidate4070tiWmma* active_4070ti_wmma = nullptr;
std::optional<Choice> (*select_before_4070ti_wmma)(
    const DeviceInfo&, const ShapeInfo&, const std::optional<Choice>&) = nullptr;

std::optional<Choice> select_4070ti_wmma(
    const DeviceInfo& device,
    const ShapeInfo& shape,
    const std::optional<Choice>& table_choice) {
  const auto before = select_before_4070ti_wmma(device, shape, table_choice);
  const auto& candidate = *active_4070ti_wmma;
  if (!before.has_value() || shape.gemv || shape.ignore_alignment ||
      device.name.find("4070 ti super") == std::string::npos ||
      shape.op != candidate.row.op ||
      device.max_shared_bytes < candidate.shared_bytes ||
      !q4gsw_coopmat_fits(device, shape, candidate.row)) {
    return before;
  }
  return Choice{candidate.row.kernel_base, candidate.row.dims,
                candidate.row.rowmajor_a};
}

} // namespace

void install_4070ti_wmma() {
  const char* name = std::getenv("ET_VK_SARC_WMMA_PROFILE");
  if (name == nullptr || *name == 0 || !get_override().allow_unverified) {
    return;
  }
  for (const auto& candidate : kCandidates4070tiWmma) {
    if (std::strcmp(name, candidate.profile) == 0) {
      active_4070ti_wmma = &candidate;
      break;
    }
  }
  if (active_4070ti_wmma == nullptr) {
    std::cerr << "[sarc_dev] unknown WMMA profile: " << name << std::endl;
    std::abort();
  }
  register_candidates(&active_4070ti_wmma->row, 1);
  Override o = get_override();
  select_before_4070ti_wmma = o.select;
  o.select = select_4070ti_wmma;
  set_override(o);
  std::cerr << "[sarc_dev] WMMA profile active: " << name << std::endl;
}

} // namespace sarc
} // namespace vkcompute
