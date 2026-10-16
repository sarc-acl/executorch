#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace vkcompute::sarc;

int main() {
  DeviceInfo device{"intel bmg g31", false, 32, 16, 32, true, true, true, 49152};
  ShapeInfo shape;
  shape.M = 2048;
  shape.N = 14336;
  shape.K = 4096;
  shape.group_size = 128;
  shape.half = true;
  shape.input = shape.output = Storage::kTexture3D;
  shape.weight = Storage::kTexture2D;
  shape.io_width_packed = true;
  const auto hooks = get_override();
  auto same = [](const auto& a, const auto& b) {
    return a.has_value() == b.has_value() &&
        (!a || a->kernel_base == b->kernel_base);
  };
  struct Case {
    const char* profile;
    Op op;
    const char* kernel;
    int64_t n;
    bool buffer;
  };
  const Case cases[] = {
      {"b70-wmma-dq-n13", Op::kDq8caLinear,
       "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8n13", 14336, true},
      {"b70-wmma-dq-i", Op::kDq8caLinear,
       "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i", 14336, true},
      {"b70-wmma-q4-table", Op::kQ4gswLinear,
       "sarc_linear_q4gsw_coopmat_t128x128k16g44s16m8fli", 14336, true},
      {"b70-wmma-dq-table", Op::kDq8caLinear,
       "sarc_linear_dq8ca_coopmat_zpg_t256x64k32g48s16m8", 14336, true},
      {"b70-wmma-q4-k32", Op::kQ4gswLinear,
       "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g84s16m8flw", 512, false},
      {"b70-wmma-q4-k16", Op::kQ4gswLinear,
       "sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib", 14336, false},
      {"b70-wmma-dq-n26", Op::kDq8caLinear,
       "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8n26", 14336, true},
  };
  int variants = 0;
  for (const auto& c : cases) {
    shape.op = c.op;
    shape.N = c.n;
    for (bool buffer : {false, true}) {
      shape.input = shape.output = buffer ? Storage::kBuffer : Storage::kTexture3D;
      unsetenv("ET_VK_SARC_B70_WMMA_PROFILE");
      const auto original = select(device, shape);
      setenv("ET_VK_SARC_B70_WMMA_PROFILE", c.profile, 1);
      const auto picked = select(device, shape);
      if (!buffer || c.buffer) {
        assert(picked && picked->kernel_base == c.kernel);
        ++variants;
      } else {
        assert(same(original, picked));
      }
      shape.batch = 2;
      assert(!select(device, shape));
      shape.batch = 1;
      shape.gemv = true;
      assert(!select(device, shape));
      shape.gemv = false;
      const auto other_op = shape.op;
      shape.op = c.op == Op::kQ4gswLinear ? Op::kDq8caLinear : Op::kQ4gswLinear;
      const auto unchanged = select(device, shape);
      unsetenv("ET_VK_SARC_B70_WMMA_PROFILE");
      assert(same(unchanged, select(device, shape)));
      shape.op = other_op;
      device.name = "intel bmg g21";
      const auto sibling = select(device, shape);
      setenv("ET_VK_SARC_B70_WMMA_PROFILE", c.profile, 1);
      assert(same(sibling, select(device, shape)));
      device.name = "intel bmg g31";
      assert(get_override().sdpa_fused_add == hooks.sdpa_fused_add);
      assert(get_override().sdpa_fused_serves == hooks.sdpa_fused_serves);
      assert(get_override().softmax_variant == hooks.softmax_variant);
    }
  }
  assert(variants == 12);
  shape.op = Op::kDq8caLinear;
  shape.N = 128;
  unsetenv("ET_VK_SARC_B70_WMMA_PROFILE");
  const auto small = select(device, shape);
  for (const char* name : {"b70-wmma-dq-n13", "b70-wmma-dq-n26", "unknown"}) {
    setenv("ET_VK_SARC_B70_WMMA_PROFILE", name, 1);
    assert(same(small, select(device, shape)));
  }
  shape.input = shape.output = Storage::kTexture3D;
  struct FamilyCase {
    const char* profile;
    Op op;
    int64_t n;
    const char* kernel;
  };
  const FamilyCase family[] = {
      {"xe2-family-a", Op::kQ4gswLinear, 512, "sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib"},
      {"xe2-family-a", Op::kQ4gswLinear, 14336, "sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib"},
      {"xe2-family-a", Op::kDq8caLinear, 14336, "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i"},
      {"xe2-family-a", Op::kDq8caLinear, 1024, "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i"},
      {"xe2-family-b", Op::kQ4gswLinear, 512, "sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib"},
      {"xe2-family-b", Op::kDq8caLinear, 14336, "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8n13"},
      {"xe2-family-b", Op::kDq8caLinear, 1024, "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i"},
  };
  for (const auto& c : family) {
    shape.op = c.op;
    shape.N = c.n;
    setenv("ET_VK_SARC_B70_WMMA_PROFILE", c.profile, 1);
    const auto picked = select(device, shape);
    assert(picked && picked->kernel_base == c.kernel);
  }
  std::cout << "B70_WMMA_PROFILES_PASS 9 profiles, 12 storage variants, 7 family cases\n";
}
