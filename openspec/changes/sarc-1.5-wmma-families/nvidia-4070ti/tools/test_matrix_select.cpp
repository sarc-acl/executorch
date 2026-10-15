#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace vkcompute::sarc;

int main(int argc, char** argv) {
  assert(argc == 3);
  const int candidate = std::atoi(argv[1]);
  DeviceInfo device;
  device.name = "nvidia geforce rtx 4070 ti super";
  device.coopmat = device.int8_coopmat = true;
  device.subgroup_size = 32;
  device.max_shared_bytes = 49152;
  ShapeInfo shape;
  shape.M = candidate == 11 ? 128 : 2048;
  shape.N = 4096;
  shape.K = candidate == 8 ? 14336 : 4096;
  shape.group_size = 128;
  shape.half = shape.io_width_packed = true;
  shape.op = candidate == 6 || candidate == 7 || candidate == 10
      ? Op::kDq8caLinear : Op::kQ4gswLinear;
  shape.input = shape.output = candidate >= 3 && candidate <= 5 || candidate == 7
      ? Storage::kBuffer : Storage::kTexture3D;
  shape.weight = candidate == 5 ? Storage::kBuffer : Storage::kTexture2D;
  const auto choice = select(device, shape);
  assert(choice.has_value() && choice->kernel_base == argv[2]);
  assert(choice->rowmajor_a == (shape.op == Op::kDq8caLinear));
  assert(std::strcmp(get_override().softmax_variant, "4070ti_nzf") == 0);
  shape.M = 1;
  shape.gemv = true;
  assert(!select(device, shape).has_value());
  std::cout << "profile " << candidate << ": exact storage/kernel, layout, attention control and decode exclusion PASS\n";
}
