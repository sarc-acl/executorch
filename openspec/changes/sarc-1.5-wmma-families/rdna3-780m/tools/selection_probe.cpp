#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 8) {
    return 2;
  }
  using namespace vkcompute::sarc;
  DeviceInfo device;
  device.name = argv[1];
  device.is_amd = std::atoi(argv[2]);
  device.subgroup_size = std::atoi(argv[3]);
  device.min_subgroup_size = std::atoi(argv[4]);
  device.max_subgroup_size = std::atoi(argv[5]);
  device.subgroup_size_control = std::atoi(argv[6]);
  device.max_shared_bytes = std::atoi(argv[7]);
  device.coopmat = device.int8_coopmat = true;
  std::cout << "softmax|" << (get_override().softmax_variant == nullptr
      ? "-" : get_override().softmax_variant) << '\n';
  int id, op, io, weight, layout;
  ShapeInfo shape;
  while (std::cin >> id >> op >> shape.M >> shape.N >> shape.K >> shape.group_size
         >> io >> weight >> layout >> shape.half >> shape.has_bias >> shape.batch
         >> shape.gemv >> shape.io_width_packed >> shape.ignore_alignment) {
    shape.op = static_cast<Op>(op);
    shape.input = shape.output = static_cast<Storage>(io);
    shape.weight = static_cast<Storage>(weight);
    shape.int8_layout = static_cast<Int8Layout>(layout);
    const auto choice = select(device, shape);
    std::cout << id << '|';
    if (choice.has_value()) {
      std::cout << choice->kernel_base << '|' << choice->dims.m << '|'
                << choice->dims.n << '|' << choice->dims.k << '|'
                << choice->dims.wg_size() << '|' << choice->rowmajor_a;
    } else {
      std::cout << '-';
    }
    std::cout << '\n';
  }
  return 0;
}
