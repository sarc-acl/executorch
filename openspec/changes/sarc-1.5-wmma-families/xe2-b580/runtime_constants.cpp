#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
  using namespace executorch::runtime;
  runtime_init();
  for (int arg = 1; arg < argc; ++arg) {
    auto loader = executorch::extension::FileDataLoader::from(argv[arg]);
    if (!loader.ok()) {
      return 1;
    }
    auto program = Program::load(&loader.get());
    if (!program.ok()) {
      return 2;
    }
    for (size_t i = 0; i < program->num_methods(); ++i) {
      auto name = program->get_method_name(i);
      if (!name.ok()) {
        return 3;
      }
      if (std::strcmp(*name, "forward") == 0) {
        continue;
      }
      alignas(64) uint8_t memory[65536];
      MemoryAllocator allocator(sizeof(memory), memory);
      MemoryManager manager(&allocator);
      auto method = program->load_method(*name, &manager);
      if (!method.ok() || method->execute() != Error::Ok) {
        return 4;
      }
      const auto meta = program->method_meta(*name);
      if (!meta.ok() || meta->num_inputs() != 0) {
        return 5;
      }
      for (size_t output = 0; output < meta->num_outputs(); ++output) {
        const auto& value = method->get_output(output);
        std::cout << argv[arg] << ',' << *name << ',' << output << ',';
        if (value.isInt()) {
          std::cout << value.toInt();
        } else if (value.isBool()) {
          std::cout << (value.toBool() ? "true" : "false");
        } else {
          return 6;
        }
        std::cout << '\n';
      }
    }
  }
}
