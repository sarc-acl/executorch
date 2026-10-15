#include <cstdio>
#include <vector>

#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::MmapDataLoader;
using namespace executorch::runtime;

int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  runtime_init();
  auto loader = MmapDataLoader::from(argv[1], MmapDataLoader::MlockConfig::NoMlock);
  if (!loader.ok()) {
    std::fprintf(stderr, "loader error: %u\n", unsigned(loader.error()));
    return 3;
  }
  auto program = Program::load(&loader.get());
  if (!program.ok()) {
    std::fprintf(stderr, "program error: %u\n", unsigned(program.error()));
    return 4;
  }
  const char* names[] = {
      "get_max_context_len", "get_max_seq_len", "get_n_layers",
      "get_vocab_size", "use_kv_cache", "use_sdpa_with_kv_cache",
      "enable_dynamic_shape", "get_bos_id", "get_eos_ids"};
  std::printf("{");
  bool first = true;
  for (const auto* name : names) {
    auto meta = program->method_meta(name);
    if (!meta.ok() || meta->num_inputs() != 0 || meta->num_backends() != 0 ||
        meta->num_instructions() != 0) {
      std::fprintf(stderr, "not a CPU constant method: %s\n", name);
      return 5;
    }
    std::vector<Span<uint8_t>> buffers;
    for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
      auto size = meta->memory_planned_buffer_size(i);
      if (!size.ok() || size.get() != 0) {
        std::fprintf(stderr, "constant method requests tensor memory: %s\n", name);
        return 6;
      }
      buffers.emplace_back(nullptr, 0);
    }
    std::vector<uint8_t> arena(1024 * 1024);
    MemoryAllocator allocator(arena.size(), arena.data());
    HierarchicalAllocator planned(Span<Span<uint8_t>>(buffers.data(), buffers.size()));
    MemoryManager memory(&allocator, &planned);
    auto method = program->load_method(name, &memory);
    if (!method.ok() || method->execute() != Error::Ok) {
      std::fprintf(stderr, "constant method execution failed: %s\n", name);
      return 7;
    }
    std::printf("%s\"%s\":[", first ? "" : ",", name);
    first = false;
    for (size_t i = 0; i < method->outputs_size(); ++i) {
      const auto& value = method->get_output(i);
      if (!value.isInt() && !value.isBool()) {
        std::fprintf(stderr, "unexpected constant output type: %s\n", name);
        return 8;
      }
      auto number = value.isInt() ? value.toInt() : int64_t(value.toBool());
      std::printf("%s%lld", i ? "," : "", static_cast<long long>(number));
    }
    std::printf("]");
  }
  std::printf("}\n");
  return 0;
}
