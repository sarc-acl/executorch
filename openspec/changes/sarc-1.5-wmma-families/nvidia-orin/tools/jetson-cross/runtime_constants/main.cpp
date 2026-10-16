#include <executorch/extension/module/module.h>

#include <cstdio>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s model.pte\n", argv[0]);
    return 2;
  }
  using executorch::extension::Module;
  Module module(argv[1], Module::LoadMode::Mmap);
  const char* methods[] = {
      "get_max_context_len", "get_max_seq_len", "get_n_layers",
      "get_vocab_size", "use_kv_cache", "use_sdpa_with_kv_cache",
      "enable_dynamic_shape", "get_bos_id", "get_eos_ids"};
  for (const char* name : methods) {
    const auto result = module.execute(name);
    if (!result.ok() || result->empty()) {
      std::fprintf(stderr, "constant method failed: %s\n", name);
      return 1;
    }
    std::printf("%s", name);
    for (const auto& value : *result) {
      if (value.isInt()) {
        std::printf(" %lld", static_cast<long long>(value.toInt()));
      } else if (value.isBool()) {
        std::printf(" %s", value.toBool() ? "true" : "false");
      } else {
        std::fprintf(stderr, "unexpected constant type: %s\n", name);
        return 1;
      }
    }
    std::printf("\n");
  }
  return 0;
}
