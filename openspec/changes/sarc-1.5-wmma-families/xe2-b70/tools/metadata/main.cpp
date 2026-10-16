#include <executorch/extension/module/module.h>

#include <iostream>

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    executorch::extension::Module module(argv[i]);
    std::cout << argv[i];
    for (const char* method : {"get_max_context_len", "get_max_seq_len", "use_kv_cache", "use_sdpa_with_kv_cache", "enable_dynamic_shape", "get_bos_id"}) {
      const auto result = module.execute(
          method, std::vector<executorch::runtime::EValue>{});
      if (!result.ok() || result->size() != 1) {
        std::cerr << "metadata method failed: " << method << '\n';
        return 1;
      }
      const auto& value = result->at(0);
      std::cout << ',' << method << '=';
      if (value.isInt()) {
        std::cout << value.toInt();
      } else if (value.isBool()) {
        std::cout << value.toBool();
      } else {
        return 1;
      }
    }
    std::cout << '\n';
  }
}
