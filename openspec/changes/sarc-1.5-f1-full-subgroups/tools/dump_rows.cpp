// dump_rows: every registered SARC selection row and sweep candidate as CSV, for tools/local_size.py.
// GPU-free; built like test_sarc_select (sarc/tools/check.sh step 3).
#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdio>

using namespace vkcompute::sarc;

static void dump(const char* list, const std::vector<Row>& rows) {
  for (const Row& r : rows) {
    std::printf(
        "%s,%s,%d,%s,%u,%u,%u,%u,%s\n",
        list,
        r.device_substr,
        static_cast<int>(r.op),
        r.kernel_base,
        r.dims.sg_grid_x,
        r.dims.sg_grid_y,
        r.dims.subgroup_size,
        r.dims.wg_size(),
        r.status == Status::kVerified ? "verified" : "unverified");
  }
}

int main() {
  std::printf("list,device,op,kernel_base,sg_grid_x,sg_grid_y,subgroup_size,local_size_x,status\n");
  dump("row", rows());
  dump("candidate", candidates());
  return 0;
}
