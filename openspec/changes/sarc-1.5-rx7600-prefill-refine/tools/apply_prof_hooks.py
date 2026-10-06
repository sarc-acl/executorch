#!/usr/bin/env python3
"""Dev-zone edits that go with gen_prof.py: candidate rows for the PROF kernels (impl/sarc_dev/Overrides.cpp)
and the ET_VK_DUMP_OUTPUT_DIR raw-output dump in test/sarc_dev/utils.cpp. Idempotent. usage: <executorch tree>"""
import pathlib, sys
root = pathlib.Path(sys.argv[1]) / "backends/vulkan"
o = root / "runtime/graph/ops/impl/sarc_dev/Overrides.cpp"; t = o.read_text()
if "sarc_dev_prof_q4gsw" not in t:
    a = "};\n\n// dq8ca (8da4w) sweep candidates, selected with ET_VK_SARC_DQ8CA_VARIANT."
    assert t.count(a) == 1
    t = t.replace(a, """    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_q4gsw.yaml):
    // the release 780M tile with shader-clock phase counters written over its output.
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_dev_prof_q4gsw_t128x128k32g42s32f32cp", tile(4, 2, true),
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
""" + a)
    a = "};\n\nstd::string& requested_variant() {"
    assert t.count(a) == 1
    t = t.replace(a, """    // 780M phase timing, MEASUREMENT ONLY (glsl/sarc_dev/sarc_dev_prof_dq8ca_zpg.yaml).
    {"", nullptr, Op::kDq8caLinear,
     "sarc_dev_prof_dq8ca_zpg_t128x64k32g42s32p", dq_tile(128, 64, 4, 2),
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified},
""" + a)
    o.write_text(t)
u = root / "test/sarc_dev/utils.cpp"; t = u.read_text()
if "ET_VK_DUMP_OUTPUT_DIR" not in t:
    a = """              output_ref.staging, data_ptr, data_numel, output_spec.dtype);
        }

        // Print output tensor data if output printing is enabled
"""
    assert t.count(a) == 1
    t = t.replace(a, """              output_ref.staging, data_ptr, data_numel, output_spec.dtype);
        }

        // Measurement aid: dump the raw output of this dispatch, for the PROF
        // kernels that write in-kernel phase counters into it.
        if (const char* dump_dir = std::getenv("ET_VK_DUMP_OUTPUT_DIR")) {
          static int dump_index = 0;
          const std::string path = std::string(dump_dir) + "/out_" +
              std::to_string(dump_index++) + "_" + test_case.name() + ".bin";
          if (data_ptr != nullptr) {
            const size_t elem_bytes = output_spec.dtype == vkapi::kHalf ? 2 : 4;
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
              std::fwrite(data_ptr, elem_bytes, data_numel, f);
              std::fclose(f);
            }
          }
        }

        // Print output tensor data if output printing is enabled
""")
    if "#include <cstdio>" not in t:
        t = t.replace('#include "utils.h"\n', '#include "utils.h"\n#include <cstdio>\n#include <cstdlib>\n', 1)
    u.write_text(t)
print("ok")
