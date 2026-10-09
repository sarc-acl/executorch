# sarc-1.5-f1-full-subgroups: log

Task: `/mnt/linux-share/hmz-campaigns/f1-b580/TASK.md`. Branch `topic/f1-full-subgroups` (from `a434a4962`), not
pushed. Host `fedora`, Arc B580 = Vulkan device 0 (`ETVK_DEVICE_INDEX=0`), ANV Mesa 26.2.3, kernel
7.2.9-200.fc44. Artifacts `/mnt/linux-share/hmz-campaigns/f1-b580/.artifacts/`.

## State

(2026-10-09 23:50 UTC) Step 1 committed (`bf9ae6ba3`); step 2 done (0 violators); build `f1` of `bf9ae6ba3`
running; nothing has been run on the GPU yet. Next: the pre-registration below is committed, then the chain
(`tools/chain.sh`): SPIR-V identity, validation layer old / new, session A and its gate, session B and its
gate, kernel level, logits.

## Decision needed from the owner

None.

## Pre-registration (written and committed before any timed run of this task)

Arms: old = build `topic7` (`e1e450530`, copied unchanged from the fused-port campaign), new = build `f1`
(`bf9ae6ba3`). Sessions with the campaign's `session.sh` / `e2e5.sh`, its validity rules and calibration
unchanged (`thresholds.txt`: CLKMIN 2635 MHz, BUSYMAX 5.0 %, 7 valid runs per arm and cell, arms interleaved,
no idle wait, a run above BUSYMAX is invalid and replaced by up to 12 extra pairs a cell). Per cell: median of
the first 7 valid runs of each arm, new / old (`tools/summarize.py`).

- Session A (`sA-fused1`): old against new, both with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused1`.
- Session B (`sB-noenv`): old against new, both with no environment.
- **"No effect"**: every cell's difference is inside the campaign's noise band (+-2 %) AND the geomean of the
  six cells is inside +-1 %, in a session. The verdict is given per session. Anything else is reported as an
  effect with its size and sign; nothing is tuned and nothing is worked around.
- A session that cannot complete a cell because of the desktop's share is repeated once, at least 30 minutes
  later, then reported as it is.
- Kernel level (`test_llama_microbench --linear --regime=prefill --storage=texture3d --skip-correctness`, both
  schemes = 12 shapes, and `--sdpa`; 3 processes per arm, arms interleaved, both environments): per case the
  median of the 3 process medians, new / old, with the same +-2 % / +-1 % reading. The verdict of the task is
  the sessions'; a kernel case outside the band is listed with its size beside them.
- Correctness expectation: the gate of each configuration passes on the new build as it did on the old one,
  and wherever a tool dumps values (last-position logits of `tools/logits_probe`; the `[sdpa-error]` records
  of the SDPA correctness test, which does not dump its outputs) old and new are identical. A difference is a
  finding.

## Record

### Step 1: the change (`bf9ae6ba3`)

`backends/vulkan/runtime/vk_api/Pipeline.{h,cpp}`, `Adapter.cpp`, and the hook list `sarc/HOOKS`. Both compute
stage creation sites call one helper that returns `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`
when (a) a required subgroup size is in force, (b) the device reports `computeFullSubgroups`, (c) the local
size in X (specialization constant 0, as all three places that fill the descriptor build it) is a multiple of
the required subgroup size.

- Cache key: not changed. The flag is a function of the key (required size, specialization constant 0) and of
  a device constant that the pipeline cache now carries.
- Enabled, not only reported: `Adapter.cpp` `create_logical_device` copies the queried
  `VkPhysicalDeviceSubgroupSizeControlFeatures` (so `computeFullSubgroups` as reported) and chains it into
  `VkDeviceCreateInfo` whenever `subgroupSizeControl` is reported; the extension name is in the requested
  list. Nothing to change in `Device.cpp`. To be confirmed by the validation layer (VUID
  `VkPipelineShaderStageCreateInfo-flags-02785` would fire).
- Condition (c) is there because one existing pipeline would otherwise become invalid:
  `Context::get_shader_pipeline` (`ETVK_INSPECT_PIPELINES` builds only) creates any shader with a local size of
  4 x 4 x 1. No production pipeline is affected by (c) (step 2).
- A SARC shader's required subgroup size comes from its yaml `SUBGROUP_SIZE` (`gen_vulkan_spv.py` writes
  `// REQUIRED_SUBGROUP_SIZE = N` into the generated source), so form B covers every cooperative-matrix SARC
  kernel: 401 generated SARC variants and 10 upstream ones carry one.

### Step 2: local size against required subgroup size (`data/local-size.csv`, `tools/local_size.py`)

The local size is a specialization constant set by the op: `sg_grid_x * sg_grid_y * subgroup_size` of the
selection row (`Select.h` `wg_size()`), or S x G for the B580 fused kernel. `tools/dump_rows.cpp` (GPU-free,
built like `test_sarc_select`) prints every row and candidate (`data/rows.csv`, 250); `local_size.py` joins
them with the generated shaders of build `topic7`. Result: 401 lines (33 release zone, 368 development zone),
**0 violators**: every local size in X is a multiple of the shader's required subgroup size, and every row's
subgroup size equals its shader's. 6 development-zone sweep variants are named by no row (cannot be
dispatched); their own grid constants give a multiple too. Host and driver as found: Mesa 26.2.3, the B580
reports `subgroupSizeControl` and `computeFullSubgroups` true, subgroup sizes 16 to 32.
