# Adreno 840 (S26): does cooperative matrix run on matrix hardware?

Labels: [M] measured or read from a tool on this device; [I] inference from [M] data; [O] outside information,
not checked here.

**Short answer: no route gave ISA.** The Adreno driver returns pipeline statistics but no internal
representations, and no offline Adreno compiler or profiler was available. The statistics are consistent with
cooperative matrix being lowered to ordinary ALU instructions (fp16: 2 FMAs per 32-bit ALU instruction; int8: a
4-way dot per 16-bit ALU instruction), with no instruction class that would identify a matrix unit. That is an
inference from instruction counts, not a disassembly.

## Device and tools

- Galaxy S26 Ultra SM-S948U1, serial <s26-serial>, build `BP4A.251205.006/S948U1UEU1AZAB`, Adreno (TM) 840
  (kgsl `gpu_model` Adreno840v2) [M].
- Driver "Qualcomm Technologies Inc. Adreno Vulkan Driver", build bacff47b1e / I498f01ff61, 11/20/25, compiler
  E031.50.19.13, driver version 0842.19.3 [M].
- igpu-roofline `inspect` (repo commit 463b2ff, inspector sha256 8be3b707…), campaign newdev-20260927 [M].
- ExecuTorch inspection build: `topic/vulkan-pipeline-inspect` (661856b4b) applied on dev/1.5 + `topic/s26-acc-reg`,
  `tools/sarc-build-native.sh <tree> android --inspect`, NDK r29, Vulkan SDK 1.4.350.1 glslc [M].

## Route (a): VK_KHR_pipeline_executable_properties — statistics yes, ISA no

- The extension and `pipelineExecutableInfo` are exposed (capabilities.json) [M].
- Pipelines created with `CAPTURE_STATISTICS | CAPTURE_INTERNAL_REPRESENTATIONS` return one executable
  ("Compute Shader", "Adreno Shader") with 27 statistics and **zero internal representations**, for every pipeline:
  - the roofline matrix, dot and ALU shaders;
  - the ExecuTorch 4w kernels (stock TIN GEMM, SARC adreno row, the `gr` candidate) and the stock 8da4w tiled kernel.
  [M]
- So there is no ISA text on this driver.
- The ExecuTorch inspection build first returned **empty** statistics on Adreno. It enables the extension but not
  the `pipelineExecutableInfo` feature. AMDVLK and PAL tolerate that; Adreno does not.
  `inspect-enable-pipelineExecutableInfo.patch` (local, not committed) adds the feature struct, after which
  statistics appear [M]. This should go into `topic/vulkan-pipeline-inspect`.
- With the capture flags, roofline matrix variants with CHAINS ≥ 2 fail `vkCreateComputePipelines` with -13
  (`matrix_fp16_64x16x16_c2.json`) [M].
  - Whether they fail without the flags is UNVERIFIED here.
  - The confirmed matrix roofs use CHAINS = 1 variants: `matrix_fp16_64x32x16_c1`, `matrix_int8_64x64x32_c1`
    (best-configurations.json) [M].

### Statistics (static counts, whole shader; `statistics.csv`, raw JSON/text in the subdirectories) [M]

Roofline shaders: FEED=0 (operands register-resident), CHAINS=1, so the loop body is one coopMatMulAdd.

| pipeline | all | ALU 32-bit | ALU 16-bit | full / half regs | ALU fiber occupancy |
|---|---|---|---|---|---|
| matrix_fp16 64×16×16 | 268 | 144 | 0 | 13 / 26 | 28 % |
| matrix_fp16 64×32×16 (roof) | 409 | 275 | 0 | 13 / 26 | 28 % |
| matrix_fp16 64×64×16 | 692 | 532 | 0 | 13 / 26 | 28 % |
| matrix_int8 64×16×32 | 528 | 88 | 148 | 23 / 43 | 15 % |
| matrix_int8 64×32×32 | 529 | 41 | 260 | 23 / 43 | 15 % |
| matrix_int8 64×64×32 (roof) | 870 | 65 | 516 | 23 / 43 | 15 % |
| alu_fp16_v1_c4 (roof; 64 fma in body) | 116 | 5 | 67 | 2 / 3 | 100 % |
| alu_fp32_v1_c8 (roof; 128 fma) | 185 | 140 | 0 | 5 / 0 | 78 % |
| dot8_c8 (roof; 64 dotPacked4x8) | 209 | 100 | 64 | 21 / 0 | 18 % |
| ET stock 4w `q4gsw_linear_gemm__tin__w_4x8_nc` | 764 | 97 | 175 | 9 / 25 (13 overall) | 12 % |
| ET SARC 4w `…t64x64k32g21s64m64x32x16` (4 static 64×32×16 MMAs) | 1823 | 1262 | 32 | 26 / 48 | 6 % |
| ET SARC 4w `…x16gr` (ACC_GROUP_FP32_REG) | 2582 | 1465 | 32 | 40 / 47 | 6 % |
| ET stock 8da4w `linear_dq8ca_q4gsw_tiled` | 861 | 484 | 0 | 33 / 8 | 9 % |

Units of the register footprints are the driver's, undocumented.

Instruction count per MMA:
- **fp16:** doubling N adds 131, then 257 ALU-32 instructions. That is ≈128 ALU-32 instructions per 64×16×16
  block [M arithmetic]. A 64×16×16 MMA on a 64-lane subgroup is 256 FMAs per lane, i.e. **2 FMAs per ALU-32
  instruction** [I].
- **int8:** doubling N adds 112, then 256 ALU-16 instructions. That is ≈128 ALU-16 instructions per 64×16×32
  block [M arithmetic]: 512 MACs per lane, i.e. **4 MACs per ALU-16 instruction**, the width of one packed 4×8 dot
  [I].
- **No matrix class:** the driver reports no statistic for a matrix unit. Every MMA instruction is counted as
  ordinary ALU (Complex = 0; the other classes are unchanged by the MMA shape) [M].

## Route (b): AGI / Snapdragon Profiler — not available

- Downloads from this host were blocked. The sandbox denied outbound `curl` to github.com (AGI releases) and
  qualcomm.com. Per instructions, I did not work around it.
- Snapdragon Profiler needs a Qualcomm account (QPM) to download [O].
- Nothing is installed locally: no AGI or Snapdragon Profiler on the host or in `/tool/pkg`. The phone has
  `com.sokatoa.utils` and `com.qti.snapdragon.qdcm_ff` (a display-calibration package, not a profiler) [M].
- On the phone, `perfetto --query` lists only `android.gpu.memory` [M]. With `debug.graphics.gpu.profiler.perfetto`
  unset there is no `gpu.counters` source; setting it was not tried.
- Whether AGI shows Adreno ISA for compute dispatches at all is UNVERIFIED.

## Route (c): offline Adreno compiler — not available

- No Qualcomm offline compiler is available to me. The igpu-roofline campaign records the same
  (`offline-isa/unavailable.json`: "no public offline compiler for this GPU") [M].
- Mesa turnip/ir3 is not a substitute:
  - the local Mesa tree (31e9a6b, 2026-09-16) lists Adreno 840 but does not expose VK_KHR_cooperative_matrix (no
    cooperativeMatrix feature in `src/freedreno/vulkan`) [M];
  - it would be a different compiler from Qualcomm's in any case.

## Does fp16 differ from int8?

Yes, in the statistics [M]:
- **Instruction class:** fp16 MMA work is counted as 32-bit ALU, int8 as 16-bit ALU.
- **Registers:** int8 uses more (23/43 vs 13/26).
- **Occupancy:** int8 is lower (15 % vs 28 %).

Both scale linearly with the number of 64×16 output blocks, ≈128 instructions per block [M arithmetic].

## Consistency with the roofs (all [I], built on [M] counts)

- **fp16:** the fp32 FMA roof (3.972 TFLOP/s, 1 FMA per ALU-32 instruction) corresponds to an ALU-32 issue rate
  that, at 2 fp16 FMAs per instruction, gives ≈7.9 TFLOP/s. The fp16 matrix roof is 6.95 (88 % of that) and the
  fp16 FMA roof 7.76. So a packed-fp16 ALU lowering explains matrix_fp16 ≈ 0.9× FMA without a matrix unit.
- **int8:** the counts give no reason for the 1.74× over dot_int8 through instruction width. Both use one 4-way dot
  per instruction.
  - The dot8 roof shader issues ≈2.6 ALU instructions per dot: a separate accumulate add plus the operand
    recurrence (164 ALU per 64 dots).
  - The matrix loop issues ≈1.1 per dot4 (581 ALU per 512 dot4).
  - That 2.3× instruction overhead can account for the 1.74× without matrix hardware; a real dot8 roof would then
    be higher than 7.05.
  - Alternatively the int8 path uses a faster instruction that the driver still counts as ALU-16.
  - The statistics cannot separate these two cases.

**Verdict:** without ISA, it is not proven whether Adreno 840 has a matrix unit. Nothing in the driver's
statistics indicates one, and the counts are fully consistent with ALU lowering for fp16 [I]. For int8 the
question stays open [I].
- Next evidence that would settle it: Adreno ISA from Snapdragon Profiler or an internal Qualcomm offline compiler
  (needs an account or licence), or a dot8 roof variant with fused accumulation and no recurrence in the loop.

## Commands

```
# roofline statistics: igpu-roofline campaign newdev-20260927, results/…/s26/<s26-serial>/pipeline-inspection/
# ExecuTorch dumps (device dir /data/local/tmp/et15-inspect):
J=12 tools/sarc-build-native.sh <tree> android --inspect        # tree = dev/1.5 + 661856b4b + a5bd11cfd + patch
adb -s <s26-serial> shell 'cd /data/local/tmp/et15-inspect && mkdir -p stats-sarc4w && ET_VK_SARC_UNVERIFIED=1 \
  ET_VK_DUMP_PIPELINE_STATS=stats-sarc4w ./test_llama_microbench --linear --model=3.2-1b --scheme=4w \
  --regime=prefill --storage=texture3d --skip-correctness'
#   stock: no env; gr: + ET_VK_SARC_Q4GSW_VARIANT=t64x64k32g21s64m64x32x16gr; 8da4w: --scheme=8da4w, no env
# The microbench exits 1 on these runs only because texture3d cases are labelled "unexpected_coopmat".
```

These are compile-time statistics; no timing was taken, and runs were short and spaced 30 s apart.

## Files

- `statistics.csv`: the table above, all statistics columns.
- `roofline-pipeline-inspection/*.json`: raw inspector output (copied from the campaign).
- `executorch-dumps/*.txt`: raw ET_VK_DUMP_PIPELINE_STATS output.
- `inspect-enable-pipelineExecutableInfo.patch`: the Adapter.cpp fix for the inspection build.
