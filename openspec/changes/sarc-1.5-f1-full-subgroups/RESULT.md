# F1 on the Jetson Orin Nano: pipelines created with the full-subgroups flag

Task `f1-orin`, branch `topic/f1-full-subgroups-orin` (not pushed). Device `duck-naughty`: Jetson Orin Nano, L4T
R39.2.1, NVIDIA driver 595.78, Vulkan 1.4.329, 15 W mode and clocks as found (devfreq 306 to 612 MHz); measured
2026-10-10 16:27 UTC to 2026-10-11 01:46 UTC. Old = build `topic4` (`0bed38090`, the fused-port campaign's final
build, bundle copied unchanged), new = build `f1` (`0f038a736`). Running record: `LOG.md`; raw data: `data/`.

**Answer.** Creating the cooperative-matrix pipelines with `REQUIRE_FULL_SUBGROUPS` changes nothing measurable in
the end-to-end numbers of this device. Correctness: every gate item of both configurations reads on the new build
as on the old one, and every dumped value (492 logits vectors, the SDPA error records) is identical. Speed:
**no effect** by the pre-registered rule in both sessions (`orin-fused1`: geomean -0.05 %, largest cell -0.22 %;
no environment: +0.00 %, largest cell +0.05 %). Three things a reader should know before relying on this:

1. **The validation layer is not installed on the device**, so there is no before / after count of VUID 10770
   here. The flag is proven by a debug print in a build that is not committed (section 3).
2. **Kernel level, the SDPA suite does not read "no effect" by the rule**: 8 decode-size cases (one token, 1 to
   2 ms) of the stock attention kernels are 17 to 25 % slower on the new build in the 3-process table. These
   kernels have no required subgroup size, so the patch does not change their pipelines, and a check made
   afterwards (10 processes per arm, not pre-registered) finds the same separate time levels in both arms
   (section 7). I read it as run-to-run variation of these short cases, not as an effect; it is not proven.
3. Two gate items fail on the new build exactly as on the old one ("as the parent"): the buffer-storage
   production-diff cases, and the SDPA correctness status with no environment (section 6).

## 1. The change (`0f038a736`, "Release-zone change (owner decision 2026-10-10): full-subgroups flag")

`git am` of `f1-full-subgroups.patch`: applied cleanly, no adaptation. `git patch-id --stable` of the patch file
and of the commit: `4fe6e4868068acf93cba8e8f20f49b63d53b7cd4` both. `git diff --name-only da92b4b3c HEAD` outside
this directory: `backends/vulkan/runtime/vk_api/{Adapter.cpp,Pipeline.cpp,Pipeline.h}`, `sarc/HOOKS` (+50 / -11),
nothing else. Between the old build's commit and the new one (`0bed38090..0f038a736`) the same four files are the
only difference outside `openspec/`.

## 2. Specification

The sentences relied on are those quoted in the task file and in `b580-example/RESULT.md` section 2 (VUID
`RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770`, `VkPipelineShaderStageCreateInfo-pNext-02757`, `-flags-02785`).
The `vulkan-docs` server's exact lookup returned nothing for the three VUIDs in this session; no new quote is added.

## 3. Proof that the flag is set (`data/flagproof/`)

`VK_LAYER_KHRONOS_validation` is not installed on `duck-naughty` (`/etc/vulkan/explicit_layer.d` empty, no other
layer directory, no package), and nothing was installed. Fallback: build `f1dbg` = the tree of `0f038a736` plus
`data/flagproof/flagprint.patch` (one `fprintf` of the stage flags inside the patch's helper, applied to the
build tree only, never committed, never timed). One linear suite (`--linear --regime=prefill --scheme=4w`) and one
SDPA tier (`--sdpa-correctness-only --sdpa-tier=all`) per configuration:

| configuration | pipelines with a required subgroup size (32) | of them with stage flags `0x2` | pipelines without one | of them with flags `0x0` |
|---|---:|---:|---:|---:|
| `orin-fused1` | 16 (12 linear, local size X 256; 4 fused attention, X 32) | 16 | 19 | 19 |
| no environment | 12 (linear, X 128 and 256; the attention kernels are the stock ones) | 12 | 21 | 21 |

`0x2` is `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`; every line prints
`computeFullSubgroups=1`. The device reports `subgroupSizeControl = true`, `computeFullSubgroups = true`, subgroup
size 32 to 32 (`vulkaninfo`). Old build: both creation sites of `Pipeline.cpp` at `0bed38090` pass the literal
`0u` as stage flags (the patch's `-` lines); it was not run with a print. What this does not show: that the
driver accepts the pipelines as valid (no layer); no pipeline creation failed in any run of this task.

## 4. Local size against required subgroup size (`data/local-size.csv`, `tools/local_size.py`)

Re-run on this branch (`data/rows.csv`: 294 rows and candidates; generated shaders of build `f1`): 398 lines (147
release zone, 251 development zone), **0 violators**; 10 upstream shaders with a required size are listed, their
local size is not derived. The Orin fused kernel's local size is X x Y x S of its name
(`SdpaOrinFused.cpp`). What this device dispatches (kernels named in the logs of `verify.sh` and of the SDPA
passes, `data/*/dispatched-kernels.csv`), plus the two unverified table rows:

| configuration | kernel | local size X | required | multiple |
|---|---|---:|---:|---|
| no environment | `sarc_linear_q4gsw_coopmat_t256x128k16g22s32` | 128 | 32 | yes |
| no environment | `sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32` | 256 | 32 | yes |
| no environment | `sarc_linear_dq8ca_coopmat_zpgtr_t128x128k64g44s32mk32ra` | 512 | 32 | yes |
| table rows, verified, not seen in these logs | `sarc_linear_q4gsw_coopmat_t128x128k16g22s32` | 128 | 32 | yes |
| table rows, unverified (need `ET_VK_SARC_UNVERIFIED=1`) | `sarc_sdpa_qk_coopmat_4070ti_t128x64k32g42s32`, `sarc_sdpa_av_coopmat_4070ti_t64x64k32g42s32` | 256 | 32 | yes |
| `orin-fused1` | `sarc_linear_q4gsw_coopmat_orin_t256x128k16g42s32bt` | 256 | 32 | yes |
| `orin-fused1` | `sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32c` | 256 | 32 | yes |
| `orin-fused1` | `sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g24s32mk32ra` | 256 | 32 | yes |
| `orin-fused1` | `sarc_dev_orin_sdpa_fused3sb_d64_t32x32g11s32rko`, `..._d128_t16x64g11s32rko` | 32 | 32 | yes |

## 5. Build and SPIR-V identity (`data/spirv-identity.txt`, `data/build/`)

`f1` is built by the campaign's `build-orin.sh` from an export of `0f038a736` (tree = `topic4`'s tree plus the
changed paths), image `localhost/et-jetson-cross:jp7.2.1` `d182f725bb32` as `topic4`, the image's own `glslc`
(`GLSLC_DIR` unset). 3240 `.spv` files in each build tree, sha256 lists identical: **0 differences**;
`libllama_runner.so` has `topic4`'s hash. `spirv_golden.py`: FAIL with the same 14 DIFF lines as `topic4`: as
the parent. One difference in provenance: the recipe hash in `f1.src.txt` is not `topic4`'s, because the campaign
added the optional `GLSLC_DIR` switch to `build.sh` / `container.sh` after `topic4` was built; it is off here.
The campaign's tools run from a copy whose only difference is two root constants (`data/tools.diff`).

## 6. Correctness on the new build, both configurations

| item | `orin-fused1` (`sA-fused1`) | no environment (`sBn-noenv`, `sB-noenv`) |
|---|---|---|
| unmodified `verify.sh` against the parent snapshot (`gate_check.py verify`) | ACCEPT, 0 findings (control `s0-parent-verify`); 22 of 22 runner calls rc 0 | ACCEPT, 0 findings, 0 differing lines against `s0n-noenv`; 22 of 22 rc 0 |
| `verify.out` against the old build's, rates removed | 0 differing lines of 34 (old: campaign `s5-c1`) | 0 differing lines of 34 (old: campaign `s7n-noenv`) |
| production-diff, 12 logs | 6 ALL PASSED (texture3d), 6 FAILED (buffer); old build 6 and 6, result lines equal log by log: **as the parent** | the same: 6 and 6, 0 differing lines: **as the parent** |
| SDPA tiers `all` / `extended` / `full`, 12 passes each | 36 passes (+ `peaked`, `fused` x 3): 222 of 222 cases PASSED, 0 mismatches, `gate_check.py sdpa` ACCEPT | 36 passes: 0 mismatches in every case, but every pass returns 1 and `gate_check.py sdpa` reads REJECT (576 findings): the stock kernels run where the test expects cooperative-matrix ones. The old build returns 1 in every tier too, and the pristine parent control does: **as the parent** |
| SDPA records, old against new, same session (`sdpa-records.txt`; the test does not dump outputs) | IDENTICAL in 4 tiers; all 39 gate passes of these tiers equal the old build's records | IDENTICAL in 4 tiers; all 36 passes equal |
| next token, four prompts, six cells | SAME in 24 of 24 | SAME in 24 of 24 |
| last-position logits, 41 prompts x 6 cells, old against new, byte for byte (`data/logits/`) | 246 vectors, 0 differ | 246 vectors, 0 differ |
| gate verdict | `GATE_ACCEPTED`, all steps passed | `verify`: `GATE_ACCEPTED`; timed session: `SESSION_ACCEPTED` |

`verify.sh` prints `correctness rc=1` and `linear ... rc=1` in both configurations, in lines equal to the old
build's (the buffer cases of the microbench do not dispatch a cooperative-matrix kernel on this device).

## 7. Speed, old against new

Pre-registered in `LOG.md` (commit `ed406de8f`, before the first timed run): "no effect" = every cell inside
+-2 % and the geomean inside +-1 %. Tok/s, 2048-token prefill, median of the first 5 valid runs per arm, arms
interleaved; tables printed by `tools/session_table.py <runs.csv>`. Throttle state 0 in every clock sample.

**Session A, `orin-fused1`** (`sA-fused1`, timed runs 17:50 to 18:18 UTC):

| cell | old `topic4` | new `f1` | new / old | difference | spread old / new | valid old / new | invalid | clock MHz |
|---|---:|---:|---:|---:|---|---|---|---|
| 1B 4w | 1651.61 | 1651.61 | 1.00000 | +0.00 % | 0.2 / 0.2 % | 5 / 5 | none | 612 to 612 |
| 1B 8da4w | 1523.81 | 1520.42 | 0.99778 | -0.22 % | 0.6 / 0.4 % | 5 / 5 | none | 612 to 612 |
| 3B 4w | 659.79 | 659.58 | 0.99968 | -0.03 % | 0.1 / 0.1 % | 5 / 5 | none | 612 to 612 |
| 3B 8da4w | 595.18 | 595.35 | 1.00029 | +0.03 % | 0.1 / 0.1 % | 5 / 5 | none | 612 to 612 |
| 8B 4w | 305.58 | 305.31 | 0.99911 | -0.09 % | 0.1 / 0.2 % | 5 / 5 | none | 612 to 612 |
| 8B 8da4w | 277.36 | 277.39 | 1.00014 | +0.01 % | 0.1 / 0.1 % | 5 / 5 | none | 612 to 612 |

Geomean over 6 cells: 0.99950 (-0.05 %); largest cell difference -0.22 %; 60 timed runs, 60 valid. Verdict: NO EFFECT (every cell inside +-2 %, geomean inside +-1 %)

**Session B, no environment** (`sB-noenv`, timed runs 20:00 to 20:37 UTC):

| cell | old `topic4` | new `f1` | new / old | difference | spread old / new | valid old / new | invalid | clock MHz |
|---|---:|---:|---:|---:|---|---|---|---|
| 1B 4w | 891.21 | 890.82 | 0.99956 | -0.04 % | 0.2 / 0.2 % | 5 / 5 | none | 612 to 612 |
| 1B 8da4w | 823.81 | 823.81 | 1.00000 | +0.00 % | 0.1 / 0.1 % | 5 / 5 | none | 612 to 612 |
| 3B 4w | 360.50 | 360.50 | 1.00000 | +0.00 % | 0.1 / 0.2 % | 5 / 5 | none | 612 to 612 |
| 3B 8da4w | 320.25 | 320.40 | 1.00047 | +0.05 % | 0.1 / 0.1 % | 5 / 5 | none | 612 to 612 |
| 8B 4w | 189.77 | 189.81 | 1.00018 | +0.02 % | 0.1 / 0.1 % | 5 / 5 | none | 612 to 612 |
| 8B 8da4w | 170.51 | 170.48 | 0.99984 | -0.02 % | 0.0 / 0.1 % | 5 / 5 | none | 612 to 612 |

Geomean over 6 cells: 1.00001 (+0.00 %); largest cell difference +0.05 %; 60 timed runs, 60 valid. Verdict: NO EFFECT (every cell inside +-2 %, geomean inside +-1 %)

Neither session was repeated. The absolute rates agree with the campaign's (`s8-pristine`: final stack 1651.61 /
1521.55 / 659.58 / 595.18 / 305.54 / 277.21, pristine 891.60 / 823.81 / 360.50 / 320.35 / 189.84 / 170.48).

**Kernel level** (`data/kernel/kernel.csv`; 3 processes per arm, interleaved; speed new / old per case from the
median of the process medians):

| environment | suite | cases counted | geomean | min | max | outside +-2 % |
|---|---|---:|---:|---:|---:|---:|
| `orin-fused1` | linear-4w | 24 | +0.01 % | -0.07 % | +0.15 % | 0 |
| `orin-fused1` | linear-8da4w | 24 | +0.02 % | -0.05 % | +0.08 % | 0 |
| `orin-fused1` | sdpa | 27 | **-3.89 %** | **-24.68 %** | +0.21 % | **4** |
| no environment | linear-4w | 24 | -0.00 % | -0.09 % | +0.14 % | 0 |
| no environment | linear-8da4w | 24 | +0.05 % | -0.25 % | +0.52 % | 0 |
| no environment | sdpa | 36 | **-2.26 %** | **-20.21 %** | +1.89 % | **4** |

The linear suites read no effect. The SDPA suite does not: by the rule it is an effect in both environments.
The 8 cases outside the band are the four parts (qk, softmax, av, total) of ONE decode-size case (M = 1, stock
tiled kernels) per environment: Llama 3.2 3B under `orin-fused1` (-24.68 / -20.54 / -23.58 / -24.33 %) and
Llama 3.2 1B with no environment (-19.72 / -17.45 / -20.21 / -19.74 %). Every prefill-size case (S = 2048) is
inside +-0.2 %, the fused attention kernel included (`total` / `coopmat`: 8B 18250.5 -> 18248.2 us, 3B 13893.1
-> 13886.0 us, 1B 11514.4 -> 11526.8 us; its 9 per-stage placeholders are not counted). The microbench's return
status is the same in both arms in all 36 runs (1 for linear; SDPA 0 with the profile, 1 without).

What is known about the 8 cases: (a) the stock tiled kernels have no required subgroup size, so the new build
creates their pipelines with stage flags 0 as the old one does (section 3); (b) inside one arm the three
processes already sit on separate levels (3B, `orin-fused1`: old 1513 / 1510 / 2006 us, new 1215 / 2005 /
2000 us); (c) a check made after seeing this, NOT pre-registered and replacing nothing (`tools/f1-decode.sh`,
10 processes per arm, decode cases only, 01:09 to 01:46 UTC; `data/kernel-decode/decode.txt`):

| environment | model | old, range (us) | new, range (us) | old median | new median | new / old (speed) |
|---|---|---|---|---:|---:|---:|
| `orin-fused1` | 8B | 2489 to 2526 | 2349 to 2522 | 2515 | 2514 | +0.03 % |
| `orin-fused1` | 3B | 1149 to 2017 | 1513 to 2022 | 1871 | 2004 | -6.61 % |
| `orin-fused1` | 1B | 2009 to 2294 | 2278 to 2292 | 2289 | 2288 | +0.06 % |
| no environment | 8B | 2174 to 2507 | 2142 to 2510 | 2495 | 2501 | -0.25 % |
| no environment | 3B | 1200 to 1512 | 1010 to 1503 | 1261 | 1208 | +4.36 % |
| no environment | 1B | 1709 to 2281 | 1375 to 2280 | 2062 | 2111 | -2.33 % |

Both arms cover the same levels (about 1.2, 1.5 and 2.0 ms for 3B), the sign changes from case to case, and the
1B case that read -19.74 % reads -2.33 %. The GPU clock was 306 MHz (idle) before and after every one of
these processes; the cases last 1 to 2 ms each. That fits a clock-ramp variation of very short dispatches
that the medians of 3, or of 10, processes do not average out; the cause was not measured. Decode speed end to
end was not a subject of this task's sessions (prefill only); `verify.out`'s decode lines are compared with the
rates removed.

**Verdict (pre-registered, per session): no effect** in session A and in session B. Kernel level: no effect for
the linear suites; the SDPA suite is outside the band on decode-size stock-kernel cases, as described.

## 8. Not done, and limits

- No validation-layer run (layer absent): the other messages the B580 run lists (10167, 06266, 00344, 09600,
  02252, 08740 / 08742) are neither confirmed nor excluded for this driver.
- The old build was not run with a flag print; its flags are read from the source.
- Stock ExecuTorch was not re-measured (task). Nothing was tuned; no tile, shader or table changed.

## 9. Owner decision of 2026-10-10 (16:25 UTC): `nvtop` is not a foreign GPU process

Applied as written, with nothing to change: the foreign-process guard of this task's copy of the tools
(`common.sh` `others()`) matches known GPU programs by name and has never contained `nvtop`, so a running
`nvtop` could not invalidate a run here. No run of this task was rejected for it or for any other foreign
process (120 of 120 timed runs valid); every other validity rule is as in the campaign.
