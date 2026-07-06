# Disposition Summary: M5 EVT1 `4w` Linear Coopmat Retune

Status as of 2026-07-05. Schema per `../data-model.md`'s Retuned Shader
Change / Documentation Clarification records.

## Shader changes (`linear_qw_coopmat.glsl`)

| name | risk_level | correctness_result | perf_result | disposition | disposition_reason |
|---|---|---|---|---|---|
| `fp16_accumulate` | precision_risk | **PASS** (K=2048, K=4096, Buffer+Texture3D, rank2+rank3, `abs=0.5`/`rel=0.05`) | informational only, no pre-change baseline diff yet | keep_with_caveat | Correctness confirmed on real M5 EVT1 hardware, known-good driver. Formal tier-1 A/B against a pre-change baseline (`research.md` Decision 1) not yet captured -- see Open below. |
| `loop_flattening` | same_math_code_shape | **PASS** (same run as above) | informational only, no pre-change baseline diff yet | keep | Correctness confirmed; same-math change, no precision risk. Formal perf A/B not yet captured but not required to justify keeping a same-math simplification (spec Clarifications). |
| `vectorized_dequant` | same_math_code_shape | **PASS** (same run as above) | informational only, no pre-change baseline diff yet | keep | Same as `loop_flattening`. |

**How this was actually measured**: the three changes are interleaved in
the same committed diff (commit `133044739`, `research.md` Decision 3), so
one correctness run against the current `HEAD` shader validates all three
together, not individually. Ran `test_coopmat_linear_bench` (rebuilt with
FR-008's new production-K cases) on M5 EVT1 with
`COOPMAT_BENCH_CORRECTNESS_ONLY=1`, on the verified `f14c51b6f8` driver
(see T008 below):

| case | storage | kernel dispatched | result |
|---|---|---|---|
| `linear_q4gsw_M128_K2048_N128` | Texture3D (tiled) | `linear_q4gsw_tiled_...` | PASSED |
| `linear_q4gsw_M128_K2048_N128` | Buffer (**coopmat**) | `linear_q4gsw_coopmat_buffer_texture2d_half` | PASSED |
| `linear_q4gsw_M128_K4096_N128` | Texture3D (tiled) | `linear_q4gsw_tiled_...` | PASSED |
| `linear_q4gsw_M128_K4096_N128` | Buffer (**coopmat**) | `linear_q4gsw_coopmat_buffer_texture2d_half` | PASSED |
| `linear_q4gsw_M128_K4096_N128_rank3batch1` | Buffer (**coopmat**) | `linear_q4gsw_coopmat_buffer_texture2d_half` | PASSED |
| `linear_dq8ca_q4gsw_*` (all 5 of the above, `8da4w` sibling op) | both | `linear_dq8ca_q4gsw_coopmat_...` / tiled | PASSED |

All 10 of this feature's new FR-008 cases PASSED (all `linear_q4gsw`, the
`4w` op this feature actually changed) plus their `linear_dq8ca_q4gsw`
(`8da4w`, unchanged by this feature) siblings also passed. This is the
first real evidence, on the actual target device, that the fp16-accumulate
change does not diverge beyond `abs=0.5`/`rel=0.05` tolerance even at
K=4096 -- the specific risk flagged in-code and in this spec's User Story 3.

**GFLOP/s observed in this same run** (informational -- NOT a Decision-1
pre-change baseline diff; see Open Items):

| shape | tiled GFLOP/s | coopmat GFLOP/s |
|---|---|---|
| K=2048, N=128 | 220.5 | 424.6 |
| K=4096, N=128 | 221.0 | 434.1 |

## Documentation clarification (`linear_dq8ca_qw_coopmat.glsl` / `QuantizedLinear.cpp`)

| files | finding_date | validation_gate | disposition |
|---|---|---|---|
| `linear_dq8ca_qw_coopmat.glsl`, `QuantizedLinear.cpp` (`add_linear_dqa_qw_node`) | 2026-06-30 | None — comment-only, no runtime effect | keep |

## Correctness Harness Extension (FR-008) — DONE

`kCorrectnessShapes`: `{M:128,K:2048,N:128,group_size:128}`,
`{M:128,K:4096,N:128,group_size:128}`; `kRank3CorrectnessShapes`:
`{M:128,K:4096,N:128,group_size:128,batch:1}` — written (T006/T007), and
**a second fix was needed and applied**: `bench_reference()`'s hardcoded
`M > 256 || K > 256 || N > 256` size guard was silently throwing for these
K=2048/4096 cases (marking them `SKIPPED`, not `PASSED` -- a false
"validated" impression with zero actual reference computation). Raised to
`M > 256 || N > 256 || K > 4096` (M/N caps unchanged, so the unrelated
M=1024/N=14336 perf-sweep shapes still correctly skip the expensive O(M·N·K)
CPU reference) so these cases actually execute the check instead of being
silently excluded.

## Device access and driver verification (T008) — DONE

1. **Device access, corrected**: an earlier session ran `adb devices` on
   this workstation (`sj1-yanwen-d01`) directly and wrongly concluded no
   device was reachable. The M5 EVT1 is attached to a *different* host and
   IS reachable: `ssh yanwen.xu@sj1-dmckee-d01` then
   `adb -s 0000088f8e579c33` (confirmed `getprop ro.soc.model` -> `s5e9975`).
   See `.shared-context/instruction-for-ai/devices-and-access.md`.
2. **Driver identity, resolved**: found the driver flashed on the device
   (47,671,472 B, md5 `993d49a9…`) matched none of the four documented
   builds — backed it up to NFS, flashed the documented known-good
   `f14c51b6f8` (md5 `c9861e9906…`, user-confirmed `setenforce 0` step),
   verified exact md5 match plus 16/16 coopmat correctness PASS on a
   prebuilt NFS binary. Full detail in commit `8d6471cad`.

## Build fix (also found and resolved this session)

`test_coopmat_linear_bench` failed to link (`undefined symbol:
add_matmul_coopmat_node`) because the installed `libvulkan_backend.a` was
stale relative to `GemmCoopmat.cpp` (restored in commit `b19116260`).
Per `.shared-context/instruction-for-ai/build.md`'s documented two-step
Android recipe, re-running `cmake --build cmake-out-android-vk --target
install` (19s, mostly cache-hit) reinstalled a fresh `libvulkan_backend.a`
and the link succeeded immediately after.

## Open items

- **Formal tier-1 A/B still not captured**: `research.md` Decision 1 calls
  for comparing against a *fresh pre-change* (pre-`133044739`) M5 EVT1
  build, not just reporting the post-change numbers in isolation (T007/T010
  in `tasks.md`). The GFLOP/s figures above are real but informational
  only until that baseline exists.
- **Tier-1 SPIR-V verification** (`research.md` Decision 4 — confirm
  `OpCooperativeMatrix*KHR` presence and the fp16 accumulator type in the
  compiled shader) not yet done.
- `fp16_accumulate`'s disposition is `keep_with_caveat` rather than a plain
  `keep`, pending that formal perf comparison — a correctness PASS alone
  answers "is it safe," not "did it help."
