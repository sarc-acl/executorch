# sarc-1.5-xe2-prefill-refine: status

**2026-10-04 20:10 UTC — running. Baseline and A/A measured; first SDPA kernels correct and screened; no gated
candidate yet.**

Branch `topic/xe2-prefill-refine`, parent `6a7cc8cc6` (head of `topic/780m-prefill-refine`). Host
`fedora-gpu-eval`, card `b70-0` only (guest PCI `0000:01:00.0`, Vulkan device 0, **`ETVK_DEVICE_INDEX=0`**,
deviceUUID = lock UUID `868023e2-0000-0000-0100-000000000000`), ANV, Mesa 26.2.3. Nothing was run on the second
B70 or on the B580.

## Running now

`tools/screen_sdpa.sh screen1-sdpa topic1 2 <48 profiles>`: kernel-level screen of the Xe2 SDPA variants
(artifacts `raw/screen1-sdpa/`), about one hour. Nothing else uses the card.

## Needs the owner's attention

- **`nvtop` (pid 1952, pts/0, started 17:01 UTC, before the campaign) holds a DRM file of both B70 cards.** It
  is not a workload: every DRM client it owns shows zero engine cycles and zero GPU memory. The campaign guard
  records it per session as an idle monitor (`env.txt`: `idle monitors ... 1952:nvtop`) and would treat it as a
  foreign GPU process, and stop, the moment either number is non-zero. If measuring beside it is not wanted,
  close it and tell me; every session so far ran with it open.
- No other GPU process has appeared. `llama-server`, `comfyui`, `vllm`, `ollama` were inactive at the start.

## Parent control and baseline (re-measured here, not copied)

Builds: `localhost/et-vk-build:rocky10` was built on this host from `tools/Containerfile` (shaderc v2023.8).
`build/parent` = `6a7cc8cc6` exported from the object store; its 53 shipped SPIR-V variants match
`sarc/golden/spirv.json`. `build/topic1` = `9c2f22564`, golden unchanged.

Parent control `s0-parent-verify` (unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff`, no
environment): `CONTROL_RECORDED`. 12 of 12 production-diff cases ALL PASSED, 28 of 28 numeric correctness
cases and 4 of 4 rank-3 cases PASSED, decode 31 tokens, default vs tiled SAME on the real-text prompt (both
schemes) and on the unaligned prompt (4w). Status of pristine `dev/1.5` on this device that a 780M-style
checker would call a failure, recorded as such and compared line by line for every candidate:

| line | parent value | why |
|---|---|---|
| `correctness rc=1` | every numeric case PASSED | the rank-3 M = 128 8da4w cases cannot take the 256-row Xe2 tile and fall back to the tiled kernel |
| `1b 8da4w unaligned: default vs tiled output DIFFER` | DIFFER | also in `sarc-1.5-8da4w-port/results/{b580,b70}` |
| `linear <scheme> rc=1` | rc=1 | texture3d coopmat dispatch reported as unexpected, as on the 780M |
| SDPA tiers all / extended / full | stock kernels, 0 mismatches in 4 / 8 / 4 cases | no SDPA row on Intel, the test reports the missing coopmat dispatch |

Baseline + A/A, session `s1-aa` (pristine parent build against the topic build with no environment, arms
interleaved, median of 5 valid runs, tok/s; `results/xe2/sessions/s1-aa/`):

| cell | parent | topic, no env | A/A | expected (`cells.csv`) | parent vs expected |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 11770.10 | 11770.10 | 0.00 % | 11702.9 | +0.57 % |
| 1B 8da4w | 12412.10 | 12337.30 | -0.60 % | 12412.1 | 0.00 % |
| 3B 4w | 4864.61 | 4864.61 | 0.00 % | 4864.61 | 0.00 % |
| 3B 8da4w | 5278.35 | 5264.78 | -0.26 % | 5251.28 | +0.52 % |
| 8B 4w | 2435.20 | 2438.10 | +0.12 % | 2438.10 | -0.12 % |
| 8B 8da4w | 2737.97 | 2737.97 | 0.00 % | 2737.97 | 0.00 % |

A/A geomean -0.12 %, every cell inside +-0.6 %; 60 timed runs, none rejected; next token parent vs topic SAME
in all six cells on `prompt_2048.txt`, `prompt_check.txt` and `r1304.txt`. The baseline agrees with the
device's `cells.csv` numbers within 0.6 %. The timer resolution is 1 ms, i.e. 0.57 % of a 1B prefill (174 ms).

## How this host differs from the 780M protocol (all in `tools/`, reasons in the script headers)

- **Clock and throttle.** `freq0/throttle/status` reads 1 with reason `pl2` (the card's power limit) in every
  loaded run, at 2580 to 2800 MHz; the 4w cells run power-limited (median 2583 to 2750 MHz), the 8da4w cells
  at 2800 MHz. That is the card's normal clock control, so it is counted per run, not rejected. A run is
  rejected for a thermal reason (`thermal`, `prochot`, `ratl`) in any sample, or a median clock under
  `clkmin` = 2505 MHz (97 % of the lowest per-run median of `s1-aa`). The first attempt of `s1-aa` used the
  780M rule (any throttle flag) and a 0.1 s sampler, rejected all 11 runs it made and was stopped
  (`superseded/s1-aa-attempt1-sampler-0.1s-throttle-flag/`).
- **Sampler.** 10 ms (`tools/sampler.py`); a 1B prefill is 0.17 s, about 17 samples. No effect on tok/s in a
  3 x 4 comparison of sampling periods (`raw/smoke/`).
- **Cooling.** The idle package temperature drifts between 56 and 64 C with the fan hysteresis, with nothing
  running. Waiting for idle + 5 C therefore cost 120 s per run and was not cooling anything; the wait now also
  ends when the temperature has stopped falling. Start temperatures of `s1-aa`: 57 to 67 C.
- **Gate checker.** `gate_check.py` compares the two parent-status lines above against the parent control
  instead of requiring `rc=0` / `SAME`; everything else is absolute. It also requires the next token parent vs
  candidate on the three named prompts.
- **Builds and the guard.** The guard matches a running build container (its command line names `llama_main`),
  so no build runs during a measurement, by construction.

## Xe2 SDPA (order of work, step 2)

Reached from the dev zone without a hook: `impl/sarc_dev/Xe2Sdpa.cpp` registers `kUnverified` SDPA base rows
for `bmg g21` / `bmg g31` that match only while `ET_VK_SARC_DEV_PROFILE` names an `xe2-*` profile. A candidate
environment is therefore `ET_VK_SARC_UNVERIFIED=1` + `ET_VK_SARC_DEV_PROFILE=xe2-...`; every other
configuration selects exactly what the release tables select (`test_sarc_select` unchanged: 31 release rows).
Kernels: the 780M twins built for MMA 8x16x16 (fp16 x fp16 -> fp32, which Xe2 exposes, so accumulation
precision is unchanged) and subgroup 16, plus two new families with a fragment-contiguous shared-memory layout
(`sarc_sdpa_{qk,av}_coopmat_xe2`). All from `tools/gen_xe2.py`.

Base rows (`xe2-sdpa0`: QK^T `sweep_t128x64k32g44s16m8nf`, attn*V `sweep_t64x64k32g44s16m8`):
- SDPA correctness tier `all`, one pass: 4 of 4 PASSED, 0 mismatches, `pairing=ok` (`raw/smoke/`).
- Kernel level, one run (`raw/screen1-sdpa`, ms per layer, S = 2048): QK^T + softmax + attn*V
  8B 8.36 -> 2.20, 3B 6.23 -> 1.65, 1B 4.98 -> 1.61 (3.1 to 3.8x). Not yet an end-to-end number.

## Per-cell numbers against the parent

No gated candidate yet.

## Next

1. Finish the screen; choose QK^T and attn*V per head_dim; build with profile `xe2-refine1`; gate
   (`tools/gate_sdpa.sh`: 12 passes x 3 SDPA tiers, `verify.sh`, timing session, traces).
2. Per-op ETDump breakdown from that gate's traces; phase timing of the two linear kernels (`tools/prof.sh`,
   variants `sarc_dev_prof_*_s16m8*p`).
3. Roofs: igpu-roofline `fast` on `b70-0` (a copy of the fleet tool is staged in the artifact directory).
4. 8da4w linear, then 4w linear.

## Awaiting B580 confirmation

Nothing yet (no gated candidate). Every Xe2 variant so far keeps its shared memory under 46000 bytes (the B70
reports `maxComputeSharedMemorySize` 49152; the B580's value was not read here and should be confirmed), uses
workgroups of at most 1024 invocations and the 8x16x16 / subgroup-16 shapes the shipped Intel rows already use
on both cards, and does not depend on the amount of device memory.
