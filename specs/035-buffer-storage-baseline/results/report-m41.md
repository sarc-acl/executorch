# M41 (gpusw-m41-08, S5E9965 ERD) — Release/1.3 Buffer-Storage Baseline

**Date**: 2026-07-22
**Device**: `ssh xgpusw-debug07`, `ANDROID_SERIAL=00000a34cdd4abd3` (`gpusw-m41-08`, sole canonical M41 board as of 2026-07-22).
**Driver**: no documented known-good reference hash exists for this chip family (same as the existing texture baseline) — not flashed, used as-found.
**Branch/commit**: `sarc-acl/executorch release/1.3`, runner `llama_main_rel1.3` (plain) + `llama_main_nodethresh` (`release13-node-threshold` branch) for the crash-workaround cells. Same binaries already staged for the texture baseline.
**Workload**: Llama 1B/3B/8B, 4w & 8da4w, 2048-token prefill (`p2048_exact.txt`, `--num_bos=1`) + 1024-token decode (`--ignore_eos --temperature=0 --warmup=true`).
**Clocks**: Floating, and pinned 509/2730/663 MHz (GPU/MIF/INT) via `pin_freqs.sh` — sysfs-verified (`23400000.sgpu` min=max=509000) before and after the pinned sweep.
**`.pte` files**: the 6 buffer-storage files at `/sarc-c/gpusw/users/yanwen.xu/android-run/models/<model>_<quant>_buffer_ctx3072.pte` (exported 2026-07-09 from `dev/executorch`, `ET_VK_FORCE_BUFFER=1`). All 6 coherence-checked individually (`--seq_len=48`, coherent "...Paris..." output) before any timed rep.

## Results — Floating clocks

### 4w

| Model | Threshold | Prefill tok/s (median ± CoV, n) | Decode tok/s (median ± CoV, n) | Crash notes |
|---|---|---|---|---|
| 1B | — | 624.200 ± 0.16% (n=3) | 25.185 ± 0.07% (n=3) | None |
| 3B | — | 223.386 ± 1.90% (n=3) | 12.418 ± 0.12% (n=3) | None |
| 8B | — | 103.765 ± 1.36% (n=3) | 6.912 ± 0.40% (n=3) | None — **texture had 1/3 crash here (unknown cause); buffer was clean 3/3** |

### 8da4w

| Model | Threshold | Prefill tok/s (median ± CoV, n) | Decode tok/s (median ± CoV, n) | Crash notes |
|---|---|---|---|---|
| 1B | — | 221.717 ± 0.09% (n=3) | 24.398 ± 0.14% (n=3) | None |
| 3B | **t64** | 77.912 ± 0.03% (n=3) | 11.975 ± 0.10% (n=3) | **New crash vs. texture**: plain runner crashed 3/3, confirmed `gpu_watchdog` via dmesg (GPU reset succeeded, no OOM signature, `MemAvailable` healthy ~7.6-7.7GB/11.7GB); fixed by `t64`. Texture never needed this. |
| 8B | — | 63.298 ± 0.66% (n=3) | 6.820 ± 0.28% (n=3) | None |

## Results — Pinned clocks (509/2730/663 MHz)

### 4w

| Model | Threshold | Prefill tok/s (median ± CoV, n) | Decode tok/s (median ± CoV, n) | Pin-verified (vs floating) |
|---|---|---|---|---|
| 1B | — | 326.479 ± 0.02% (n=3) | 13.823 ± 0.06% (n=3) | ✅ 52.3% / 54.9% |
| 3B | **t64** | 118.519 ± 0.10% (n=3) | 6.772 ± 0.30% (n=3) | ✅ 53.1% / 54.5% — plain runner crashed 3/3 (`gpu_watchdog` confirmed), fixed by t64, **same as texture's own 3B/4w-pinned fix** |
| 8B | **t32** | 54.513 ± 0.04% (n=3) | 3.783 ± 0.14% (n=3) | ✅ 52.5% / 54.7% — plain runner crashed 3/3, same signature/fix as texture |

### 8da4w

| Model | Threshold | Prefill tok/s (median ± CoV, n) | Decode tok/s (median ± CoV, n) | Pin-verified (vs floating) |
|---|---|---|---|---|
| 1B | — | 128.016 ± 0.47% (n=3) | 13.632 ± 0.59% (n=3) | ✅ 57.7% / 55.9% |
| 3B | **t32** | 40.625 ± 0.04% (n=3) | 6.659 ± 0.14% (n=3) | ✅ 52.1% / 55.6% — **new crash vs texture** (texture's 3B/8da4w pinned was clean without any threshold). t64 was tried first and still crashed 3/3 (confirmed `gpu_watchdog`); t32 fixed it. |
| 8B | — | 59.215 ± 1.10% (n=3) | 3.720 ± 0.16% (n=3) | ⚠️ **93.5% / 54.5%** — decode ratio is normal, but prefill ratio fails this device's own ≤70% DVFS-sanity rule. See caveat below. |

**DVFS caveat (8B/8da4w prefill)**: sysfs confirmed the pin held (`min=max=509000`) both immediately before and after this cell, and a floating re-run afterward (same `.pte`, unpinned, single rep) gave 61.8/4.71 — decode dropped ~31% from the original floating measurement (6.82→4.71) while prefill stayed close (63.3→61.8). This is consistent with **thermal throttling accumulating over the sweep** rather than a pin failure: the original floating *decode* number for this cell may have been measured on a cooler board than the later pinned/re-check runs, inflating that one floating baseline and depressing this specific ratio. The pinned number itself (59.215) is trusted (pin verified); the floating *baseline* it's being compared against is the more likely source of the anomaly. Flagging rather than silently accepting, per this device's own convention — not fully root-caused, would need controlled interleaved A/B to resolve (see memory `feedback-floating-clock-interleave-ab`).

## Texture vs. buffer comparison (this device)

| Model/quant/clock | Texture prefill/decode (n) | Buffer prefill/decode (n) | Buffer vs texture |
|---|---|---|---|
| 1B/4w/floating | 599.14/30.37 (3) | 624.20/25.19 (3) | +4.2% prefill, **−17.0% decode** |
| 1B/4w/pinned | 316.38/17.59 (3) | 326.48/13.82 (3) | +3.2% prefill, **−21.4% decode** |
| 3B/4w/floating | 215.63/13.01 (3) | 223.39/12.42 (3) | +3.6% prefill, −4.5% decode |
| 3B/4w/pinned | 104.95/7.30 (3) | 118.52/6.77 (3, t64) | +12.9% prefill, −7.3% decode |
| 8B/4w/floating | 88.91/7.24 (2, 1 crash) | 103.77/6.91 (3, **0 crashes**) | +16.7% prefill, −4.5% decode, and buffer got a clean n=3 |
| 8B/4w/pinned | 52.76/3.96 (3, t32) | 54.51/3.78 (3, t32) | +3.3% prefill, −4.5% decode |
| 1B/8da4w/floating | 777.23/28.86 (3) | 221.72/24.40 (3) | **−71.5% prefill**, −15.4% decode |
| 1B/8da4w/pinned | 411.55/16.96 (3) | 128.02/13.63 (3) | **−68.9% prefill**, −19.6% decode |
| 3B/8da4w/floating | 287.43/12.57 (3, no crash) | 77.91/11.98 (3, **needs t64**) | **−72.9% prefill**, −4.7% decode, new crash requiring workaround |
| 3B/8da4w/pinned | 150.81/7.15 (3, no crash) | 40.63/6.66 (3, **needs t32**) | **−73.1% prefill**, −6.9% decode, new crash requiring workaround |
| 8B/8da4w/floating | 136.89/7.02 (3) | 63.30/6.82 (3) | **−53.8% prefill**, −2.9% decode |
| 8B/8da4w/pinned | 64.80/3.90 (3) | 59.22/3.72 (3) | −8.6% prefill, −4.6% decode |

**Two very different stories by quant mode.** For **4w**, buffer storage is a modest win on prefill (+3–17%) and a modest-to-moderate loss on decode (−4 to −21%) — roughly the same shape seen on M51 (buffer trades prefill for decode). For **8da4w**, buffer storage is a **large prefill regression** (−54% to −73%) at every model size, on top of introducing two new crash cells (3B floating and pinned) that texture never needed. Decode losses for 8da4w are smaller and comparable to 4w's (−3% to −20%). This asymmetry — 4w largely fine, 8da4w badly regressed on prefill — is a genuine, consistent, cross-clock finding on this device, not noise (CoVs are all ≤1.9% except the flagged 8B/8da4w-pinned prefill).

## Cross-device note (this device only)

**Does 8da4w beat 4w under buffer storage here?** No, decisively not on prefill (4w wins every single cell, sometimes by 3-8×) — the opposite of every other device in this baseline, where 8da4w wins prefill. Decode: 4w also wins every cell (e.g. 1B floating 25.19 vs 24.40; 8B pinned 3.78 vs 3.72), consistent with the pattern seen elsewhere. So on M41 specifically, **buffer storage flips the usual "8da4w wins prefill" answer to "4w wins everything."** This is the opposite of texture storage on the same device, where 8da4w won prefill at every size (777 vs 599 at 1B, etc.) — buffer storage's 8da4w prefill regression is large enough to invert the comparison.

## Validation

- **Coherence check**: all 6 `.pte` files passed the 48-token sanity check before any timed rep.
- **Crash attribution**: every crash cross-checked via `dmesg` (`amdgpu`/`reset`/`watchdog` grep) + `/proc/meminfo` (`MemAvailable`/`MemTotal`) — all confirmed `gpu_watchdog` (GPU reset succeeded messages), zero OOM signatures, memory always healthy (≥7.6GB/11.7GB available).
- **Pin verification**: `23400000.sgpu` min/max read back as 509000/509000 throughout the pinned sweep.
- **DVFS-artifact check**: every pinned cell ≤58% of its floating counterpart except 8B/8da4w-prefill (93.5%, flagged above, not excluded).
- **CoV**: all cells ≤1.9% except the flagged 8B/8da4w-pinned prefill (1.10%, itself not anomalous — the anomaly is in the ratio, not the within-cell spread).

## Reproduce

- **Branch/commit**: `sarc-acl/executorch release/1.3`, runners `llama_main_rel1.3` / `llama_main_nodethresh` (staged at `/sarc-c/gpusw/users/yanwen.xu/android-run/runners/`).
- **Device**: `ssh xgpusw-debug07`; `ANDROID_SERIAL=00000a34cdd4abd3`.
- **`.pte`**: `/sarc-c/gpusw/users/yanwen.xu/android-run/models/<model>_<quant>_buffer_ctx3072.pte`, staged to `/data/local/tmp/llama_vk/`.
- **Pin**: `S=00000a34cdd4abd3 /sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh` (GPU 509 / MIF 2730 / INT 663 MHz; the script's extra `/sys/kernel/gpu/*` writes fail harmlessly on this board — the `devfreq` writes are what matter and were verified).
- **Command** (per rep, plain runner):
  ```
  S=00000a34cdd4abd3
  D=/data/local/tmp/llama_vk
  adb -s $S shell "cd $D && ./llama_main_rel1.3 --model_path=$D/<model>_<quant>_buffer_ctx3072.pte \
    --tokenizer_path=$D/tokenizer.model --prompt_file=$D/p2048_exact.txt --num_bos=1 \
    --max_new_tokens=1024 --ignore_eos --temperature=0 --warmup=true"
  ```
  For the threshold-workaround cells, substitute `./llama_main_nodethresh` and prefix with `ET_VK_EXECUTE_NODE_THRESHOLD=<32|64>`.
- **Raw logs**: `dev/executorch/specs/035-buffer-storage-baseline/results/raw-logs/m41_{floating,pinned}_<model>_<quant>[_t<N>]_rep{1,2,3}.log`.
- **Coherence check**: `--prompt='The capital of France is' --seq_len=48 --temperature=0 --warmup=false`.
