# S25 Ultra (SM-S948U1 / Adreno 840, SM8850) — Release/1.3 Buffer-Storage Baseline

**Date**: 2026-07-22
**Device**: `ssh yanwen.xu@sj1-dmckee-d01`, `ANDROID_SERIAL=R3GL10GC1AP`
**Driver**: stock Qualcomm Adreno as shipped (production retail build) — no root, no custom driver, not flashable.
**Branch/commit**: `sarc-acl/executorch release/1.3`, runner `llama_main_rel1.3` (same binary as the existing texture baseline — storage type is a `.pte`-embedded property, no rebuild needed).
**Workload**: Llama 1B/3B/8B, 4w & 8da4w, 2048-token prefill (`p2048_exact.txt`, `--num_bos=1`) + 1024-token decode (`--ignore_eos --temperature=0 --warmup=true`).
**Clocks**: Floating only. Pinned is `NR` — `adb root` fails outright (`adbd cannot run as root in production builds`), confirmed again this run; no clock-pinning mechanism exists on this device.
**`.pte` files**: the 6 buffer-storage files at `/sarc-c/gpusw/users/yanwen.xu/android-run/models/<model>_<quant>_buffer_ctx3072.pte` (exported 2026-07-09 from `dev/executorch` with `ET_VK_FORCE_BUFFER=1`; vanilla `release/1.3` has no storage-override knob of its own). All 6 coherence-checked individually (`--seq_len=48`, coherent "...Paris..." output) before the timed sweep.

## Results — Floating clocks

### 4w

| Model | Prefill tok/s (median ± CoV, n) | Decode tok/s (median ± CoV, n) | Crash notes |
|---|---|---|---|
| 1B | 547.887 ± 14.16% (n=3) | 27.226 ± 11.73% (n=3) | None — all 3 reps completed (texture had 1 late crash here, n=2) |
| 3B | **NR** | **NR** | 3/3 deterministic crash, `vkQueueSubmit(...) returned -4` at `Adapter.cpp:401`, identical to texture |
| 8B | **NR** | **NR** | 3/3 deterministic crash, same signature as 3B |

### 8da4w

| Model | Prefill tok/s (median ± CoV, n) | Decode tok/s (median ± CoV, n) | Crash notes |
|---|---|---|---|
| 1B | 724.187 ± 1.38% (n=3) | 21.932 ± 2.03% (n=3) | None |
| 3B | 333.550 ± 6.25% (n=3) | 8.625 ± 1.46% (n=3) | None |
| 8B | 127.549 ± 0.54% (n=2) | 4.999 ± 0.15% (n=2) | 1 late intermittent crash (rep3, same `vkQueueSubmit=-4` signature, after nearly-complete decode) |

Crash attribution note: this device has no root, so `dmesg`/`/proc/meminfo` are not accessible — every crash above is an **observed failure signature, not a diagnosed root cause** (same structural limitation as the existing texture-storage companion report).

## Texture vs. buffer comparison (this device)

| Model/quant | Texture prefill / decode (n) | Buffer prefill / decode (n) | Buffer vs texture |
|---|---|---|---|
| 1B/4w | 431.31 / 30.25 (n=2, 1 crash) | 547.89 / 27.23 (n=3, **0 crashes**) | +27% prefill, −10% decode, but buffer got a clean n=3 where texture didn't |
| 1B/8da4w | 688.94 / 23.74 (n=3) | 724.19 / 21.93 (n=3) | +5% prefill, −8% decode |
| 3B/4w | NR (0/3) | NR (0/3) | same deterministic crash both storage types — buffer does not fix it |
| 3B/8da4w | 295.11 / 8.49 (n=3) | 333.55 / 8.63 (n=3) | +13% prefill, +2% decode |
| 8B/4w | NR (0/3) | NR (0/3) | same deterministic crash both storage types |
| 8B/8da4w | 115.63 / 4.96 (n=2, 1 crash) | 127.55 / 5.00 (n=2, 1 crash) | +10% prefill, +1% decode; same late-intermittent-crash pattern under both |

Buffer storage does **not** fix either deterministic 4w crash (3B or 8B) — same `vkQueueSubmit=-4` signature, same call site, at every rep under both storage types. Where cells succeed, buffer is ahead on prefill by 5–27% and roughly flat-to-slightly-behind on decode (−10% to +2%) — noisier and less consistent in direction than the RDNA3 dGPU's buffer-is-uniformly-ahead pattern, but never a regression in the crash-vs-no-crash sense.

## Cross-device note

**Does 8da4w beat 4w under buffer storage on this device?** Prefill: yes where comparable (1B 724 vs 548; 3B/8B 4w is NR so no comparison exists). Decode: 4w's only valid cell (1B, 27.2) beats 8da4w's 1B (21.9); no valid 3B/8B 4w decode number exists to compare against 8da4w's 3B (8.63)/8B (5.00). This device's 4w path is too crash-prone at 3B/8B to answer the full cross-device question — only the 1B comparison is possible, and it matches the pattern seen elsewhere (prefill favors 8da4w, decode favors 4w).

## Validation

- **Coherence check**: all 6 `.pte` files passed the 48-token sanity check before any timed rep.
- **Crash attribution**: observed-signature-only, per this device's no-root limitation (documented, not a gap).
- **CoV**: 1B/4w's high CoV (14.16%/11.73%) is consistent with this device's known floating-clock noise (texture's own 1B/4w CoV was 18.27%/6.23%) — not new, this device has always been noisier than M41/M51 at floating clocks.

## Reproduce

- **Branch/commit**: `sarc-acl/executorch release/1.3`, runner `llama_main_rel1.3` (staged at `/sarc-c/gpusw/users/yanwen.xu/android-run/runners/llama_main_rel1.3`, same binary already used for texture).
- **Device**: `ssh yanwen.xu@sj1-dmckee-d01`; `ANDROID_SERIAL=R3GL10GC1AP` (mandatory `-s`, host has 3 devices attached).
- **`.pte`**: `/sarc-c/gpusw/users/yanwen.xu/android-run/models/<model>_<quant>_buffer_ctx3072.pte`, staged to `/data/local/tmp/llama_vk/` via `adb push`.
- **Command** (per rep):
  ```
  S=R3GL10GC1AP
  D=/data/local/tmp/llama_vk
  adb -s $S shell "cd $D && ./llama_main_rel1.3 --model_path=$D/<model>_<quant>_buffer_ctx3072.pte \
    --tokenizer_path=$D/tokenizer.model --prompt_file=$D/p2048_exact.txt --num_bos=1 \
    --max_new_tokens=1024 --ignore_eos --temperature=0 --warmup=true"
  ```
- **Raw logs**: `dev/executorch/specs/035-buffer-storage-baseline/results/raw-logs/s25u_floating_<model>_<quant>_rep{1,2,3}.log`.
- **Coherence check**: `--prompt='The capital of France is' --seq_len=48 --temperature=0 --warmup=false`.
