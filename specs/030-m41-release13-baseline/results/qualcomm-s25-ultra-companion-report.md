# Companion: Release/1.3 Baseline on Qualcomm S25 Ultra (2026-07-14)

> **Companion to [`m41-release13-baseline-report.md`](./m41-release13-baseline-report.md).** This
> device is a secondary/cross-device reference, same as M41 — not the Samsung M5 EVT1 active
> mission target, and not directly comparable to M41 either (different SoC vendor, different
> Vulkan driver stack entirely). Collected same-day, same session, same methodology (runner,
> workload, rep count) as the M41 study, so it's presented alongside it for convenience — not as
> an apples-to-apples performance comparison.

**Device**: Galaxy S25 Ultra, `SM-S948U1`, serial `R3GL10GC1AP`, `ssh yanwen.xu@sj1-dmckee-d01`.
SoC: `SM8850` (Qualcomm Snapdragon), Adreno 840 GPU — a fundamentally different GPU
vendor/architecture from M41's Exynos/Xclipse (AMD-RDNA-derived) and the M5 EVT1's Xclipse 970.

**Root/clock access**: `adb root` fails outright — `"adbd cannot run as root in production
builds"`. **No clock pinning is possible on this device.** Every number below is floating-clock;
there is no pinned counterpart to report, unlike the M41 study's four-table matrix.

**Driver**: stock Qualcomm Adreno driver as shipped with the ROM — no custom `.so`, no flash
performed or possible (this is a production retail-channel build, not a validation board).

**Methodology**: same `llama_main_rel1.3` runner binary as the M41 study (plain `release/1.3`, no
coopmat), same texture-storage PTEs, same 2048-prefill/1024-decode workload
(`p2048_exact.txt --num_bos=1 --max_new_tokens=1024 --ignore_eos --temperature=0 --warmup=true`),
same 3-rep sampling. A short coherence check (`--seq_len=48 --warmup=false`) passed cleanly before
the sweep ("Paris. The capital of the United States is Washington, D.C....").

## Results

| Model | Quant | Rep 1 | Rep 2 | Rep 3 | Mean | CoV |
|---|---|---:|---:|---:|---|---|
| 1B | 4w | 487.04 / 31.58 | 375.57 / 28.91 | **CRASHED** (late) | 431.31 / 30.25 (n=2) | 18.27% / 6.23% |
| 3B | 4w | **CRASHED** | **CRASHED** | **CRASHED** | n/a (0/3) | n/a |
| 8B | 4w | **CRASHED** | **CRASHED** | **CRASHED** | n/a (0/3) | n/a |
| 1B | 8da4w | 650.37 / 23.41 | 736.96 / 24.11 | 679.50 / 23.70 | 688.94 / 23.74 | 6.40% / 1.48% |
| 3B | 8da4w | 277.06 / 8.40 | 302.38 / 8.54 | 305.90 / 8.55 | 295.11 / 8.49 | 5.33% / 1.00% |
| 8B | 8da4w | 118.09 / 4.96 | **CRASHED** (late) | 113.17 / 4.96 | 115.63 / 4.96 (n=2) | 3.01% / 0.11% |

*(prefill / decode tok/s per rep. All floating clocks — no pinned column exists for this device.)*

## Crash analysis

Every crash on this device shares one signature — `vkQueueSubmit(device_queue.handle, 1u,
&submit_info, fence) returned -4`, raised from `submit_cmd` at
`backends/vulkan/runtime/vk_api/Adapter.cpp:401`. This is a **different call site** than every
crash observed on M41 (`vkQueueWaitIdle` at `Context.cpp:234`) — expected, since this is a wholly
different driver stack (Qualcomm's proprietary Adreno driver vs. AMD-PAL/Xclipse) with its own
failure mode, not the same GPU-watchdog mechanism. No `dmesg`/`/proc/meminfo` attribution check
was performed here (no root — `dmesg` is not accessible), so root cause is unconfirmed; this is
recorded as an observed failure signature, not a diagnosed one.

Two distinct patterns:

- **3B-4w and 8B-4w: crash immediately on every single rep (6/6 total), with zero generated
  text.** Deterministic and total — this looks like a hard incompatibility for the 4w-texture
  path at these model sizes on this GPU (plausibly a shape, allocation-size, or texture-dimension
  limit specific to Adreno, hit only once tensors get large enough), not a transient fault. No
  workaround was attempted or is known to exist for this signature/device.
- **1B-4w rep 3 and 8B-8da4w rep 2: crash late**, after generating nearly the full 1024-token
  decode, immediately before the stats line would have printed. Intermittent — the same
  (model, quant) config succeeded on other reps — a materially different, less severe failure
  mode than the deterministic early crashes above.

## Notable differences from M41

- **4w is far more crash-prone here than on M41.** On M41, only one cell (8B/4w/**pinned**)
  failed deterministically, and a workaround existed (`ET_VK_EXECUTE_NODE_THRESHOLD`). Here, two
  of three 4w models fail deterministically **at floating clocks** — there is no pinning to blame,
  and no known workaround for this driver's failure signature.
- **8da4w is comparatively robust on both devices** — the only 8da4w failures on either device are
  the single intermittent late-crash types (M41's floating 8B rep 2, this device's 8B rep 2),
  never the deterministic every-rep failures seen in 4w.
- **CoV is markedly higher here than on the equivalent M41 floating cells** (e.g., 1B-4w's 18.27%
  here vs. M41's 0.72% for the same cell) — both from the reduced sample size (n=2 after a crash)
  and from this device's own floating-clock cold-start/throttle behavior, which appears more
  pronounced than M41's. Per this workstream's own convention (constitution Principle VII) for
  floating tiled configs, this mean should be read as directional, not a stable reference point.

## Framing

Per the M41 report's own convention, this device is also a secondary/cross-device reference —
not the Samsung M5 EVT1 active mission target, and additionally not directly comparable to M41
(different GPU vendor entirely, no shared driver lineage, no pinned counterpart). Its value here
is as an independent data point on how the vanilla `release/1.3` baseline behaves outside the
Xclipse family, collected with identical methodology to M41 for that reason alone — not as a
head-to-head performance comparison between the two secondary devices.
