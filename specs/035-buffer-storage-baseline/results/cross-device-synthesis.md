# Cross-Device Buffer-Storage Synthesis

**Date**: 2026-07-22
**Devices covered**: RDNA3 discrete GPU (RX 7900 XTX, complete — `specs/034-rdna3-dgpu-baseline/results/report-buffer.md`), M41 (complete, this spec), S25 Ultra (complete, this spec). **M51 excluded** — buffer sweep handed off to the user mid-run (partial data only: 1B/4w clean, 1B/8da4w crashes deterministically, 3B/4w 2/3 reps); not included below.

## Does 8da4w beat 4w under buffer storage, device by device?

| Device | Prefill | Decode |
|---|---|---|
| RDNA3 dGPU | **Yes**, all 3 sizes (e.g. 1B 9615 vs 4830) | No, 4w wins all 3 sizes (e.g. 1B 296.9 vs 269.9) |
| M41 | **No** — 4w wins every size, by 3–8× (e.g. 1B 624 vs 222) | No, 4w wins every size (e.g. 1B 25.2 vs 24.4) |
| S25 Ultra | Yes where comparable (1B 724 vs 548; 3B/8B 4w is `NR`, crashed) | Mixed — 4w's only valid cell (1B, 27.2) beats 8da4w (1B, 21.9); no 3B/8B 4w decode number exists to compare |

**No longer a universal "yes" on prefill.** Under texture storage, every device in this project's baseline had 8da4w winning prefill. Under buffer storage, **M41 flips this entirely** — its 8da4w prefill regresses so badly (−54% to −73% vs its own texture numbers) that plain 4w wins prefill at every model size, the only device where that happens. Decode keeps the same "4w usually wins" shape as texture, everywhere.

## Buffer vs. texture, by device (aggregate pattern)

| Device | Prefill | Decode | Notable stability change |
|---|---|---|---|
| RDNA3 dGPU | Buffer ≥ texture, 0–7% across all 6 cells, never behind | Buffer ≥ texture, same range | None — 0 crashes either storage type |
| M41 | **Quant-dependent**: 4w +3–17% (buffer wins), 8da4w −54–73% (buffer badly regresses) | Buffer −3 to −21% (texture usually wins), both quants | **New crashes**: 3B/8da4w now crashes under buffer at both floating and pinned clocks (needs `t64`/`t32` node-threshold fix); texture never needed this fix there. 8B/4w floating, conversely, *stopped* crashing under buffer (texture had 1 crash there). |
| S25 Ultra | Buffer ahead where comparable, +5–27% | Mixed, −10% to +2% | Same two deterministic 4w crashes (3B, 8B) persist under buffer, identical signature — buffer does not fix Adreno's incompatibility. No new crashes; 8B/8da4w's late-intermittent crash pattern also persists. |

**Key finding: buffer storage is not a uniform win or loss — its effect is both device- and quant-specific**, in contrast to the RDNA3 dGPU where it was a clean, universal small win. M41's 8da4w prefill regression is the standout anomaly: large (>50%), consistent across model sizes and both clock policies, and accompanied by two new crash cells that required the node-threshold workaround to recover. This looks like a genuine buffer-storage dequantization/dispatch cost specific to 8da4w on this Exynos+driver combination, not a benign artifact — worth a follow-up investigation (out of scope here, same as the RDNA3 dGPU report's own 8da4w-correctness caveat was left as follow-up).

## Crash/stability changes introduced or removed by buffer storage

- **New crashes** (didn't exist under texture): M41 3B/8da4w (floating + pinned, both fixed by node-threshold `t64`/`t32`); M51 1B/8da4w (floating, deterministic, **unresolved** — M51 work handed off before a fix was attempted).
- **Crashes removed** (existed under texture, gone under buffer): M41 8B/4w floating (texture had 1/3 unknown-cause crash; buffer ran clean 3/3).
- **Crashes unchanged**: S25 Ultra's 3B/4w and 8B/4w deterministic crashes, and 8B/8da4w's late-intermittent crash, all persist identically under buffer storage with the same `vkQueueSubmit=-4` signature — buffer storage neither fixes nor worsens Adreno's incompatibility here.
- **Correctness-only changes** (no crash, but decode text quality shifts): RDNA3 dGPU's 8da4w garbled-text-under-texture became `!!!!`-repetition-under-buffer (1B/3B); M41 and S25 Ultra were not separately re-checked for this specific correctness dimension in this pass (out of scope — this baseline's harness validates exit code + stats line only, per the established convention, not text quality beyond the 48-token coherence check).

## What this means for the "does buffer storage help" question

There is no single answer across this project's devices:
- **RDNA3 dGPU**: yes, unambiguously — small, consistent win, zero downside.
- **S25 Ultra**: mostly yes on throughput where cells are valid, no change to the pre-existing crash set.
- **M41**: it depends entirely on quant mode — a clear win for 4w, a severe regression for 8da4w (both in throughput and in newly-introduced crashes).
- **M51**: unknown/incomplete — the one new crash found (1B/8da4w) is a plausible early signal that M41's 8da4w regression pattern might generalize to Exynos devices specifically (both M41 and M51 are Exynos/Xclipse; RDNA3 dGPU and Adreno are not), but this is speculative until M51's sweep is resumed and completed.

## Sources

- `specs/034-rdna3-dgpu-baseline/results/report-buffer.md`, `report.md` (RDNA3 dGPU texture+buffer)
- `specs/035-buffer-storage-baseline/results/report-m41.md` (M41 texture-vs-buffer, this spec)
- `specs/035-buffer-storage-baseline/results/report-s25-ultra.md` (S25 Ultra texture-vs-buffer, this spec)
- `specs/031-release13-4w-crash-survey/results/report.md`, `specs/030-m41-release13-baseline/results/*.md` (texture baselines referenced for comparison)
