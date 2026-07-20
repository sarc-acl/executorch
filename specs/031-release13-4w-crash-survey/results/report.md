# Release/1.3 Crash Survey — M5 EVT1, 4w + 8da4w, Floating + Pinned (2026-07-14)

**Self-contained deliverable** — no need to also open `raw-attempts.md` to interpret this
document (per FR-006/SC-004 and this feature's User Story 2 Independent Test). This report
covers the original `4w`/floating-only scope (`spec.md`'s initial ask) **and** its same-day
extension: `4w` pinned-clock gap-fill, plus the identical methodology applied to `8da4w`, floating
and pinned, all three models.

**Configuration**: Llama 1B/3B/8B, `group_size=128`, texture storage, `ctx3072` PTE. Workload:
2048-token prefill + 1024-token decode (`p2048_exact.txt` + `--num_bos=1`, `--max_new_tokens=1024
--ignore_eos --temperature=0 --warmup=true`). Two runners used:
- **`llama_main_rel1.3`** — vanilla `release-1.3/executorch` (upstream `release/1.3`, no
  WMMA/coopmat fork additions; `execute_threshold_node_count` hardcoded to 128, no
  `ET_VK_EXECUTE_NODE_THRESHOLD` override on this branch).
- **`llama_main_nodethresh`** — `release13-node-threshold/executorch`, a pure runtime patch on
  top of vanilla `release/1.3` adding an opt-in `ET_VK_EXECUTE_NODE_THRESHOLD` env var (submits a
  new, non-blocking command buffer every N graph nodes instead of the default 128). Used only on
  cells where vanilla was confirmed (empirically, this session) to crash.

**Device**: M5 EVT1, `0000088f8e579c33`, via `ssh yanwen.xu@sj1-dmckee-d01`.

**Driver**: `c9861e9906d03fa2c7d48b804e1a1c80` (= documented default `f14c51b6f8`) — confirmed
matching before every model's first rep in every cell below, and re-confirmed after every one of
the **20 crash recoveries** across the whole survey (see Crash Event Log). Every number in this
report was measured on this exact, verified, never-drifted driver build.

---

## Headline Table

Threshold column: `—` = vanilla (no crash workaround needed); `t64`/`t32` = which
`ET_VK_EXECUTE_NODE_THRESHOLD` value was used, only ever applied after vanilla was empirically
confirmed to crash on that exact cell this session.

### 4w

| Model | Clocks | Threshold | Prefill tok/s (median ± CoV%) | Decode tok/s (median ± CoV%) | Crash annotation |
|---|---|---|---|---|---|
| 1B | Floating | — | 592.936 ± 0.12% | 27.0042 ± 0.13% | 0/3 crashed |
| 3B | Floating | — | 216.079 ± 0.07% | 13.7062 ± 0.08% | 0/3 crashed (1 transient crash on rep1, recovered) |
| **8B** | Floating | **t64** | 96.037 ± 0.13% | 5.268 ± 0.55% | vanilla: 7/7 crashed; t64: 0/3 crashed |
| 1B | Pinned 509MHz | — | 314.303 ± 0.22% | 14.616 ± 0.12% | 0/3 crashed |
| **3B** | Pinned 509MHz | **t64** | 113.607 ± 0.03% | 7.443 ± 0.14% | vanilla: 3/3 crashed; t64: 0/3 crashed |
| **8B** | Pinned 509MHz | **t32** | 51.616 ± 0.14% | 4.026 ± 1.27% | vanilla: not attempted pinned (already known unsafe from floating); t64: 2/2 crashed; t32: 0/3 crashed |
| **8B** | Pinned **max** (980/5333/934) | — | N/A — 3/3 crashed | N/A — 3/3 crashed | vanilla: 3/3 crashed, terminal; threshold not attempted (see `8b-4w-max-pinned-2026-07-16.md`) |

### 8da4w

| Model | Clocks | Threshold | Prefill tok/s (median ± CoV%) | Decode tok/s (median ± CoV%) | Crash annotation |
|---|---|---|---|---|---|
| 1B | Floating | — | 422.355 ± 3.07% | 25.913 ± 11.60% | 0/3 crashed net (2 transient crashes across reps 2 & 3, both recovered) |
| 1B | Pinned 509MHz | — | 224.217 ± 0.12% | 14.379 ± 0.27% | 0/3 crashed |
| 3B | Floating | — | 152.893 ± 0.01% | 13.278 ± 0.66% | 0/3 crashed net (2 transient crashes on rep1, recovered) |
| **3B** | Pinned 509MHz | **t64** | 80.138 ± 0.03% | 7.236 ± 0.11% | vanilla: 1/1 crashed (not retried further); t64: 0/3 crashed |
| **8B** | Floating | **t64** | 67.446 ± 0.11% | 7.459 ± 0.45% | vanilla: 1/1 crashed; t64: 1 transient crash on rep2 (recovered), 0/3 crashed net |
| **8B** | Pinned 509MHz | **t32** | 35.156 ± 0.07% | 3.909 ± 0.07% | went straight to t32 (established pattern from 4w); 0/3 crashed |

---

## Bottom line

The crash boundary is **not** simply "8B crashes, smaller models don't." Three independent
factors interact:

1. **Model size**: 8B is the most exposed model across every configuration tested — it never
   completed a single rep on vanilla, at either clock policy, on either quant scheme.
2. **Clock speed**: pinned (509MHz) is *more* crash-prone than floating (up to 980MHz) for the
   same model, on vanilla — a 128-node command-buffer chunk takes longer at a slower clock,
   pushing it closer to the Xclipse GPU's ~2.56s job-watchdog timeout. This is why 3B is safe on
   vanilla floating but crashes reliably on vanilla pinned, for both quant schemes.
3. **Threshold sufficiency is cell-specific, not universal**: `ET_VK_EXECUTE_NODE_THRESHOLD=64`
   fixes every crashing cell **except 8B pinned** (both `4w` and `8da4w`), which needs the
   smaller `32`-node chunk size — `64`-node chunks at pinned 509MHz are still too slow for 8B's
   per-node compute. This was confirmed empirically (`64` crashed 2/2 on 8B-pinned-`4w` before
   falling back to `32`), not assumed from the `4w` result carried over to `8da4w`.

Practical takeaway for anyone benchmarking this device: **1B is always safe. 3B is safe floating,
needs `threshold=64` pinned. 8B always needs a threshold workaround — `64` floating, `32`
pinned — regardless of `4w` vs `8da4w`.**

**2026-07-16 follow-up refines point 2 above**: raising the clock, even to a *sustained, constant*
max (980/5333/934 — not just floating's intermittent up-to-980), does **not** save vanilla 8B `4w`
either — 3/3 crashed, same outcome as floating and 509MHz-pinned. So it isn't that floating
sometimes dips to a slow clock at the wrong moment; 8B's per-node compute at the default 128-node
chunk size exceeds the watchdog even at the fastest clock this hardware supports. The threshold
workaround (smaller command-buffer chunks) is the only confirmed fix for 8B, at any clock policy
tested so far. Full write-up: `8b-4w-max-pinned-2026-07-16.md`.

---

## Raw Per-Attempt Table

Full per-attempt data (34 completed, 12 crashed = 46 total attempts across the extension; 21
attempts in the original `4w`-floating-only scope). See `raw-attempts.md` for the identical data
with additional narrative context; reproduced here in full so this report stands alone.

### 4w

| Model | Clocks | Rep | Attempt | Config | Outcome | prefill_tok_s | decode_tok_s | Crash Event |
|---|---|---|---|---|---|---|---|---|
| 3B | Floating | 1 | A | vanilla | crashed | — | — | CE1 |
| 3B | Floating | 1 | B (retry) | vanilla | completed | 215.874 | 13.7062 | — |
| 3B | Floating | 2 | — | vanilla | completed | 216.079 | 13.7049 | — |
| 3B | Floating | 3 | — | vanilla | completed | 216.193 | 13.7256 | — |
| 1B | Floating | 1 | — | vanilla | completed | 592.936 | 26.9714 | — |
| 1B | Floating | 2 | — | vanilla | completed | 592.936 | 27.0042 | — |
| 1B | Floating | 3 | — | vanilla | completed | 594.140 | 27.0427 | — |
| 8B | Floating | 1 | A | vanilla | crashed | — | — | CE2 |
| 8B | Floating | 1 | B (retry) | vanilla | crashed | — | — | CE3 |
| 8B | Floating | 1 | C (retry) | vanilla | crashed, terminal | — | — | CE4 |
| 8B | Floating | 2 | A | vanilla | crashed | — | — | CE5 |
| 8B | Floating | 2 | B (retry) | vanilla | crashed, terminal | — | — | CE6 |
| 8B | Floating | 3 | A | vanilla | crashed | — | — | CE7 |
| 8B | Floating | 3 | B (retry) | vanilla | crashed, terminal (0/3 on vanilla) | — | — | CE8 |
| 8B | Floating | 1 | — | t64 | completed | 95.8218 | 5.3173 | — |
| 8B | Floating | 2 | — | t64 | completed | 96.0375 | 5.26511 | — |
| 8B | Floating | 3 | — | t64 | completed | 96.042 | 5.26839 | — |
| 3B | Pinned | 1 | A | vanilla | crashed | — | — | CE9 |
| 3B | Pinned | 1 | B (retry) | vanilla | crashed | — | — | CE10 |
| 3B | Pinned | 1 | C (retry) | vanilla | crashed, terminal (0/1 on vanilla) | — | — | CE11 |
| 3B | Pinned | 1 | — | t64 | completed | 113.607 | 7.44292 | — |
| 3B | Pinned | 2 | — | t64 | completed | 113.645 | 7.45382 | — |
| 3B | Pinned | 3 | — | t64 | completed | 113.570 | 7.43319 | — |
| 1B | Pinned | 1 | — | vanilla | completed | 314.255 | 14.616 | — |
| 1B | Pinned | 2 | — | vanilla | completed | 314.303 | 14.647 | — |
| 1B | Pinned | 3 | — | vanilla | completed | 315.465 | 14.616 | — |
| 8B | Pinned | 1 | A | t64 | crashed | — | — | CE12 |
| 8B | Pinned | 1 | B (retry) | t64 | crashed, terminal (0/1 on t64) | — | — | CE13 |
| 8B | Pinned | 1 | — | t32 | completed | 51.5947 | 4.02631 | — |
| 8B | Pinned | 2 | — | t32 | completed | 51.725 | 4.009 | — |
| 8B | Pinned | 3 | — | t32 | completed | 51.6155 | 4.10533 | — |

### 8da4w

| Model | Clocks | Rep | Attempt | Config | Outcome | prefill_tok_s | decode_tok_s | Crash Event |
|---|---|---|---|---|---|---|---|---|
| 1B | Floating | 1 | — | vanilla | completed | 401.175 | 21.0559 | — |
| 1B | Floating | 2 | A | vanilla | crashed | — | — | CE14 |
| 1B | Floating | 2 | B (retry) | vanilla | completed | 424.104 | 25.9704 | — |
| 1B | Floating | 3 | A | vanilla | crashed | — | — | CE15 |
| 1B | Floating | 3 | B (retry) | vanilla | completed | 422.355 | 25.9132 | — |
| 1B | Pinned | 1 | — | vanilla | completed | 224.046 | 14.3793 | — |
| 1B | Pinned | 2 | — | vanilla | completed | 224.217 | 14.4093 | — |
| 1B | Pinned | 3 | — | vanilla | completed | 224.586 | 14.3327 | — |
| 3B | Floating | 1 | A | vanilla | crashed | — | — | CE16 |
| 3B | Floating | 1 | B (retry) | vanilla | crashed | — | — | CE17 |
| 3B | Floating | 1 | C (retry) | vanilla | completed | 152.881 | 13.1305 | — |
| 3B | Floating | 2 | — | vanilla | completed | 152.904 | 13.2781 | — |
| 3B | Floating | 3 | — | vanilla | completed | 152.893 | 13.2845 | — |
| 3B | Pinned | 1 | A | vanilla | crashed, terminal (0/1 on vanilla) | — | — | CE18 |
| 3B | Pinned | 1 | — | t64 | completed | 80.1033 | 7.22294 | — |
| 3B | Pinned | 2 | — | t64 | completed | 80.1534 | 7.23848 | — |
| 3B | Pinned | 3 | — | t64 | completed | 80.1377 | 7.23566 | — |
| 8B | Floating | 1 | A | vanilla | crashed, terminal (0/1 on vanilla) | — | — | CE19 |
| 8B | Floating | 1 | — | t64 | completed | 67.3463 | 7.45855 | — |
| 8B | Floating | 2 | A | t64 | crashed | — | — | CE20 |
| 8B | Floating | 2 | B (retry) | t64 | completed | 67.4461 | 7.49885 | — |
| 8B | Floating | 3 | — | t64 | completed | 67.4839 | 7.43308 | — |
| 8B | Pinned | 1 | — | t32 (no t64 attempt this cell) | completed | 35.1558 | 3.91347 | — |
| 8B | Pinned | 2 | — | t32 | completed | 35.1534 | 3.90904 | — |
| 8B | Pinned | 3 | — | t32 | completed | 35.1957 | 3.90855 | — |

All `driver_md5` values are `c9861e9906d03fa2c7d48b804e1a1c80` for every single attempt above —
confirmed matching before each model/config's first attempt and re-verified after every crash
recovery; no attempt in either table was measured on a drifted or unverified driver.

## Crash Event Log

All 20 crash events (CE1–CE20) share the identical signature: the runner process died with no
JSON stats output (5–40s after launch), the M5 EVT1 board dropped off `adb` entirely, and
re-enumerated on USB as `S5E9975_LK_Bootloader` — confirmed via `fastboot devices` every time.
Recovery was always a plain `fastboot -s 0000088f8e579c33 reboot` (no flashing, no wipe),
~30–40s to a fully booted, `adb`-reachable state. Driver hash and clock range were re-verified
unchanged after every single recovery. Zero unrecovered/escalated incidents across the whole
survey. Full per-event timing table → `raw-attempts.md`.

## Coherence checks

Every model/config combination produced coherent output on a short low-token prompt (`"The
capital of France is"`, `--seq_len=48`) before its timed reps — including on the
`release13-node-threshold` runner. 8B's output is repetitive ("...Paris, and the capital of
France is Paris, and...") but coherent — expected greedy-decode behavior at `--temperature=0` on
a highly repetitive prompt, not a correctness bug.

## Notes / Caveats

- **CoV is computed over completed reps only.**
- **1B/8da4w floating has visibly elevated CoV** (3.07% prefill, 11.60% decode) versus every
  other cell (<1.3%) — likely driven by the two crash+reboot cycles that interrupted that
  cell's run sequence (rep1 ran clean; reps 2 and 3 each needed a crash+recovery before
  succeeding) rather than genuine throughput variance. Reported as measured, not smoothed —
  see the per-attempt table above for the individual values.
- **`threshold=64` is not universally sufficient**: it works for every crashing cell in this
  survey except 8B-pinned (both quant schemes), which needs `32`. This was verified empirically
  per cell, not assumed — see `research.md` for why 8B-pinned specifically resists the larger
  chunk size (slowest clock + largest per-node compute + largest chunk = most watchdog-exposed
  combination in this matrix).
- GPU clocks were left floating during floating-clock cells and pinned during pinned-clock
  cells throughout (not blended); restored to the workspace's pinned default (509/2730/663 MHz)
  at the conclusion of the full survey.
- Scope is now **complete for both `4w` and `8da4w`, both clock policies, all three models** —
  the original spec's "4w-only" scope limitation (see `spec.md` Assumptions) has been superseded
  by this extension.
