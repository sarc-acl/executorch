# 8B `4w` Vanilla, Max-Pinned Clocks — M5 EVT1 (2026-07-16)

**Self-contained deliverable.** Extends `results/report.md`'s 4w table with a third clock policy
for the 8B cell: **max-pinned** (GPU 980000 / MIF 5333000 / INT 934000, all sustained constant via
`pin_freqs.sh` overrides — not floating, not the workspace's 509/2730/663 default). Tests
`report.md`'s own "Bottom line" hypothesis #2 (slower clocks push the 128-node vanilla command
buffer closer to the ~2.56s watchdog) by removing clock variance entirely: if that hypothesis is
the whole story, sustained max clock should be at least as safe as floating (which reaches up to
980MHz intermittently) — floating already crashed vanilla 8B 7/7 (`report.md`'s 4w table), so this
tests whether *guaranteed* max, not just occasionally-reached max, changes the outcome.

**Headline finding: it does not.** Vanilla `release/1.3` 8B `4w` (`llama_main_rel1.3`, no
`ET_VK_EXECUTE_NODE_THRESHOLD`) crashed **3/3** attempts at sustained max-pinned clocks, the same
outcome as at floating and at 509MHz pinned. This refines (does not just confirm) the report's
clock-speed hypothesis: raising the clock, even to a constant maximum, is not sufficient to avoid
the watchdog for 8B's per-node compute at the default 128-node command-buffer chunk size. The
`ET_VK_EXECUTE_NODE_THRESHOLD` workaround (smaller chunks) remains the only confirmed fix for this
model, consistent with `report.md`'s existing bottom line ("8B always needs a threshold
workaround").

**Configuration**: Llama 3.1 8B, `4w`, `group_size=128`, texture storage, `ctx3072` PTE. Workload:
2048-token prefill + 1024-token decode (`p2048_exact.txt` + `--num_bos=1`,
`--max_new_tokens=1024 --ignore_eos --temperature=0 --warmup=true`). Runner:
**`llama_main_rel1.3`** (vanilla `release-1.3/executorch`, no WMMA/coopmat fork additions,
`execute_threshold_node_count` hardcoded to 128 — the same binary used throughout
`results/report.md`).

**Device**: M5 EVT1, `0000088f8e579c33`, via `ssh yanwen.xu@sj1-dmckee-d01`.

**Driver**: `c9861e9906d03fa2c7d48b804e1a1c80` (= documented default `f14c51b6f8`) — confirmed
matching before the first attempt and re-confirmed after every one of the 3 crash recoveries below.

**Clocks**: pinned via `pin_freqs.sh` with `GPUFREQ=980000 MIFFREQ=5333000 INTFREQ=934000`
(the hardware's confirmed max on all three domains — see `../../hardware/README.md` for the
probed OPP tables). Verified via sysfs bounds (`min_freq`/`max_freq` on
`23400000.sgpu`, `scaling_devfreq_min`/`scaling_devfreq_max` on `17000010.devfreq_mif` /
`17000020.devfreq_int`), not `cur_freq` — re-verified after each of the 3 reboot recoveries
(clock pins and `adb root` do not survive a reboot; both were re-applied and re-confirmed before
each retry).

---

## Coherence check

`--prompt='The capital of France is' --seq_len=48 --temperature=0 --warmup=false` at max-pinned
clocks: `"The capital of France is Paris, and the capital of France is Paris, ..."` — coherent,
`prefill_token_per_sec: 14.0056`, `decode_token_per_sec: 14.3639`. Passed before the timed attempts
below.

## Attempts

| Rep | Attempt | Config | Outcome | prefill_tok_s | decode_tok_s | Crash Event |
|---|---|---|---|---|---|---|
| 1 | A | vanilla, max-pinned | crashed | — | — | CE21 |
| 1 | B (retry) | vanilla, max-pinned | crashed | — | — | CE22 |
| 1 | C (retry) | vanilla, max-pinned | crashed, terminal (0/1 on vanilla max-pinned) | — | — | CE23 |

Stopped at 3/3 crashed on rep 1 (matches `report.md`'s own precedent for calling an 8B vanilla cell
terminal — see e.g. CE2–CE4/CE5–CE6/CE7–CE8 for 8B floating). No further reps attempted; the
`ET_VK_EXECUTE_NODE_THRESHOLD` fallback was intentionally **not** tried here — the point of this
cell was specifically to test whether max-pinned clocks alone (i.e. without the threshold
workaround) change the outcome, and it's now answered: no.

## Crash Event Log (CE21–CE23)

All three share the same signature as every prior crash in this spec: the runner process exited
with code 0 but produced no JSON stats line and no coherent output; the board dropped off `adb`
within the run and re-enumerated on USB as `S5E9975_LK_Bootloader` (confirmed via `fastboot
devices`, appearing ~15–30s after the crashed attempt). Recovery was a plain `fastboot -s
0000088f8e579c33 reboot` each time (no flashing, no wipe), followed by:
1. `adb -s $S wait-for-device` + boot-completed check (~10-15s to fully booted/adb-reachable).
2. Driver md5 re-verify — `c9861e9906d03fa2c7d48b804e1a1c80` every time, no drift.
3. Re-root + re-run `pin_freqs.sh` with the max overrides (clock pins don't survive reboot) +
   sysfs re-verify — `980000`/`980000` (GPU), `5333000` (MIF), `934000` (INT) every time.

Zero unrecovered/escalated incidents. Clocks were restored to the workspace's pinned default
(509/2730/663) after this cell concluded, per this workstream's standing convention.

## Notes / Caveats

- Only the 8B `4w` cell was tested at max-pinned clocks — this was a targeted follow-up to the
  existing floating/509-pinned matrix in `report.md`, not a full re-run of all 12 (model × quant ×
  clock) cells at a fourth clock policy. 1B/3B were not re-tested here since they were already
  confirmed safe (or safe-with-threshold=64) at both floating and 509-pinned in the original
  survey, and the open question this cell answers is 8B-specific.
- `pin_freqs.sh`'s legacy `/sys/kernel/gpu/{min,max}_freq` write targets return `Permission
  denied`/`No such file` on this kernel build — harmless, the effective pin happens via the
  `/sys/class/devfreq/23400000.sgpu/{min,max}_freq` writes in the same script, confirmed via sysfs
  readback every time in this session.
