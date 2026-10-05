# sarc-1.5-b580-prefill-refine: status

**2026-10-05 03:55 UTC — running. Parent control, baseline and A/A are done; candidate 0 (`b580-refine0`,
the B70's SDPA kernels) is in its gate.**

Branch `topic/b580-prefill-refine`, parent `6a7cc8cc6` (no profile). Host `fedora` (the owner's desktop), Arc
B580 = PCI `0000:03:00.0`, Vulkan device 0, `ETVK_DEVICE_INDEX=0`, lock
`86800be2-0000-0000-0300-000000000000`, ANV Mesa 26.2.3. Nothing was run on the Ryzen iGPU (device 1).

## Running now

Detached chains, one GPU job at a time, status files under `.artifacts/logs/`. **If the machine reboots these
are gone**; restart from the first step whose status line is missing.

1. `chain2.sh` (`chain2.status`): probe build, `gate_sdpa.sh s2-c0` (parent against build `topic1` with
   `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine0`), `sdpa_ref.sh` (error against the fp32
   reference, both arms), `probe.sh s2-c0` (logits), `decide.py --arithmetic`, `decode_ab.sh`.
2. `after2.sh` waits for it and starts `chain3.sh` (`chain3.status`): SDPA kernel screen of all 59 `b580-*`
   profiles (2 rounds), 8da4w and 4w linear tile screens (24 and 17 tiles, 2 rounds), phase timing of the two
   shipped linear tiles, igpu-roofline `fast`.

## How noisy the card was

The desktop session was idle and locked during everything so far (`loginctl`: `IdleHint=yes`,
`LockedHint=yes`). Nine processes hold the card (gnome-shell, Xwayland, firefox, ghostty, ...; listed in every
session's `env.txt`); their share of engine time inside the timed prefill windows was 0.00 % in all 84 runs of
`s1-aa` (0.3 to 0.5 % in a smoke run made while the display was on). The "about 20 % from gnome-shell" of the
campaign notes was not observed tonight. Calibration from `s1-aa`: `BUSYMAX` = 5 % (the floor of the rule in
`proposal.md`), `CLKMIN` = 2699 MHz (97 % of the lowest per-run median, 2783 MHz in the 8B 4w cell), idle
package temperature 47 C. A/A spread asks for 5 repeats, not 7 (rule in `proposal.md`: largest A/A cell
deviation 0.08 %, largest arm spread 0.43 %).

## Parent control and baseline (re-measured here)

Builds: `build/parent` = `6a7cc8cc6`, copied by rsync from `fedora-gpu-eval` (the B70 campaign's build, same OS
and driver); the three binaries have the sha256 recorded there and its 53 shipped SPIR-V variants match
`sarc/golden/spirv.json` on this copy (`results/b580/parent.copy.txt`). `build/topic1` = `98e3eccf4`, built here
in `localhost/et-vk-build:rocky10`, golden unchanged (53 variants).

Parent control `s0-parent-verify` (unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff`, no
environment): `CONTROL_RECORDED`. 12 of 12 production-diff cases ALL PASSED, 28 of 28 numeric and 4 of 4
rank-3 correctness cases PASSED, decode 31 tokens, default vs tiled SAME on the real-text prompt (both
schemes) and on the unaligned prompt (4w). The device-status lines of pristine `dev/1.5` are the same as on
the B70 and are compared line by line for every candidate: `correctness rc=1`, `linear <scheme> rc=1`,
`1b 8da4w unaligned: default vs tiled output DIFFER`, SDPA tiers on the stock kernels with 0 mismatches.

Baseline + A/A, session `s1-aa` (pristine parent against build `topic1` with no environment, arms interleaved,
median of 5 valid runs, tok/s; `results/b580/sessions/s1-aa/`):

| cell | parent | topic, no env | A/A | expected (`cells.csv`) | parent vs expected |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 8641.35 | 8641.35 | 0.00 % | 8605.04 | +0.42 % |
| 1B 8da4w | 8865.80 | 8865.80 | 0.00 % | 8789.70 | +0.87 % |
| 3B 4w | 3436.24 | 3436.24 | 0.00 % | 3419.03 | +0.50 % |
| 3B 8da4w | 3524.96 | 3524.96 | 0.00 % | 3506.85 | +0.52 % |
| 8B 4w | 1725.36 | 1726.81 | +0.08 % | 1693.96 | +1.85 % |
| 8B 8da4w | 1845.05 | 1845.05 | 0.00 % | 1680.07 | **+9.82 %** |

A/A geomean +0.01 %; 60 timed runs, none rejected; next token parent vs topic SAME in all six cells on
`prompt_2048.txt`, `prompt_check.txt` and `r1304.txt`. The timer resolution is 1 ms = 0.43 % of a 1B prefill
(232 to 237 ms), which is why many readings are identical.

**8B 8da4w is 9.8 % above the expected value, outside the 5 % agreement fixed in `proposal.md`.** Cause, from
the old run table (`sarc-1.5-e2e-benchmark/results/runs_all.csv`): its five SARC runs of that cell were
1809, 1798, 1664, 1613 and 1680 tok/s, two fast and three slow, a 12 % spread, on a desktop that was in use
(the report's "Arc B580 variance" section says so); the published 1680 is the median of the slow group.
Tonight's ten runs of that cell lie between 1840 and 1847 (spread 0.3 %) with 0.00 % foreign engine time,
2 % above the old fast group, like the other five cells (+0.4 to +1.9 %). So the old median was depressed
by desktop activity, not a different kernel or configuration: the dispatched kernels in the parent control
are the shipped rows. Every comparison below is against the parent re-measured in the same session, and the
"original `dev/1.5` numbers" are quoted with this caveat.

## Per-cell numbers against the parent

Candidate 0 is being gated; nothing to report yet.

## Next

1. Candidate 0 decision (reference-error rule).
2. Screens on this card, then B580-specific candidates per shape.

## Blocking

Nothing.
