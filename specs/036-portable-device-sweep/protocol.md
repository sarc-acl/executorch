# New-device WMMA tile sweep protocol (specs/036)

One page. Run this on any new GPU to retune the coopmat tile defaults for
both quant modes. Everything lives in `specs/036-portable-device-sweep/`;
all commands run from `scripts/` with the repo's venv python (needs
`optuna`; `pip install optuna` once per environment).

Prerequisites: a working `cmake-out-vk` build (backend + `llama_main` +
`test_coopmat_linear_bench` — see BENCHMARKING.md), the target-model buffer
ptes + tokenizer + 2048-token prompt file on disk, and `vulkaninfo` on PATH.

## 0. Fingerprint the device

```bash
python3 device_fingerprint.py
```

Record the printed block. If the device has a known quirk (e.g. Xclipse PAL
crashing on int8 WMMA at subgroup 32), pass `--quirk no_int8_wmma_sg32` to
every `sweep.py` call below. Do NOT inherit another device's quirks or
pinned subgroup sizes — re-testing them is cheap and specs/035 showed they
flip between drivers (s32 was banned on M5, wins on RADV).

## 1. Know your group_size

`--group-size` is the quantization group size baked into the pte being swept
(this box's buffer ptes: 128). A wrong value silently mislabels K-tile
legality. Check the export config, not your memory.

## 2. (Optional) sanity-print the legal universe

```python
import tile_constraints as tc, device_fingerprint as dfp
fp = dfp.fingerprint(); lim = dfp.limits_from_fingerprint(fp)
print(len(tc.enumerate_legal("q4gsw", lim, 128)),
      len(tc.enumerate_legal("dq8ca", lim, 128)))
```

A few hundred per shader is normal. Zero means the fingerprint or group_size
is wrong.

## 3. Microbench prefilter?

Measured on 780M/RADV (specs/035 round-1 data, `results/rank-correlation-780m-*.md`):

- q4gsw: best case rho 0.912, e2e-top5-in-top10 recall 1.00
- dq8ca: rho 0.999, recall 1.00

So on this device the normal-mode bench (~20 s/token) is a trustworthy
*prefilter*: it never dropped a real e2e-top-5 candidate. It is still not a
winner-picker — specs/027 saw the microbench winner lose the e2e final on
M5 — so the last word on the winner is always e2e (which the sweep's
confirm stage provides anyway). On a NEW device, treat the prefilter as
unproven until re-validated: after the first sweep, run
`rank_correlation.py` against that device's data before letting the
microbench rank prune candidates there. Prefiltering pays off mainly where
e2e is expensive (adb-tethered Android, 8B-model ranking, or wide scans of
hundreds of candidates); on this box a 1B e2e measurement costs about the
same as one bench run, so the default sweep skips the prefilter.

## 4. Sweep (per shader; ~1–2 h each at default budget)

```bash
python3 sweep.py --shader q4gsw --group-size 128 --budget 60 --batch-size 16 --early-stop 15
python3 sweep.py --shader dq8ca --group-size 128 --budget 60 --batch-size 16 --early-stop 15
```

What it does for you: seeds every known device's winners + the shipped
default, batches yaml-append + rebuild, correctness-gates and
dispatch-proves every candidate before measuring, interleaves control runs
(re-baselining on >3% drift; `--strict` aborts instead), diffs every run's
output hash against control (temperature-0 miscompute check), blocklists
deterministic failures, and 5-rep-confirms the top 5. Interrupted? Rerun
the same command — the journal resumes it. Non-default pte: `--pte`.

If the summary lists `remeasure_pending` tokens, re-screen them
(`measure.py --tokens ... --stage screen`) before trusting the ranking.

## 5. Validate top-3 on the largest target model

```bash
python3 measure.py --shader q4gsw --stage validate \
  --pte <8B-or-3B buffer pte> --tokens <top3 comma-separated>
```

Rank flips between 1B and 8B happen; the shipped default must win at the
size users run.

## 6. Winner checks before shipping

- Full 44-case gate + a normal-mode bench run for the winner token
  (any FAILED → disqualify, take the next finalist).
- dq8ca only: the tsweep shader is the dbuf2 loop but the shipped default
  shader is dbuf4 — re-verify the winner geometry through the shipped
  shader path (edit its yaml defaults, rebuild, one confirm run), as
  specs/035 did.

## 7. Ship

Edit the two shader yamls' default geometry + `kQ4gswCoopmatDims` /
`kDq8caQ4gswCoopmatDims` in `QuantizedLinear.cpp` (see commit dffce09839
for the shape of that change). Note: defaults are per-branch/device today;
if this toolkit's results end up spanning several devices, dispatch will
need a device check keyed on the fingerprint's `device_name`/`driver_id`.

Commit everything new under `results/` (runs jsonl, journal, summary,
blocklist) plus the tsweep yaml additions the sweep made — see
`contracts/sweep-result-schema.md` for what each file is.
