# Second set, items 2 and 3 (8da4w and 4w linear kernels, staging row pitch), screen of 2026-10-09 08:07 to 08:50 UTC

Owner decision 2026-10-08 23:38 UTC and coordinator note 2026-10-09 04:26 UTC. The RX 7600's round-2 recipe (`610ed0e98`: 8da4w A staging row pitch of 24 bytes; `d6d67ba78`: 4w
72-byte A / 88-byte B staging rows) is ported as dev-zone variants under the `7900xtx` prefix, default off: copies of the release 8da4w zpg body and 4w body with options
(`glsl/sarc_dev/sarc_dev_7900xtx_{dq8ca_zpg,q4}_body.glslh`, reproducible from the release bodies by `tools/gen_7900xtx_{dq,q4}_body.py`; yaml and row files by
`tools/gen_7900xtx_{dq,q4}.py`). Build `c7` (commit 064a07fc3), microbench `s2-screen`, `tools/q-set2.sh`.

- `screen2-8da4w.csv`, `screen2-4w.csv`: kernel-level screens, 3 rounds, twelve prefill shapes, texture3d (`tools/linear_screen.sh`), 25 and 20 kernels, 0 undispatched rows.
- `picks-*.csv`, `screen2-*-ratios.txt` (`tools/screen_pick2.py`): ratio = incumbent time / variant time in the same round, the incumbent being the kernel the profile
  `7900xtx-refine5` uses on that shape (not the release table row). Rule (R8): 3 % faster in every round.
- Result: **no pitch variant qualifies on any shape**: padding the A staging row to 20 / 24 / 32 bytes makes the incumbent tiles 0.55 to 0.9x (the K step 32 tiles
  with pitch 6: 0.75x for t256x64k32g24), the 4w 72 / 88-byte rows 0.52 to 0.99x; the controls (pitch 4 / bp 8: the release staging) reproduce the incumbents (0.98 to 1.00).
  The single entry in `picks-8da4w.csv` is the existing 128 x 64 tile on 8B w2 (1.034 here, 1.026 in the first screen of candidate 3: noise-level, not a pitch variant).
- `phases/`: phase timing of the twins (shader clock, share of a wave's cycles, median over the twelve shapes): 4w table kernel: barrier 51.8 %, LDS stores 15.9 %,
  MMA 28.6 %, fetch 1.9 % (the same pattern as 8da4w); 8da4w t256x64k32g24: barrier 31.5 -> 31.6 %, LDS stores 17.4 -> 31.4 % with the 24-byte pitch (barrier + LDS 48.9 -> 63.0 %, up);
  so the barrier + LDS share does not drop, which the owner decision asked to show first. The RDNA3 7900 XTX does not behave like the RX 7600 here (cause not investigated; UNVERIFIED).
