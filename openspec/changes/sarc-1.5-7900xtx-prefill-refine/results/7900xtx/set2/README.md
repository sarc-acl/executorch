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
  The single entry in `picks-8da4w.csv` is the existing 128 x 64 tile `sweep_t128x64k32g22s32` on 8B w2: worst-round ratio **1.034 here, which passes the 3 % rule of this screen**; the same pair read 1.026 in the screen of candidate 3 and 1.024 in the synchronisation screen
  (rounds 2870.24/2695.76, 2834.24/2701.06, 2777.66/2687.36 us here). It is not a pitch variant. It is gated as candidate 7 (profile `7900xtx-refine6`); an earlier version of this file called it "noise-level", a judgment made after seeing the data and withdrawn.
- `phases/`: phase timing of the twins (shader clock, share of a wave's cycles, median over the twelve shapes): 4w table kernel: barrier 51.8 %, LDS stores 15.9 %,
  MMA 28.6 %, fetch 1.9 % (the same pattern as 8da4w); 8da4w t256x64k32g24: barrier 31.5 -> 31.6 %, LDS stores 17.4 -> 31.4 % with the 24-byte pitch (barrier + LDS 48.9 -> 63.0 %, up);
  so the barrier + LDS share does not drop, which the owner decision asked to show first. The RDNA3 7900 XTX does not behave like the RX 7600 here (cause not investigated; UNVERIFIED).

## Second screen of item 2 (build `c8`, commit d5d65ac5f, 2026-10-09 09:15 to 09:35 UTC, `tools/q-set2b.sh`)
Branch-free chunk loop (`bf`), the stores of the next chunk interleaved with the MMAs (`bfaXbY`), and the ablation twins on the incumbent tile `t256x64k32g24s32`
(`sync-variants-8da4w.txt`, `screen3-8da4w.csv`): **no synchronisation variant qualifies** (`picks-8da4w-sync.csv`: 0 of 12 shapes; medians 0.90 to 0.97 of the incumbent).
What the ablations bound (median over the twelve shapes, ratio incumbent / twin): without the barrier 1.008, without the stores of the next chunk 1.069, without all staging 1.069,
without staging and barrier 1.101. The barrier of the chunk loop is worth under 1 % of the kernel and the staging at most 7 to 10 %; the barrier share of the phase timing (31 %) is
time a wave waits while the other waves of the workgroup work, not a cost that removing the barrier recovers.

## Item 3 (4w) in the same terms (build `c9`, commit a554a5791, 2026-10-09 09:57 to 10:02 UTC, `tools/q-set2c.sh`)
- Phase timing of the release 4w table kernel (twin `sarc_dev_prof_q4gsw_t256x128k32g24s32f32cbtp`): barrier 51.8 %, LDS stores 15.9 %, MMA 28.6 %, fetch 1.9 % (`phases/s2-q4-table.csv`): the same pattern as 8da4w, so the treatment applies.
- Variants screened (`screen2-4w.csv`, 20 kernels, 3 rounds): the RX 7600's 72-byte A / 88-byte B staging rows (`ap4bp12`) and the other pitch combinations on the table kernel's grid (g24) and on 16 waves (g28):
  **0 of 12 shapes qualify** (worst-round ratios 0.42 to 1.00; the control copy `bp8` 0.986 median). `picks-4w.csv`.
- Ablation twins (`ablation-4w.txt`, `screen3-4w.csv`; measurement only): without the barrier of the chunk loop 1.069 (median over the shapes; up to 1.143), without the stores of the next chunk 1.025,
  without both 1.090. This is a ceiling that cannot be reached (the barrier orders those stores against the MMA fragment loads of the other waves). A K step of 64 (half the barriers) was not tried: the workgroup
  memory the SPIR-V declares for the release table kernel `t256x128k32g24s32f32cbt` is 61,440 bytes (spirv-dis of the c9 build, reviewer's script and mine agree), already about twice the 32,768 bytes the device query reports
  (`max_shared_mem_bytes` of the microbench; AMDVLK `maxComputeSharedMemorySize`), and a K64 tile would need more. (An earlier version of this file said the table kernel uses about 30 KB: that estimate was wrong.)
- Conclusion for items 2 and 3: no variant passes the 3 % rule on any shape, so there is nothing to gate; the ceilings measured are +1 % (8da4w barrier), +7 to +10 % (8da4w staging and barrier together) and +7 % (4w barrier).
