# Complete kernel-level linear screens (2026-10-07 23:53 to 2026-10-08 01:30 UTC)

`tools/chain4.sh` -> `tools/linear_screen.sh`: `test_llama_microbench --linear --regime=prefill --storage=texture3d` on the
twelve real prefill shapes (M = 2048), 3 rounds, order rotated per round, one `gl.sh` job per (round, kernel), cooled
to 55 C before each job. 4w: the table kernel + 20 candidates (`screen-4w.csv`, 63 jobs); 8da4w: the table kernel + 23
candidates (`screen-8da4w.csv`, 72 jobs). Every row has `dispatched=1` (each kernel ran on every shape). Binary: the
parent build's `test_llama_microbench` (`STAGE.md`), env `ET_VK_SARC_UNVERIFIED=1`. Kernel timings, not a timed session.

Rule (R8): a kernel replaces the table kernel of a shape only if at least 3 % faster in every round (`tools/screen_pick.py`).
- `picks-4w.csv`, `speedups-4w.txt`: 6 of 12 shapes pass (3.2 to 4.0 % in the worst round).
- `picks-8da4w-without-texelwise.csv`, `speedups-8da4w.txt`: all 12 shapes pass with `zpg_t256x64k64g48s32afmb1` (worst round 1.106 to 1.142).
- `picks-8da4w-all-kernels.csv`: the texel-wise family (`zpg_bt_*`) also passes against the table kernel on every shape
  (worst round 1.114 to 1.127 for `bt_t128x64k64g42s32`), but against `afmb1` its best kernel per shape reads 0.977 to
  1.011 in the worst round (ahead on 4 of 12 shapes, by at most 1.1 %): nowhere the 3 % that replacing `afmb1` needs.
Candidate 3 takes the per-shape picks without the texel-wise family; candidate 4 (texel-wise staging) is judged against
those picks (kernel-level result below).
