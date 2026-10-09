# results/rx7600/round2

Round 2 (owner decision 2026-10-08 23:38 UTC): the linear kernels. Placeholders as in `../README.md`.

## Diagnosis of the shipped 8da4w kernel (`sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1`, build `r2a`, commit `70a819b0c`)

- `r2a-screen-8da4w.csv`, `r2a-screen-8da4w-summary.txt`: kernel-level screen, 3 rounds, twelve prefill shapes (`tools/linear_screen.sh`,
  `tools/screen_from_logs.py`: the microbench of that build did not classify the new kernels as coopmat, so the dispatch time of each
  job is read from its log). Ratios are incumbent time / variant time. The `ablN` kernels remove work and write wrong results by design
  (measurement only): bit 1 = A global fetch, 2 = B global fetch, 4 = LDS stores of the next chunk, 8 = the whole MMA loop, 16 = the
  barrier. Sum of the twelve shapes: shipped kernel 35044 us; no A fetch 34011; no B fetch 32947; no stores 32448; no fetch and no stores
  32490; no barrier 33331; no MMA loop 1103 (3 %). The staging (fetch + stores) is therefore about 7 % of the kernel and **the MMA loop
  (LDS fragment loads + WMMA) about 93 %**; the loop alone runs at about 67 % of the 43.9 TOP/s int8 roof (cited, not re-measured).
  `pad4` / `pad8` (A slab padding against LDS bank conflicts of the A stores) change nothing (0.996 / 1.003); the 256 x 128 tile (32 x 32
  per wave) is 0.73.
- `phases/r2a-*.csv`: in-kernel phase timing of the twins (`p` variants; share of a wave's time): shipped tile barrier 52 to 55 %, MMA 27 to
  29 %, fetch 5 to 9 %, LDS stores 7 %; padding does not move any of them.
- `r2a-isa-afmb1.txt`: RADV statistics of the shipped kernel: 64 VGPRs (a 1024-invocation workgroup caps the register budget), 3 spills
  outside the loop, every fragment load split into two `ds_read_b64`.
