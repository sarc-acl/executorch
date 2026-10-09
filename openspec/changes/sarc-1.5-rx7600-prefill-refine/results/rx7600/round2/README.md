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

## Round 2, build `r2b` (micro build of commit `9706804a5`; one screen round, all variants of that commit)

`r2b-screen-8da4w.csv` / `-summary.txt`, `r2b-isa.txt` (`tools/isa_stats.py` on `RADV_DEBUG=shaders` dumps). One round, noise about +-1 %.
Ratios against the shipped kernel (`afmb1`, 35414 us over the twelve shapes in this round):
- `bf` (branch-free chunk loop) 1.014 (one round: inside the noise band, to be repeated); interleaving the stores with the MMAs
  (`bfa1b1` 0.991, `bfa1b2` 1.000, `bfa2b2` 0.997): nothing.
- `uv4` (shared staging arrays typed uvec4, aimed at ds_read_b128 fragment loads): 0.888; with the stores removed (`uv4abl4`) 1.003 against
  1.063 for the uint arrays without stores (`abl4`, `r2a`). RADV still emits two `ds_read_b64` per fragment row with uvec4 arrays
  (`r2b-isa.txt`), so the typing buys nothing and the component-wise 32-bit stores cost: rejected.
- Smaller workgroups (A_BLOCKS / B slots scaled): `g44` 0.926, `g28` 0.942, `g24` 0.904, `g42` 0.877 (all plain; the `uv4` twins 0.80 to
  0.86). They get 128 VGPRs and no spill (the shipped 1024-invocation kernel gets 64 VGPRs and 3 spills outside the loop) and are
  still slower: the register cap is not what limits the shipped kernel.

## Round 2, build `r2d` (micro build of commit in `stage` STAGE.md; two screen rounds)

Ratios against the shipped kernel (`afmb1`, 35401 us over the twelve shapes). Row pitch of the A / B staging in shared memory (uint; 4 = the
shipped 16 bytes) with the drain tile aliased onto the A buffer (`csha`, needed to stay under 64 KiB):
- `pa6csha` (A pitch 24 bytes, B unchanged): **1.055 / 1.050**, every one of the 12 shapes at least 1.03 in both rounds (passes the 3 % rule
  of `proposal.md`); `pa6pad4csha` (plus 4 uint between the K slabs) 1.049 / 1.046: the A stores are not the point.
- `pb6` (B pitch 24 bytes, A unchanged): 0.958 / 0.955; `pa6pb6csha` 0.968 / 0.965. Same instruction mix as the shipped kernel in
  `r2d-isa.txt` (two `ds_read_b64` per fragment, 64 VGPRs): only the LDS addresses differ.
- Fragment reuse (`abl32`: the fragments of K slab 0 serve the MMAs of slabs 1 to 3, a quarter of the LDS fragment loads): 1.056 / 1.053;
  no staging and no barrier (`abl23`) 1.108 / 1.093; reuse and no staging (`abl39`) 1.067 / 1.080; reuse and no barrier (`abl48`) 1.091 /
  1.090; reuse, no staging, no barrier (`abl55`): 28085 us, i.e. the WMMA loop cannot get much past about 78 % of the cited int8 roof in this
  structure (per-shape times in the CSV). The LDS fragment loads are therefore worth at most about 5 % of the kernel, and the staging plus
  barrier about 10 %; the shipped kernel is at about 62 % of the roof.

## Round 2, build `r2e` (micro build; two screen rounds; pitch grid)

`r2e-screen-8da4w.csv` / `-summary.txt`, `r2e-phase-compare.txt`, `phases/r2e-pa6pb4csha.csv`. Pitch in uint (4 = 16 bytes), `csha` = drain tile
in the A buffer:
- `pa6pb4csha` (A 24 bytes, B 16 bytes): 1.052 / 1.048, 11 of 12 shapes at least 1.03 in both rounds (the twelfth passes in one round);
  `pa4pb4csha` (csha alone) 1.000 / 0.997: the aliasing itself is neutral. Phase twins: total cycles 0.878, MMA phase 0.660, barrier 0.907,
  LDS stores 0.985 of the shipped tile's (`r2e-phase-compare.txt`).
- Any B pitch other than 16 bytes is slower (24 bytes 0.96, 32 bytes 0.94); odd pitches (20 / 28 bytes: rows not 8-byte aligned) are 0.26 to
  0.58 (the fragment loads leave the aligned path). `bf` on top of `pa6pb4csha`: 1.031 / 1.027, not additive: dropped.
