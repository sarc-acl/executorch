# Kernel-level screens on the Arc B580 (not gates: no correctness check, no end-to-end number)

Linear screens: `tools/screen.sh` (`test_llama_microbench --linear --regime=prefill --storage=texture3d`,
per-layer-weighted kernel time in us, `_x` = base / token, above 1 is faster; `base` = the shipped Xe2 tile,
which the B580 shares with the B70). `<name>.csv` has the medians over the rounds, `<name>.txt` every round
per shape. SDPA screens: `tools/screen_sdpa.sh` (ms per layer, S = 2048).

Raw rows: `<name>-rows.csv` (linear: one row per token, round and shape, key `token,rep,model,op,storage,variant`)
and `<name>-runs.csv` (SDPA: one row per profile, round, model, sub-op and variant). The linear row files were
written on 2026-10-05 10:51 UTC by `screen.sh` in its recover-only mode from the cached JSON of the original
runs, with no kernel run again and the cached logs and JSON unchanged (sha256 of all 473 files before and
after); their `rc` and `temp_c` columns are empty because the screens ran before the scripts saved a status
marker per run. Both scripts now resume by saved row keys and never overwrite a result (`tools/screen_rows.py`,
`tools/test_resume.sh`). One thing the earlier scripts did lose: `screen1-sdpa` was first started at 06:15 UTC
and stopped seconds later so that a build could run; the log of that interrupted `base` run was overwritten
when the screen was started again at 07:12 UTC. It was never part of a result.

| file | build | what |
|---|---|---|
| `screen2-8da4w` | topic1 | every 8da4w tile in the dev zone (24, the B70 campaign's), 2 rounds |
| `screen3-4w` | topic1 | every 4w tile in the dev zone (17), 2 rounds; noisy, see below |
| `screen4-8da4w` | topic2 | three more texel-wise 8da4w tiles for this card (`b580bt`), 2 rounds, cooled runs |
| `screen1-sdpa` | topic1 | all 59 `b580-*` SDPA profiles (every SDPA kernel in the dev zone, one at a time), 2 rounds |
| `screen5-4w` | topic3 | quiet rerun of the 4w screen: 5 tiles, 3 rounds, cooled runs |

## screen2-8da4w

Card quiet (the desktop session was idle and locked until about 05:40 UTC, i.e. for round 1 and half of
round 2). `xe2bt_t128x128k64g84s16m8` (balanced K = 64 tile, texel-wise weight staging) is the fastest tile
on every one of the 12 shapes in both rounds: 1.19x to 1.45x per shape, **1.31x** per-layer-weighted geomean
(B70: 1.26x with the same tile). The 128 x 64 K = 64 tiles are second (1.23x; B70 1.12x). No other tile is
3 % faster than it on any shape in both rounds, so one tile serves all shapes, as on the B70. Everything
that lost on the B70 loses here in the same order (0.28x to 0.95x).

## screen3-4w: negative, and a measurement trap

Round 1 of `base` was the first run after the 35-minute 8da4w screen (card at 54 C, no cool-down between
screen runs at that time) and about when the owner started using the desktop: its large shapes are 17 to 21 %
slower than in round 2 (8B `w1_w3` 6561 against 5431 us). Against that slow base the band-drain tiles
(`sweep_t128x128k16g44s16m8flib`, `xe2s_t128x128k16g44s16m8flib`) looked 1.17 to 1.22x faster after round 1.
Against round 2 they are **1.00x** (8B `w1_w3` 5422 / 5425 against 5431 us), which is the B70's result too
(1.004x and 1.002x). No 4w tile is faster than the shipped one in both rounds. Profiles `b580-refine2` and
`b580-refine2x` were added on the strength of round 1 alone and are not candidates; they are kept for an
end-to-end confirmation of the null result. `screen.sh` now cools before every run, and a tile is chosen only
if it wins in every round (the rule fixed in `proposal.md`).

## screen4-8da4w

The three tiles of the texel-wise family that fit the static rules and that the B70 campaign had not built
(`impl/sarc_dev/B580Linear.cpp`): `b580bt_t128x128k32g84s16m8` 1.10x, `b580bt_t64x128k64g84s16m8` 0.80x,
`b580bt_t64x128k64g88s16m8` 0.45x, against 1.30x for `xe2bt_t128x128k64g84s16m8` in the same screen. With
the 46000-byte shared-memory rule nothing larger fits (a 256-row or K = 128 tile of this family needs 52 to
76 kB), so the 8da4w tile space of this family is exhausted on this card.

## screen1-sdpa

Kernel times in us per layer, S = 2048, round 1 / round 2 (the two rounds agree within 0.5 % everywhere):

| QK^T kernel | 8B | 3B | 1B |
|---|---|---|---|
| stock (`base`) | 4697 / 4702 | 3632 / 3636 | 2538 / 2538 |
| `sweep_t128x64k32g44s16m8nf` (base row, `b580-sdpa0`) | 1259 / 1262 | 948 / 948 | 659 / 658 |
| `pk_t128x64k32g44s16m8nf` (candidate 0) | 619 / 618 | 472 / 471 | 423 / 422 |
| `xe2c_t128x64k32g44s16m8nf` | **572 / 571** | **440 / 439** | 416 / 417 |
| `xe2c_t64x128k32g44s16m8nf` | 582 / 581 | 439 / 440 | 411 / 412 |

| attn*V kernel | 8B | 3B | 1B |
|---|---|---|---|
| stock (`base`) | 5730 / 5726 | 4424 / 4410 | 3030 / 3027 |
| `sweep_t64x64k32g44s16m8` (base row; candidate 0 for head_dim 64) | 831 / 834 | 635 / 633 | 496 / 497 |
| `xe2_t128x64k32g44s16m8` (candidate 0 for head_dim 128) | **657 / 658** | **521 / 523** | 493 / 497 (falls back to 64 x 64) |
| `ml_t128x128k32g48s16m8` | 714 / 716 | 564 / 558 | 488 / 490 |
| `xe2_t64x64k32g44s16m8` | 808 / 807 | 622 / 625 | 489 / 490 |

- QK^T: the column-major fragment-layout kernel `xe2c_t128x64k32g44s16m8nf` is 7.6 % (8B) and 6.8 % (3B)
  faster than candidate 0's `pk` kernel in both rounds, 1.4 % on 1B; the best 1B kernel
  (`xe2c_t64x128k32g44s16m8nf`) is 2.7 % faster than `pk`, under the 3 % rule, so head_dim 64 keeps `pk`.
  B70: the same kernel, 7 % (8B) and 6.5 % (3B), 0 % on 1B.
- attn*V: `xe2_t128x64k32g44s16m8` is the best head_dim-128 kernel (as on the B70); for head_dim 64 nothing
  is 3 % faster than the 64 x 64 tile.
- The truncated softmax is 1074 (8B), 810 (3B), 1075 (1B) us in every profile: larger than QK^T or attn*V.
- Every ranking is the B70's ranking. Ratio of B580 to B70 kernel time: 1.5 for QK^T, 1.37 for attn*V, 1.35
  for the softmax.

## screen5-4w: the 4w screen again on a quiet card

Three rounds with a cool-down before every run, repeat spread at most 0.4 %. Band drain on the shipped
geometry 1.000x (`sweep_...flib`) and 1.000x (`xe2s_...flib`); `sweep_t128x64k16g24s16m8fli` 0.91x,
`sweep_t64x128k16g42s16m8fli` 0.87x, `xe2s_t128x256k16g84s16m8flib` 0.92x (B70: 1.004x, 1.002x, 0.90x, 0.86x,
0.95x). The shipped 4w tile stays, as on the B70. This replaces the noisy `screen3-4w` as the 4w evidence;
that one is kept for its other 11 tiles (0.15x to 0.76x) and for the record of the trap.
