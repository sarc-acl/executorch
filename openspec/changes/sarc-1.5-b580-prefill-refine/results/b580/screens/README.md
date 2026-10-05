# Kernel-level screens on the Arc B580 (not gates: no correctness check, no end-to-end number)

Linear screens: `tools/screen.sh` (`test_llama_microbench --linear --regime=prefill --storage=texture3d`,
per-layer-weighted kernel time in us, `_x` = base / token, above 1 is faster; `base` = the shipped Xe2 tile,
which the B580 shares with the B70). `<name>.csv` has the medians over the rounds, `<name>.txt` every round
per shape. SDPA screens: `tools/screen_sdpa.sh` (ms per layer, S = 2048).

| file | build | what |
|---|---|---|
| `screen2-8da4w` | topic1 | every 8da4w tile in the dev zone (24, the B70 campaign's), 2 rounds |
| `screen3-4w` | topic1 | every 4w tile in the dev zone (17), 2 rounds; noisy, see below |

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
