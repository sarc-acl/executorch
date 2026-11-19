# M51 conv2d-pw-coopmat: thresholds fixed before data

Fixed 2026-11-19 with the device owner, before any M51 measurement. All values are relative.

| Decision | Rule |
|---|---|
| A/A acceptance | geometric mean of the A/A ratios within 1% of 1, and every cell within ±2% |
| Noise band | the larger of 2% and the cell's A/A spread |
| Shape win | the best passing variant is faster than `pw_tiled_4x8` by more than the noise band |
| Campaign success | geometric mean over the timed shape set ≥ 1.15× (`pw_tiled_4x8` time / best variant time) |
| Candidate adoption | passes the correctness gate, and every cell it changes gains beyond the noise band |
| Stop rule | two consecutive gated candidates each gain < 2% geometric mean |
| Correctness gate | 10 of 10 repetitions pass (full grid and reduced grid) |
| Runs per cell | median of 5 valid runs, arms interleaved |
| Shared memory | ≤ 64 KiB per workgroup, never exceeded |
