# Kernel-level screens on b70-0 (not gates: no correctness check, no end-to-end number)

SDPA screens: `tools/screen_sdpa.sh` (`test_llama_microbench --sdpa`, prefill S = 2048 on a 3072 cache, ms per
layer; `<name>-runs.csv` has every run, `<name>.csv` the medians; `base` = the stock kernels the parent runs).
Linear screens: `tools/screen.sh` (`--linear --regime=prefill --storage=texture3d`, per-layer-weighted kernel
time in us, `_x` = base / token, above 1 is faster; `base` = the shipped Xe2 tile).

| file | build | what |
|---|---|---|
| `screen1-sdpa` | topic1 | 48 profiles, 1 round: the 780M twins and the fragment-layout families at Xe2 shapes |
| `screen2-sdpa` | topic2 | 15 profiles, 2 rounds: `xe2-refine1`, the column-major fragment-layout QK^T, more attn*V tiles |
| `screen3-8da4w` | topic2 | 8da4w tile shapes (zpg and the 780M texel-wise `bt`), 2 rounds |
| `screen4-4w` | topic2 | 4w tile shapes, 2 rounds |
| `screen5-8da4w` | topic3 | 8da4w texel-wise staging with fewer slots than threads (`xe2bt`), 2 rounds |
| `screen6-4w` | topic3 | 4w texel-wise staging (`xe2bx`), 2 rounds |

Mislabelled row: in `screen6-4w.csv` the token `t128x128k16g44s16m8flib` matched the texel-wise kernel
`sarc_dev_linear_q4gsw_coopmat_xe2bx_t128x128k16g44s16m8flib` (tokens are matched by suffix and it was
registered first), not the release-body band-drain twin it was meant to select. That row is a third
measurement of the texel-wise kernel. The band-drain twin is measured in `screen8-4w` with the token
`sweep_t128x128k16g44s16m8flib`.

Negative results kept here on purpose: every alternative tile shape of both linear kernels (screens 3 and 4)
and texel-wise staging on a subset of the threads (screens 5 and 6) are slower than the shipped Xe2 tiles.
