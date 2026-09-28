# Figure captions: end-to-end prefill, Stock ExecuTorch 1.5 vs SARC 1.5-r2

All figures: end-to-end prefill throughput of a 2048-token prompt, measured with
ExecuTorch's Vulkan backend (`llama_main`) on five GPUs, three Llama models and two
quantization schemes (4w: int4 weights with fp16 activations; 8da4w: int8 dynamic
activations with int4 weights). Each build ran n = 5 interleaved timed repeats per
cell. Data: `../cells.csv` (per-cell statistics) and `../runs_all.csv` (only rows
whose `log` starts with `logs/prefill` are timed runs). Regenerate with `./make_all.sh`.

## Figure 1 (`fig1_speedup.pdf`)

**SARC prefill speedup over stock ExecuTorch 1.5.** Left: 4w; right: 8da4w (shared
y-axis). Bars show the speedup, defined as the median SARC throughput divided by
the median stock throughput over n = 5 runs each, for Llama 3.2 1B, Llama 3.2 3B
and Llama 3.1 8B on each GPU. The number above each bar gives the speedup. Error
bars are an approximate 95 % paired bootstrap confidence interval of the ratio of
medians: the two builds ran back to back within each repeat, so repeats are
resampled as pairs. With n = 5 the interval is coarse. The
dashed line marks 1.0× (parity with stock). The speedup is between 1.48× and 5.35×
in every cell. It increases with model size on four of the five GPUs (on the 780M, 8B is slightly
below 3B), and it is larger for
4w than for 8da4w on the AMD and Intel GPUs.

## Figure 2 (`fig2_throughput.pdf`)

**Absolute prefill throughput, stock vs SARC.** Rows: quantization scheme (4w,
8da4w). Columns: model (Llama 3.2 1B, 3.2 3B, 3.1 8B). Within each panel, grey
bars show Stock ExecuTorch 1.5 and vermilion bars show SARC 1.5-r2 on each GPU.
Bar height is the median prefill throughput (tokens/s, log scale, shared across
panels) of n = 5 runs. The whiskers span the minimum to the maximum of the five
runs, and the open circles are the individual runs, jittered horizontally. For 41 of
the 60 bars the min–max spread is below 1 % of the median, so those whiskers are
shorter than the markers at this scale. The visibly wider whiskers are on the Arc
B580, which also drives the host's display.

## Figure 3 (`fig3_heatmap.pdf`)

**Speedup of SARC over stock, per GPU and configuration.** Rows are GPUs, and
columns are model size (1B, 3B, 8B) grouped by quantization scheme (4w, 8da4w).
Each cell shows the ratio of median prefill throughputs (SARC / stock, n = 5 runs
each) to two decimals. The colour scale (cividis, colour-blind safe) starts at
1.0× (no speedup). The 95 % confidence intervals are shown in Figure 1.
