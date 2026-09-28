# Figure captions: end-to-end prefill, Stock ExecuTorch 1.5 vs SARC 1.5-r2

All figures show end-to-end prefill throughput for a 2048-token prompt, measured
with ExecuTorch's Vulkan backend (`llama_main`). The set covers six GPUs, three
Llama models and two quantization schemes. 4w is int4 weights with fp16
activations; 8da4w is int8 dynamic activations with int4 weights. Each build ran
n = 5 interleaved timed repeats per cell. Data come from `../raw6/cells.csv`
(per-cell statistics) and `../raw6/runs_all.csv`. Only rows whose `log` starts
with `logs/prefill` are timed runs. Regenerate the figures with `./make_all.sh`.

**† Pre-release: RX 7900 XTX.** Its SARC rows ran with `ET_VK_SARC_UNVERIFIED=1`,
so SARC correctness is unverified on this GPU. Its `.pte` files come from a
different export, so its absolute tokens/s are not strictly comparable with the
other GPUs. It uses the AMDVLK driver. It is placed last in every figure and marked
the same way throughout: a dagger (†) on its label, white hatching on its bars or
cells, and a reddish-purple tint (a background band in Figures 1 and 2, a row
outline in Figure 3).

The earlier five-GPU renders are kept as `fig*_5gpu.{pdf,png}`. They were made
from `../cells.csv` before that file was regenerated, and they are not rebuilt by
`make_all.sh`.

## Figure 1 (`fig1_speedup.pdf`)

**SARC prefill speedup over stock ExecuTorch 1.5.** Left: 4w; right: 8da4w, with a
shared y-axis. Bars show the speedup for Llama 3.2 1B, Llama 3.2 3B and Llama 3.1
8B on each GPU. The speedup is the median SARC throughput divided by the median
stock throughput, over n = 5 runs of each build. The number above each bar gives
the speedup. Error bars are an approximate 95 % paired bootstrap confidence
interval of the ratio of medians. The two builds ran back to back within each
repeat, so repeats are resampled as pairs. With n = 5 the interval is coarse. The
dashed line marks 1.0× (parity with stock).

Every cell is faster than stock, with speedups from 1.48× to 5.35×. The speedup
increases with model size on four of the six GPUs. The exceptions are the 780M and
the RX 7900 XTX†, where 8B is below 3B. The speedup is larger for 4w than for 8da4w
on the AMD and Intel GPUs. The RX 7900 XTX† values (pre-release) are 3.06×, 4.01×
and 3.83× for 4w and 2.15×, 2.59× and 2.25× for 8da4w (1B, 3B, 8B).

## Figure 2 (`fig2_throughput.pdf`)

**Absolute prefill throughput, stock vs SARC.** Rows are quantization schemes (4w,
8da4w) and columns are models (Llama 3.2 1B, 3.2 3B, 3.1 8B). Within each panel,
grey bars show Stock ExecuTorch 1.5 and vermilion bars show SARC 1.5-r2 on each
GPU. Bar height is the median prefill throughput over n = 5 runs, on a log-scale
y-axis in tokens/s that is shared across panels. The whiskers span the minimum to
the maximum of the five runs. The open circles are the individual runs, jittered
horizontally.

For 50 of the 72 bars, the min–max spread is below 1 % of the median, so those
whiskers are shorter than the markers at this scale. The visibly wider whiskers are
on the Arc B580, which also drives the host's display. The RX 7900 XTX† bars are
hatched. Its absolute throughputs come from a different `.pte` export and are not
strictly comparable with the other GPUs.

## Figure 3 (`fig3_heatmap.pdf`)

**Speedup of SARC over stock, per GPU and configuration.** Rows are GPUs. Columns
are model sizes (1B, 3B, 8B), grouped by quantization scheme (4w, 8da4w). Each cell
shows the ratio of median prefill throughputs (SARC / stock, n = 5 runs of each
build) to two decimals. The colour scale (cividis, colour-blind safe) starts at
1.0× (no speedup). The RX 7900 XTX† row (pre-release) is hatched and outlined.
The 95 % confidence intervals are shown in Figure 1.
