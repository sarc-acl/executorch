# Figure captions: end-to-end prefill, Stock ExecuTorch 1.5 vs SARC 1.5-r2

All figures show end-to-end prefill throughput for a 2048-token prompt, measured
with ExecuTorch's Vulkan backend (`llama_main`). The set covers six GPUs, three
Llama models and two quantization schemes. 4w is int4 weights with fp16
activations; 8da4w is int8 dynamic activations with int4 weights. Each build ran
n = 5 interleaved timed repeats per cell. Data come from `../raw6/cells.csv`
(per-cell statistics) and `../raw6/runs_all.csv`. Only rows whose `log` starts
with `logs/prefill` are timed runs. Regenerate the figures with `./make_all.sh`.

The speedup-only Figures 1 and 3 also include a seventh device, Xclipse (M51)‡. Its
speedup rows come from the contribution file `m51/speedups.csv`, using the prompt
`the_x2048` like the other GPUs. It never appears in Figure 2, which shows absolute
throughput.

**† Pre-release: RX 7900 XTX.** Its SARC rows ran with `ET_VK_SARC_UNVERIFIED=1`,
so SARC correctness is unverified on this GPU. Its `.pte` files come from a
different export, so its absolute tokens/s are not strictly comparable with the
other GPUs. It uses the AMDVLK driver.

**‡ Xclipse (M51): internal device, relative speedups only; pre-release rows.** No
absolute throughput is shown for this device. Its 4w kernel is in development, so no
4w speedup is shown for it: Figure 1 has an empty slot labelled "4w: in
development, not shown", and Figure 3 has grey "n/a" cells.

Both pre-release devices are placed last and marked the same way: a dagger (†) or
double dagger (‡) on the label, white hatching on bars or cells, and a
reddish-purple tint (a background band in Figures 1 and 2, a row outline in
Figure 3).

Earlier renders are kept but are not rebuilt by `make_all.sh`:
`fig*_5gpu.{pdf,png}` (five GPUs, made from `../cells.csv` before that file was
regenerated) and `fig1_speedup_6gpu.*` / `fig3_heatmap_6gpu.*` (six GPUs, without
the Xclipse (M51)).

## Figure 1 (`fig1_speedup.pdf`)

**SARC prefill speedup over stock ExecuTorch 1.5.** Left: 4w; right: 8da4w, with a
shared y-axis. Bars show the speedup for Llama 3.2 1B, Llama 3.2 3B and Llama 3.1
8B on each GPU. The speedup is the median SARC throughput divided by the median
stock throughput, over n = 5 runs of each build. The number above each bar gives
the speedup. Error bars are an approximate 95 % paired bootstrap confidence
interval of the ratio of medians. The two builds ran back to back within each
repeat, so repeats are resampled as pairs. With n = 5 the interval is coarse. The
dashed line marks 1.0× (parity with stock).

Every plotted cell is faster than stock, with speedups from 1.48× to 5.35×. The
speedup increases with model size on four of the seven devices. The exceptions are
the 780M, the RX 7900 XTX† and the Xclipse (M51)‡, where 8B is below 3B. Among the
six GPUs with both schemes shown, the speedup is larger for 4w than for 8da4w on
the AMD and Intel GPUs. The RX 7900 XTX† values (pre-release) are 3.06×, 4.01×
and 3.83× for 4w and 2.15×, 2.59× and 2.25× for 8da4w (1B, 3B, 8B). The Xclipse
(M51)‡ shows 8da4w only: 2.10× [1.89, 2.71], 2.75× [2.61, 2.76] and 2.65× [2.64,
2.66]. Its 1B interval is wide because of repeat spread. Its 4w slot is empty and
labelled "4w: in development, not shown".

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
1.0× (no speedup). The RX 7900 XTX† and Xclipse (M51)‡ rows (pre-release) are
hatched and outlined. Grey "n/a" cells mark 4w on the Xclipse (M51)‡, which is in
development and not shown.
The 95 % confidence intervals are shown in Figure 1.
