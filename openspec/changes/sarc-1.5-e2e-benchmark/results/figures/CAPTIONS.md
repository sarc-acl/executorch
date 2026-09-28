# Figure captions: end-to-end prefill, Stock ExecuTorch 1.5 vs SARC 1.5-r2

All figures show end-to-end prefill throughput for a 2048-token prompt, measured
with ExecuTorch's Vulkan backend (`llama_main`). The set covers seven GPUs, three
Llama models and two quantization schemes. 4w is int4 weights with fp16
activations; 8da4w is int8 dynamic activations with int4 weights. Each build ran
n = 5 interleaved timed repeats per cell. Data come from `../raw7/cells.csv`
(per-cell statistics) and `../raw7/runs_all.csv` ("the" × 2048 prompt, built by
`../analyze7.py`; the six-GPU rows are identical to `../raw6/`). Only rows whose `log` starts
with `logs/prefill` are timed runs. Regenerate the figures with `./make_all.sh`.

The speedup-only Figures 1 and 3 also include an eighth device, Xclipse (M51)‡. Its
speedup rows come from the contribution file `m51/speedups.csv`, using the prompt
`the_x2048` like the other GPUs. It never appears in Figure 2, which shows absolute
throughput.

**† Pre-release: RX 7900 XTX and RX 7600.** Their SARC rows ran with
`ET_VK_SARC_UNVERIFIED=1`, so SARC correctness is unverified on these GPUs. Their
`.pte` files come from a different export, so their absolute tokens/s are not
strictly comparable with the other GPUs. Drivers: AMDVLK (RX 7900 XTX) and RADV,
Mesa 26.2.3 (RX 7600).

**‡ Xclipse (M51): internal device, relative speedups only; pre-release rows.** No
absolute throughput is shown for this device. Its 4w kernel was fixed on 2026-09-28; the
end-to-end re-measurement is pending, so no 4w speedup is shown for it: Figure 1
has an empty slot labelled "4w: re-measuring, not shown", and Figure 3 has grey
"n/a" cells.

The pre-release devices are placed last (RX 7900 XTX†, RX 7600†, then Xclipse
(M51)‡) and marked the same way: a dagger (†) or double dagger (‡) on the label,
white hatching on bars or cells, and a tint (a background band in Figures 1 and 2,
a row outline in Figure 3). The tint is reddish purple for the RX 7900 XTX† and
the Xclipse (M51)‡, and wine red for the RX 7600†.

Earlier renders are kept but are not rebuilt by `make_all.sh`:
`fig*_5gpu.{pdf,png}` (five GPUs, made from `../cells.csv` before that file was
regenerated), `fig1_speedup_6gpu.*` / `fig3_heatmap_6gpu.*` (six GPUs, without
the Xclipse (M51)), `fig1_speedup_6gpu_m51.*` / `fig3_heatmap_6gpu_m51.*` (six GPUs
plus the Xclipse (M51)) and `fig2_throughput_6gpu.*` (six GPUs). Their captions
are in `CAPTIONS_6gpu_m51.md`.

## Figure 1 (`fig1_speedup.pdf`)

**SARC prefill speedup over stock ExecuTorch 1.5.** Top: 4w; bottom: 8da4w, with a
shared y-axis. (With eight device groups the two panels are stacked rather than
side by side, so the labels stay legible.) Bars show the speedup for Llama 3.2 1B, Llama 3.2 3B and Llama 3.1
8B on each GPU. The speedup is the median SARC throughput divided by the median
stock throughput, over n = 5 runs of each build. The number above each bar gives
the speedup. Error bars are an approximate 95 % paired bootstrap confidence
interval of the ratio of medians. The two builds ran back to back within each
repeat, so repeats are resampled as pairs. With n = 5 the interval is coarse. The
dashed line marks 1.0× (parity with stock).

Every plotted cell is faster than stock, with speedups from 1.48× to 5.35×. The
speedup increases with model size on four of the eight devices. The exceptions are
the 780M, the RX 7900 XTX†, the RX 7600† and the Xclipse (M51)‡, where 8B is below
3B. Among the seven GPUs with both schemes shown, the speedup is larger for 4w than
for 8da4w on the AMD and Intel GPUs. The RX 7900 XTX† values (pre-release) are
3.06×, 4.01× and 3.83× for 4w and 2.15×, 2.59× and 2.25× for 8da4w (1B, 3B, 8B).
The RX 7600† values (pre-release) are 2.88×, 3.38× and 3.23× for 4w and 1.94×,
2.22× and 1.92× for 8da4w. The Xclipse
(M51)‡ shows 8da4w only: 2.10× [1.89, 2.71], 2.75× [2.61, 2.76] and 2.65× [2.64,
2.66]. Its 1B interval is wide because of repeat spread. Its 4w slot is empty and
labelled "4w: re-measuring, not shown" (kernel fix merged, end-to-end
re-measurement pending).

## Figure 2 (`fig2_throughput.pdf`)

**Absolute prefill throughput, stock vs SARC.** Rows are quantization schemes (4w,
8da4w) and columns are models (Llama 3.2 1B, 3.2 3B, 3.1 8B). Within each panel,
grey bars show Stock ExecuTorch 1.5 and vermilion bars show SARC 1.5-r2 on each
GPU. Bar height is the median prefill throughput over n = 5 runs, on a log-scale
y-axis in tokens/s that is shared across panels. The whiskers span the minimum to
the maximum of the five runs. The open circles are the individual runs, jittered
horizontally.

For 59 of the 84 bars, the min–max spread is below 1 % of the median, so those
whiskers are shorter than the markers at this scale. The visibly wider whiskers are
on the Arc B580, which also drives the host's display. The RX 7900 XTX† and
RX 7600† bars are hatched. Their absolute throughputs come from a different `.pte`
export and are not strictly comparable with the other GPUs.

## Figure 3 (`fig3_heatmap.pdf`)

**Speedup of SARC over stock, per GPU and configuration.** Rows are GPUs. Columns
are model sizes (1B, 3B, 8B), grouped by quantization scheme (4w, 8da4w). Each cell
shows the ratio of median prefill throughputs (SARC / stock, n = 5 runs of each
build) to two decimals. The colour scale (cividis, colour-blind safe) starts at
1.0× (no speedup). The RX 7900 XTX†, RX 7600† and Xclipse (M51)‡ rows
(pre-release) are hatched and outlined. Grey "n/a" cells mark 4w on the Xclipse (M51)‡, whose
end-to-end re-measurement is pending; no 4w speedup is shown.
The 95 % confidence intervals are shown in Figure 1.
