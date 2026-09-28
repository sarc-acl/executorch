# Evidence figure captions (six GPUs)

All figures: ExecuTorch Vulkan LLM prefill, 2048 tokens. "Stock" = stock ExecuTorch release 1.5 kernels;
"SARC" = tuned cooperative-matrix kernels. 4w = int4 weights, fp16 activations; 8da4w = int8 dynamic
activations, int4 weights. Paths are relative to `report/`. Regenerate with `make_all.sh`; each script
writes the plotted numbers to `<figure>.values.csv`. The earlier five-GPU versions are kept as
`*_5gpu.{pdf,png,values.csv}` with captions in `CAPTIONS_5gpu.md`.

**† RX 7900 XTX (pre-release):** its rows are unverified, come from a different .pte export and run on
the AMDVLK driver (all other AMD data: RADV). Its SARC build includes the Radeon 780M's SARC attention
kernels, so it is the second GPU (after the 780M) where SARC also speeds up attention. Read its numbers
as preliminary.

## e1_roofs_vs_kernels — measured roofs vs achieved GEMM rate (8B, 2048-token prefill)

Horizontal bars: confirmed measured roofs (igpu-roofline `fast` plan, fresh-process confirmation repeats,
clocks not pinned, not ISA-verified) in tera-ops/s (fp16 FLOP or int8 OP, log scale): fp16 scalar FMA
(`alu_fp16`), int8 dot product (`dot_int8`), fp16 cooperative matrix (the variant SARC 4w uses:
`matrix_fp16_fp32`, fp32 accumulate, on the Radeon 780M and RX 7900 XTX†; `matrix_fp16`, fp16 accumulate,
elsewhere) and int8 cooperative matrix (`matrix_int8`). Markers: aggregate achieved rate of the 8B prefill
GEMM (linear-layer) kernels, 2·M·N·K / ETDump dispatch time, 8da4w excluding the activation-quantize
dispatch; circles 4w, triangles 8da4w, grey stock, orange SARC. Each marker sits on the roof it is compared
with; the label gives the rate and its % of that roof. Stock kernels run on the scalar roofs (22–83 % of
them), 2.5–39x below the matching matrix roofs; SARC runs on the matrix roofs (37–70 % for 4w). On both
RDNA3 GPUs the int8 matrix roof is no higher than the fp16 (fp32-acc) matrix roof (780M 0.97x,
RX 7900 XTX† 1.01x; panel subtitles). SARC 8da4w reaches only 22–24 % of the int8 matrix roof on Arc B580,
Arc Pro B70 and Jetson Orin Nano.
Sources: `evidence/combined6/roofline.json` (`gpus.<gpu>.roofs.<name>.value`),
`evidence/combined6/efficiency.csv` (model = 8b; per-GPU rate sources listed in `roofline.json`
`kernel_rates_source`; Orin rates come from a single un-warmed execution).

## e2_time_breakdown — where 8B prefill GPU time goes (8B, 2048-token prefill)

Stacked bars: summed ETDump GPU dispatch time per kernel family, normalised so the stock bar of each
GPU × scheme pair is 100 %; the SARC bar length is therefore 1 / (trace speedup). Segments: prefill GEMM
(linear layers), 8-bit activation quantize (8da4w only), attention QK^T, softmax, AV + KV-cache update,
and everything else (elementwise, RMSNorm, RoPE, copy/view, stock linear-op input transpose, embedding,
LM head, staging). Numbers inside the GEMM segment are GEMM time as % of stock time. Right: absolute GPU
time in ms (SARC: trace speedup stock/SARC) and attention (QK^T + softmax + AV + KV update) share of that
bar's own time. SARC cuts GEMM from 54–91 % of stock time to 8–41 %. Where attention is unchanged
(Arc B580, Arc Pro B70, RTX 4070 Ti SUPER, Jetson Orin Nano) its share rises from 7–19 % to 33–41 % of SARC
time. On the two GPUs with SARC attention kernels, attention is 4.1–4.2x faster (780M) and 6.4–6.7x faster
(RX 7900 XTX†), so its share falls (780M 24 → 15 % and 35 → 14 %; RX 7900 XTX† 24 → 14 % and 42 → 15 %).
Source: `evidence/combined6/trace/families.csv` (model = 8b).

## e3_amdahl_model_size — why end-to-end speedup grows with model size (1B / 3B / 8B, 2048-token prefill)

Small multiples per GPU (speedup panel above, share panel below); blue 4w, green 8da4w. Top (log scale,
stock / SARC): GEMM kernel speedup (dashed, open circles; stock GEMM ms / SARC GEMM ms) and end-to-end
prefill speedup (solid, filled; median tokens/s ratio, label = 8B value). Bottom: GEMM share of stock GPU
time (%). Models: Llama 3.2 1B, 3B, Llama 3.1 8B. Where only GEMM changes, the GEMM speedup is roughly flat
with model size while the GEMM share rises (e.g. RTX 4070 Ti SUPER 4w 76 → 87 %), so the end-to-end speedup
rises (2.87 → 4.04x), as Amdahl's law predicts. On the Radeon 780M and RX 7900 XTX† (dotted, diamonds):
attention speedup and attention share of stock time. Attention speedup is 2.3x (780M) and 4.0–4.1x
(RX 7900 XTX†) at 1B, rising to ~4x (780M) and 6.2–6.7x (RX 7900 XTX†) at 3B and 8B, while the attention
share falls at 8B. As a result the end-to-end speedup on these two GPUs peaks at 3B instead of rising to
8B (780M 2.71 → 2.66x and 1.86 → 1.73x; RX 7900 XTX† 4.01 → 3.83x and 2.59 → 2.25x).
Sources: `evidence/combined6/trace/families.csv` (GEMM / attention ms and shares), `raw6/cells.csv`
(`speedup`, median of timed runs per build).

## e4_speedup_decomposition — GEMM kernel speedup = headroom × efficiency gain (8B, 2048-token prefill)

Per GPU × scheme, log scale. Grey bar: hardware headroom H = matched matrix roof / matched stock roof
(4w: `matrix_fp16` [780M and RX 7900 XTX†: `matrix_fp16_fp32`] / `alu_fp16`; 8da4w: `matrix_int8` /
`dot_int8`). Coloured bar from H to H × E: efficiency gain E = SARC `pct_of_roof` / stock `pct_of_roof`
(blue E > 1, pink E < 1). Black tick: measured GEMM rate ratio (SARC rate / stock rate). Columns: H, E,
H × E and the measured ratio; they agree within 0.21 % (rounding of `pct_of_roof`), and the measured ratio
equals the ETDump GEMM time ratio within 0.05 %. SARC 8da4w on Arc B580 / Pro B70 achieves only ~40 % of
stock int8 dot's roof efficiency (E = 0.40 / 0.41), giving 2.9x of a possible 7.2x. On the RX 7900 XTX†
4w is 2.24 × 1.61 = 3.59x (measured 3.59x); 8da4w is 2.06 × 0.78 = 1.61x, because stock int8 dot already
runs at 83 % of its roof. On Orin 8da4w E = 1.02, so its 9.5x is all headroom.
Sources: `evidence/combined6/efficiency.csv` (model = 8b), cross-check
`evidence/combined6/trace/families.csv`.
