# Evidence figure captions

All figures: ExecuTorch Vulkan LLM prefill, 2048 tokens. "Stock" = stock ExecuTorch release 1.5 kernels;
"SARC" = tuned cooperative-matrix kernels. 4w = int4 weights, fp16 activations; 8da4w = int8 dynamic
activations, int4 weights. Paths are relative to `report/`. Regenerate with `make_all.sh`; each script
writes the plotted numbers to `<figure>.values.csv`.

## e1_roofs_vs_kernels — measured roofs vs achieved GEMM rate (8B, 2048-token prefill)

Horizontal bars: confirmed measured roofs (igpu-roofline `fast` plan, 3 fresh-process repeats, clocks not
pinned, not ISA-verified) in tera-ops/s (fp16 FLOP or int8 OP, log scale): fp16 scalar FMA (`alu_fp16`),
int8 dot product (`dot_int8`), fp16 cooperative matrix (`matrix_fp16_fp32`, fp32 accumulate, on the
Radeon 780M; `matrix_fp16`, fp16 accumulate, elsewhere: the variant SARC 4w uses) and int8 cooperative
matrix (`matrix_int8`). Markers: aggregate achieved rate of the 8B prefill GEMM (linear-layer) kernels,
2·M·N·K / ETDump dispatch time, 8da4w excluding the activation-quantize dispatch; circles 4w, triangles
8da4w, grey stock, orange SARC. Each marker sits on the roof it is compared with; the label gives the rate
and its % of that roof. Stock kernels run at 22–59 % of the scalar roofs and 3.6–39x below the matching matrix roofs;
SARC runs on the matrix roofs (37–70 % for 4w). On the 780M the int8 matrix roof (14.4 TOP/s) is 0.97x the
fp16 (fp32-acc) matrix roof (14.8 TFLOP/s). SARC 8da4w reaches only 22–24 % of the int8 matrix roof on
Arc B580, Arc Pro B70 and Jetson Orin Nano.
Sources: `evidence/roofline.json` (`gpus.<gpu>.roofs.<name>.value`), `evidence/efficiency.csv`
(model = 8b; rates from `evidence/efficiency.csv`, computed from the warm ETDump traces
`evidence/trace/gemm.csv`, all five GPUs including Orin).

## e2_time_breakdown — where 8B prefill GPU time goes (8B, 2048-token prefill)

Stacked bars: summed ETDump GPU dispatch time per kernel family, normalised so the stock bar of each
GPU × scheme pair is 100 %; the SARC bar length is therefore 1 / (trace speedup). Segments: prefill GEMM
(linear layers), 8-bit activation quantize (8da4w only), attention QK^T, softmax, AV + KV-cache update,
and everything else (elementwise, RMSNorm, copy/view, embedding, LM head, staging). Numbers inside the
GEMM segment are GEMM time as % of stock time. Right: absolute GPU time in ms (SARC: trace speedup
stock/SARC) and attention (QK^T + softmax + AV + KV update) share of that bar's own time. SARC cuts GEMM
from 57–91 % of stock time to 8–41 %, so attention rises from 7–19 % to 33–41 % of SARC time on Intel,
NVIDIA and Orin. On the 780M SARC also speeds up attention (~4x), so its attention share falls (24 → 15 %,
35 → 14 %).
Source: `evidence/trace/families.csv` (model = 8b).

## e3_amdahl_model_size — why end-to-end speedup grows with model size (1B / 3B / 8B, 2048-token prefill)

Small multiples per GPU; blue 4w, green 8da4w. Top (log scale, stock / SARC): GEMM kernel speedup (dashed,
open circles; stock GEMM ms / SARC GEMM ms) and end-to-end prefill speedup (solid, filled; median
tokens/s ratio, label = 8B value). Bottom: GEMM share of stock GPU time (%). Models: Llama 3.2 1B, 3B,
Llama 3.1 8B. The GEMM speedup is roughly flat with model size while the GEMM share rises (e.g. 4070 Ti
SUPER 4w 76 → 87 %), so the end-to-end speedup rises (2.87 → 4.04x), as Amdahl's law predicts. Radeon 780M
only (dotted, diamonds): attention speedup (stock / SARC attention ms) and attention share of stock time;
attention speeds up ~4x at 3B and 8B (2.3x at 1B) while its share falls at 8B (34 → 24 % for 4w,
45 → 35 % for 8da4w), which is why the 780M end-to-end speedup does not rise from 3B to 8B
(2.71 → 2.66x, 1.86 → 1.73x).
Sources: `evidence/trace/families.csv` (GEMM / attention ms and shares), `cells.csv` (`speedup`, median of
5 runs per build).

## e4_speedup_decomposition — GEMM kernel speedup = headroom × efficiency gain (8B, 2048-token prefill)

Per GPU × scheme, log scale. Grey bar: hardware headroom H = matched matrix roof / matched stock roof
(4w: `matrix_fp16` [780M `matrix_fp16_fp32`] / `alu_fp16`; 8da4w: `matrix_int8` / `dot_int8`). Coloured
bar from H to H × E: efficiency gain E = SARC `pct_of_roof` / stock `pct_of_roof` (blue E > 1, pink E < 1).
Black tick: measured GEMM rate ratio (SARC rate / stock rate). Columns: H, E, H × E and the measured
ratio; they agree within 0.21 % (rounding of `pct_of_roof`), and the measured ratio equals the ETDump GEMM
time ratio within 0.05 %. SARC 8da4w on Arc B580 / Pro B70 uses only ~40 % of the efficiency stock int8 dot
achieves (E = 0.40 / 0.41), leaving 2.9x of a possible 7.2x; on Orin 8da4w E = 1.02, so its 9.5x is
all headroom.
Sources: `evidence/efficiency.csv` (model = 8b), cross-check `evidence/trace/families.csv`.
