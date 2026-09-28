# Samsung Xclipse (M51) — sarc-1.5-e2e-benchmark contribution (2026-09-28)

Labels: [M] measured in this campaign, [S] source, [I] inference, [O] open.

**Reduced contribution.** Per the device owner, this GPU publishes relative speedups only: no absolute tok/s,
kernel times, rates or roof values, and no device, driver or lab identifiers. So `raw/runs.csv`,
`trace/{families,gemm,totals}.csv`, `efficiency.csv` and `roofline.json` from the CONTRIBUTING-A-GPU.md §5 layout are not included; they
exist in the owner's local campaign data. Included: `speedups.csv` (per cell: speedup, paired 95 % CI, n, repeat
spread in %, output correctness), `raw/nexttoken.csv`, `trace/dispatch.csv` (kernel names and counts) and
`probe.csv` (top-1 agreement only).

## Setup
- Device: a Samsung Xclipse (M51) device, Android; the same driver build was verified before every run [M]. GPU/memory clocks fixed for the whole campaign; one measuring
  process at a time; thermal pacing on the GPU sensor [M].
- Protocol: CONTRIBUTING-A-GPU.md M1 (5 interleaved repeats, both prompts, fresh process per run), M2 (warm
  ETDump), M4 (logits probe), driven from the host over adb [M]. M3 (roofline) was measured but is not published.
- stock: `release/1.5` @ 985c1ceccc8b + the kit backport patch; sarc: `dev/1.5` @ d98227f60 with its unverified
  "xclipse" rows (`ET_VK_SARC_UNVERIFIED=1`): 4w `sarc_linear_q4gsw_coopmat_t128x128k16g22s32` (fp16 accumulate),
  8da4w `sarc_linear_dq8ca_coopmat_zpgtr_t128x64k32g42s32`, SDPA prefill `sarc_sdpa_{qk,av}_coopmat` [S].
- Both arms also carry a local, measurement-only runtime option `ET_VK_EXECUTE_NODE_THRESHOLD=32` (submit a command
  buffer every 32 graph nodes instead of 128). Without it some Llama 3.1 8B runs did not complete; with it, no run failed (132 timed/check runs, 12 traces, 24 probes) [M]. The option
  ports an opt-in env override from the 1.4 branch into ComputeGraph.cpp; it is not part of dev/1.5 [S].
- Native build (NDK r29, per-tree venv), not the pinned container; the SPIR-V golden was not checked [M].
- Models: `*_embq_ctx3072.pte`, the same export recipe as the other GPUs, exported separately (files not byte-identical) [M].

## Results (speedup = SARC / stock prefill tok/s, median of 5 interleaved repeats, paired 95 % CI)
| prompt | model | 4w | 8da4w |
|---|---|---|---|
| real 2048 | 1B | re-measurement pending | **2.18x** [2.11, 2.48] |
| real 2048 | 3B | re-measurement pending | **2.73x** [2.72, 2.75] |
| real 2048 | 8B | re-measurement pending | **2.64x** [2.63, 2.65] |
| "the"x2048 | 1B | re-measurement pending | 2.10x [1.89, 2.70] |
| "the"x2048 | 3B | re-measurement pending | 2.75x [2.61, 2.76] |
| "the"x2048 | 8B | re-measurement pending | 2.65x [2.64, 2.66] |

8da4w geomean (real text) 2.50x [M].

- **8da4w is correct** [M]: production diff 1B/3B/8B x buffer/texture3d with nonzero zero points passes; probe top-1
  equals stock on both inputs for all three models; SDPA correctness 4/4.
- **4w:** the row measured here was retired on 2026-09-28 and replaced by `t128x128k16g22s32f32xp` (fp32
  accumulate and a one-pass texture3d drain, required on this device), which passes the production diff for 1B/3B/8B x
  buffer/texture3d. Its end-to-end re-measurement is pending; the retired row's numbers are not published.
- M2 [M]: SARC dispatches every prefill GEMM (112/196/224 = 7 per layer) on the xclipse kernels and attention on
  `sarc_sdpa_*`; stock shows no `sarc_*` kernel (`trace/dispatch.csv`).
- Next token on the unaligned 1972-token check prompt: SAME in all 6 cells; for 4w the row needs M % 128, so these
  check runs fall back to stock kernels (stock vs stock) [I, S]; the zpgtr 8da4w row keeps SARC at every M [S].

## Anomalies
- Noisy cells (repeat spread > 3 %): see `speedups.csv` (1B 8da4w on both prompts, 3B 8da4w "the") [M].
- A first campaign without the node threshold had one run that did not complete (8B 4w SARC, check prompt); it was
  discarded and the whole campaign re-run with the threshold [M].

## Not measured / not published
- M3 roofs and kernel % of roof: measured, not published. M5: no re-tuning on 1.5 -> N/A. Rows remain kUnverified.
