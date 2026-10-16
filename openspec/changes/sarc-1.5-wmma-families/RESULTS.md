# sarc-1.5-wmma-families: results of the cross-device matrix (2026-10-16)

Coordinator summary of the lean finish (owner decision N13.7, 2026-10-15 23:30 UTC). Each device's evidence is
in its own change directory (`xe2-b580/`, `xe2-b70/`, `rdna3-780m/`, `nvidia-4070ti/`, `nvidia-orin/`), each with
its `STATUS.md`, `results/` and the raw rows under its artifact root. Ratios are candidate / same-session control
throughput (valid-run medians, 7 valid runs per arm and cell, interleaved), control = the device's final tuned
profile. Cells: 1B 4w, 1B 8da4w, 3B 4w, 3B 8da4w, 8B 4w, 8B 8da4w. Margin: six-cell geomean >= 0.97 and every
cell >= 0.92 on every member. Attention, decode and fp linear are the control's in both arms.

## Families

### Xe2 (B580, B70): accepted on both members

| finalist | 4w | 8da4w | kernels | B580 cells | B580 geomean | B70 cells | B70 geomean |
|---|---|---|---|---|---:|---|---:|
| xe2-family-a | `sweep_t128x128k16g82s16m8flib` every shape | xe2bto `i` every shape | 2 | 1.000 / 1.000 / 1.014 / 0.980 / 1.018 / 0.964 | 0.9957 | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 0.963 | 0.9938 |
| xe2-family-b | same | xe2bto `i`, `n13` where K*N >= 24e6 (B580 rule) | 3 | 1.007 / 1.000 / 1.014 / 1.003 / 1.017 / 1.000 | 1.0068 | 1.000 / 1.000 / 1.000 / 0.981 / 1.000 / 0.998 | 0.9964 |

Both pass the full gate on both cards (verify `--pdiff` 12/12, SDPA tiers, next tokens SAME on three prompts).
Recommended upstream set: **xe2-family-b** (every cell inside the noise band on both cards; one kernel more than
a, which costs 3.6 to 3.8 % at 8B 8da4w). The release-table 8da4w kernel `t256x64k32g48` fails on both cards
(8da4w cells 0.80 to 0.88). Today the two cards use 4 to 5 distinct linear kernels between them.

### NVIDIA Ampere/Ada (4070 Ti, Orin): accepted on both members

| finalist | 4w | 8da4w | 4070 Ti cells | 4070 Ti geomean | Orin cells | Orin geomean |
|---|---|---|---|---:|---|---:|
| nv-family-a | Orin `bt` every shape | Orin `bf` | REJECTED: production diff fails at K=14336 (8B down projection; `bt` accumulates in fp16, max abs 1.49 vs tol 0.5) on both cards | | | |
| nv-family-b | `bt` for K <= 8192, `bx` (fp32 accumulate) for K > 8192 (Orin rule) | Orin `bf` | 1.018 / 1.040 / 1.036 / 0.992 / 0.952 / 1.042 | 1.0127 | 0.999 / 1.000 / 0.999 / 1.000 / 0.999 / 1.000 | 0.9996 |

nv-family-b passes the full gate on both cards. It is the NVIDIA set: three kernels, all Orin-origin. The
4070 Ti's own kernels fail on the Orin (geomean 0.90, 0.79 and 0.96). On the 4070 Ti the `bx` part costs
about 8 % at the 8B down projection; keeping the 4070 Ti's own `ga` there is a per-device override for the fork
(proposal: overrides stay internal).

### RDNA3 on RADV (780M; RX 7600 and RX 7900 XTX company-side): not decided

On the 780M, 16 of 19 legal candidates are within the margin and none is faster than the 780M's tuned set
outside the noise band. The RX 7600 tiles fail badly on the 780M (4w 0.30 to 0.34, 8da4w 0.36 to 0.41); the
7900 XTX 8da4w sweep `t256x64k32g24` fails (0.906 to 0.932). The 7900 XTX `q4gsw sweep t128x128k32g42 ... bt`
and `dq8ca zpg sweep t128x64k32g22` / `t64x64k32g22` pass within 0.5 %. A family set needs the RX 7600 and
RX 7900 XTX (under RADV) member evidence, which is company-side (`COMPANY-SIDE.md`). Until then the 780M keeps
its tuned set.

### Xclipse (M51): no evidence (company-side).

## Caveats

- Intel: texture3d images over ANV's reported 2048 extent (RoPE tables, 8B activations), a pre-existing caveat
  identical in both arms (N13.7); original model files.
- 780M: own-user GPU-client accounting plus device-wide busy sampling (owner decision 2026-10-15 03:57 UTC).
- 4070 Ti: some 8B rows were replaced after slow loads while a foreign ComfyUI service held host memory; the
  owner stopped that service 2026-10-16 07:25 UTC.
- B580: foreign desktop share up to the e2e5.sh calibration (2.97 %, cap 5 %).

## Cost

First campaign (inventory and five device runs, 2026-10-14 to 15): $426.54, no timed cell. Lean finish
(2026-10-15 23:26 to 2026-10-16 11:15 UTC): $189.74.

## Next

1. Done 2026-10-16 (`fae9e53e76`): family key in `impl/sarc/Select.{h,cpp}` (vendor ID, driver ID,
   cooperative-matrix shapes, subgroup size) with the Xe2 and NVIDIA sets as family rows; the 4070 Ti's
   ga at K > 8192 is its per-device override. Device check on all four members: `select-check/`.
   Open: the NVIDIA key cannot tell Ampere/Ada from other NVIDIA generations with the same shapes
   (Turing, Hopper, Blackwell would match; none measured); the NVIDIA rows cover the measured Llama
   projections only.
2. RDNA3 and Xclipse member evidence through the company side.
3. Port to `pr/wmma-families` off upstream `main` (gap list: the change's upstream-gap notes).
