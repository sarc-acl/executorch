# Samsung Xclipse (M51): matrix-instruction check of the compiled ISA (2026-09-28)

Labels: [M] read from the compiled ISA, [I] inference.

Per the device owner, this summary carries only whether each kernel emits matrix instructions, their count per MMA
tile, and a short explanation of the fp16 result. No rates, driver or tool identifiers, instruction mnemonics or
further ISA detail are published; the raw ISA stays in the owner's local data.

## Per kernel

| kernel | matrix instructions emitted | count per 16x16x16 MMA tile |
|---|---|---|
| roofline matrix fp16, fp16 accumulate | yes | 1 |
| roofline matrix fp16, fp32 accumulate | yes | 1 |
| roofline matrix int8, int32 accumulate | yes | 1 |
| ExecuTorch "xclipse" 4w row (texture3d and buffer) | yes | 1 |
| ExecuTorch "xclipse" 8da4w row (texture3d and buffer) | yes | 1 |

The 4w row examined is `sarc_linear_q4gsw_coopmat_t128x128k16g22s32`, i.e. the row as it was before the
fp32-accumulate / one-pass-drain fix; the 8da4w row is `sarc_linear_dq8ca_coopmat_zpgtr_t128x64k32g42s32` [M].

## The fp16 result
fp16 cooperative-matrix work is compiled to native matrix instructions, the same as int8, so the fp16 matrix
roof matching the fp16 FMA roof is not caused by the driver lowering fp16 to FMA code or by a missing feature [M].
On this device the fp16 matrix path delivers about the same arithmetic throughput as the fp16 FMA path; its benefit
is operand reuse and fewer instructions rather than a higher peak [I].
