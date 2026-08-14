## Why

GFXSW-76300 traced an 8da4w-slower-than-4w prefill regression on M41/M51 (GFX405) to SUMD commit `69e887275e`, which gates native `v_dot4_i32_i8`/`v_dot4_u32_u8` codegen behind the `ShWmmaMatrixMultiply` feature flag (disabled by default on these chips). 8da4w's quantized matmul inner loop is dot4-bound, so it falls back to 8x-inflated software emulation (3-4x end-to-end slowdown); 4w never touches that path and is unaffected. Jayati Sahu (SUMD team) has now posted a patch (`Id11af22dd1ddc549339a90074ea22d61375da7b8`) that decouples dot4 instruction selection from the WMMA flag while keeping emulation only for the mixed-signedness/packed-vector cases her CTS testing found necessary. We need to verify, on our own hardware and workload, that it actually restores the throughput without introducing new correctness issues, and report that back on the ticket.

## What Changes

- Apply Jayati's patch on top of the known first-bad SUMD commit (`69e887275e`) and build a SUMD driver from it.
- Flash the patched driver to the M41 bisect device (`00000a34cdd4abd3`, `xgpusw-debug07`) used for the original GFXSW-76300 bisect.
- Re-run the same 4w vs 8da4w prefill benchmark from the bisect (Llama 3.2 1B, 2048-token prefill, `llama_main_rel1.3`, pinned 980/5333/800) and compare against the recorded last-good (`0b814fa6d3`: 8da4w 811.4 tok/s) and first-bad (`69e887275e`: 8da4w 221.7 tok/s) baselines.
- If the patched build is stable on M41, repeat on the primary M51 board at its own pinned clocks, since the ticket's title and original repro are specifically about the M5 EVT1 driver.
- Inspect shader disassembly (as done during the bisect) to confirm `v_dot4_i32_i8` is actually emitted again for the 8da4w kernel, not just infer it from throughput.
- Post the result as feedback on GFXSW-76300 for Jayati/Pavan.

## Capabilities

### New Capabilities
- `driver-verification/dot4-instruction-restoration`: verification behavior for confirming a SUMD driver build restores native dot4 (`v_dot4_i32_i8`/`v_dot4_u32_u8`) instruction selection for 8da4w on GFX405 targets (M41/M51) without regressing 4w or correctness.

### Modified Capabilities
(none)

## Impact

- No ExecuTorch source changes — this is purely a verification pass against an external SUMD driver patch. Any files touched live in the SUMD checkout (`/local/yanwen.xu/sumd/main` or equivalent), not this repo.
- Reuses existing GFXSW-76300 bisect assets: `scripts/bisect-test.sh`, the M41 bisect device, and the `.pte_out`/NFS PTEs already staged for that investigation — no new export needed.
- Result feeds back into JIRA GFXSW-76300 as a comment, not into any workspace doc.
