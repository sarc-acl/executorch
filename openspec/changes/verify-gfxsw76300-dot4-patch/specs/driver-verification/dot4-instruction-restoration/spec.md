## Purpose

Defines the observable behavior a SUMD driver build must exhibit to count as "fixing" GFXSW-76300: native dot4 instruction selection restored for 8da4w on GFX405 targets, with throughput recovered and no correctness regression.

## ADDED Requirements

### Requirement: Native dot4 instruction selection restored
A SUMD driver built with Jayati's patch on top of the first-bad commit (`69e887275e`) SHALL emit the native `v_dot4_i32_i8`/`v_dot4_u32_u8` instructions for the 8da4w quantized matmul kernel on GFX405 targets (M41, M51), instead of the software-emulated multiply/add-reduce sequence.

#### Scenario: Disassembly shows native dot4 after patch
- **WHEN** the 8da4w Llama 3.2 1B prefill shader is compiled and disassembled on the patched driver
- **THEN** the disassembly contains `v_dot4_i32_i8` (or `v_dot4_u32_u8`) instructions for the quantized matmul, matching the last-good baseline (`0b814fa6d3`, 128 occurrences) rather than the first-bad baseline (`69e887275e`, 0 occurrences, replaced by `v_mul_i32_i24`/`v_bfe_i32` emulation)

### Requirement: 8da4w prefill throughput recovered on M41
8da4w prefill throughput on the M41 bisect device, measured with the same workload and clocks used in the original GFXSW-76300 bisect, SHALL recover to within 10% of the last-good baseline (811.4 tok/s) and SHALL be clearly outside the regression band represented by the first-bad baseline (221.7 tok/s).

#### Scenario: M41 8da4w prefill matches pre-regression throughput
- **WHEN** the patched driver runs Llama 3.2 1B 8da4w, 2048-token prefill, pinned 980/5333/800 clocks, on M41 (`00000a34cdd4abd3`)
- **THEN** measured prefill throughput is at least 730 tok/s (>= 90% of 811.4 tok/s) and at least 3x the first-bad throughput of 221.7 tok/s

### Requirement: 4w prefill throughput unaffected
4w prefill throughput on M41, measured under the same conditions, SHALL remain within measurement noise of its pre-patch value across all bisect commits (~598-610 tok/s), confirming the patch does not perturb the unrelated int4-weight-only path.

#### Scenario: M41 4w prefill stays flat after patch
- **WHEN** the patched driver runs Llama 3.2 1B 4w, 2048-token prefill, pinned 980/5333/800 clocks, on M41
- **THEN** measured prefill throughput stays within 5% of the ~598-610 tok/s range recorded across every bisect commit (good and bad alike)

### Requirement: Correctness preserved
The patched driver SHALL preserve both output correctness (no garbage/incoherent tokens for either quant mode) and the CTS `integer_dot_product` pass status Jayati already validated.

#### Scenario: Generated text remains coherent
- **WHEN** running the 4w and 8da4w prefill+decode benchmark on the patched driver
- **THEN** decoded output text is coherent (no repeated-garbage-token pattern), matching last-good-driver behavior

#### Scenario: CTS integer dot product still passes
- **WHEN** `test_integer_ops integer_dot_product` CTS cases are run against the patched driver
- **THEN** all cases pass, consistent with Jayati's manual test report on the patch
