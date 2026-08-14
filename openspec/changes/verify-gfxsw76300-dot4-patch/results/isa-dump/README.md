# GFXSW-76300 dot4-patch verification: ISA evidence

Same pipeline hash `PipelineCs_0xE1626A9FA3C753BB` (same SPIR-V input -> same hash,
regardless of driver build) on M41 (`00000bb7cc34abd3`), 8da4w quantized-linear kernel,
Llama 3.2 1B, 2048-token prefill warmup:

| Driver | v_dot4_i32_i8 / v_dot4_u32_u8 | v_mul_i32_i24 / v_bfe_i32 (emulated fallback) | 8da4w prefill tok/s |
|---|---|---|---|
| unpatched main-tip (`f636a83b...`, + GFXSW-76434 UAF fix `46e41723d8`) | 0 | 70 | 428.5 |
| dot4-patched main-tip (`cfe0caf0...`, `cb993c15f1` = UAF fix + dot4 patch `75616cb859`) | 128 | 2 | 810.8 |

The 128-vs-0 native-dot4 count exactly matches the original bisect's last-good (128) vs
first-bad (0) counts (JIRA GFXSW-76300 comment 6021003), confirming the patch restores the
identical codegen mechanism, not just a coincidental throughput change.

Full 37-pipeline dumps for both drivers were pulled but only the dominant quantized-linear
kernel's disasm is kept here; see cmd log for the full pull if needed.
