# specs/037 M51 deep sweep: prefill speedup, coopmat over T-tiled baseline

| Model | Scheme | Baseline (T-tiled) tok/s | Coopmat tok/s | Speedup | Winner token |
|---|---|---|---|---|---|
| 1B | 4w | 706.5 (n=3) | 1675.9 (n=3) | 2.372x | tsweep_t128x128k16g22s32 |
| 1B | 8da4w | 480.5 (n=3) | 1497.1 (n=3) | 3.116x | tsweep_t64x32k32g12s64 |
| 3B | 4w | 257.2 (n=3) | 692.6 (n=3) | 2.693x | tsweep_t128x128k16g22s32 |
| 3B | 8da4w | 175.2 (n=3) | 606.5 (n=3) | 3.461x | tsweep_t64x32k32g12s64 |
| 8B | 4w | 98.3 (n=3) | 322.6 (n=3) | 3.281x | tsweep_t128x128k16g22s32 |
| 8B | 8da4w | 73.5 (n=3) | 279.6 (n=3) | 3.804x | tsweep_t64x32k32g12s64 |
