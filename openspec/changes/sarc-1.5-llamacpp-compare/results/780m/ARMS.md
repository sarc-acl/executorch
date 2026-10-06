# Radeon 780M: arms of the comparison (written as the timed session started, 2026-10-06)

Device: Radeon 780M (RADV PHOENIX), measured while the tuning campaign's queue was held at a job boundary
(coordinator hold). Models, prompt and context as on the Arc B580. Validity: the campaign's rule (median clock
in the prefill window at least 2700 MHz, no other GPU workload), sampled every 20 ms; this device has no
per-client engine accounting, so there is no foreign-busy ceiling.

## ExecuTorch Vulkan (each with `4w` and `8da4w`)

| arm | build | selection |
|---|---|---|
| `stock` | upstream `release/1.5` at `985c1ceccc` plus the compile-only backport: the binary built for the B580 session | none |
| `sarc` | `8518659ef`, head of `topic/780m-prefill-refine`, nothing selected. The campaign showed that this build with nothing selected gives the same `verify.sh` output as `dev/1.5` line by line | none |
| `tuned` | the same binary: an unmerged development branch | `ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_PROFILE=c10` (candidate 10, reproduced from the committed branch) |

Candidate 10 rests on a fused attention kernel accepted by the reference-error rule: one next-token item
(8B `8da4w`) differs from its parent's.

## llama.cpp `b11430`, Vulkan, built on the device with `-DGGML_VULKAN=ON -DGGML_NATIVE=ON`

`q4_0` and `q4_k_m`, each as `default` and `best`, by both timers (`lc`, `lb`): 8 arms. Exact lines: `arms.tsv`.
llama.cpp reports cooperative matrix and integer dot product on this device.

Screen that fixed `best` (1B, `q4_0`, llama-bench tok/s; `screen.csv`): default 2867; `-ub 2048` flash
attention on **2876**, off 1664; `-ub 1024` on 2868, off 1885; `-ub 512` off 1984. `best` = `-b 2048 -ub 2048
-fa on`, which is also the batch-aligned setting, so there is no separate `aligned` tier here. Flash attention
on is the faster choice on this device, the opposite of the Arc B580's Vulkan result.
