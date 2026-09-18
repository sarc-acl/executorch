# Independent device validation — final results

Independent measurements completed on 2026-09-17/18. This change promotes only the four positive results below, for the measured device and batch-one, group-128, texture3d 2048-token prefill projection shapes. Other shapes retain their prior defaults; explicit environment overrides take precedence. AMD and NVIDIA also compile the winning texture3d shader variants into the normal backend. Launch geometry is parsed from each selected kernel name.

The 3B projection pairs (K,N) are (3072,3072), (3072,1024), (3072,8192), and (8192,3072). The 8B pairs are (4096,4096), (4096,1024), (4096,14336), and (14336,4096).

The defaults are competitive, but the independent checks have found small per-model tile improvements. The original large speedups generally reproduce in comparable machine conditions.

| Device / workload | Patched release baseline | Current default | Challenger | Challenger vs default |
|---|---:|---:|---:|---:|
| AMD 780M, 3B 4w | 419.672 | 934.733 | 941.609 | +0.74% |
| AMD 780M, 1B 8da4w | 1580.25 | 2403.76 | 2392.52 | -0.47% |
| Intel B70, 1B 4w | 4442.52 | 8094.86 | Not tested | — |
| Intel B70, 3B 8da4w | 3175.19 | 4240.17 | 4240.17 | 0.00% |
| Intel B70, 8B 8da4w | 1351.82 | 2015.75 | 2043.91 | +1.40% |
| NVIDIA 4070 Ti SUPER, 3B 4w | 2576.10 | 8827.59 | 8982.46 | +1.75% |
| NVIDIA 4070 Ti SUPER, 3B 8da4w | 2506.73 | 8258.06 | 8000.00 | -3.12% |
| NVIDIA 4070 Ti SUPER, 8B 4w, service stopped | 1118.51 | 4602.25 | 4751.74 | +3.25% |
| NVIDIA 4070 Ti SUPER, 8B 8da4w, service stopped | 1088.78 | 4240.17 | 4023.58 | -5.11% |

Values are median prefill tok/s from three interleaved timed runs per arm, following one discard per arm. This is a targeted challenge to existing defaults, not an exhaustive search or a statistical proof of a global optimum. The release runners disabled event tracing for timing, used a fixed prompt (MD5 afcf9d3b0aa969a496909c57bfa7de7b), temperature 0, num_bos=1 and max_new_tokens=1. AMD used 90-second idle plus temperature <=45C for the promoted 3B cell; Intel/NVIDIA used 30/45-second idle for 3B/8B. All inherited ET_VK_/ETVK_ overrides were cleared, then the required device gates were set (Intel ETVK_DEVICE_INDEX=1 and ET_VK_TEXTURE_COOPMAT=1; NVIDIA ET_VK_COOPMAT_ANY_DEVICE=1).

The low-magnitude numerical probes passed for all selected defaults and challengers: 84 coopmat cases, about 29.4 million output values, plus 18 forced-tiled controls. These checks use CPU references, signed data, nonzero int8 activation zero points, multiple quantization groups, and shapes large enough to require actual coopmat dispatch. They establish coverage of these cases, not model-quality equivalence or correctness for all shapes.

A separate larger-amplitude probe exposed the expected distinction between FP16-accumulating 4w kernels and an FP32 reference. At abs 0.005 OR rel 0.005 tolerance, mismatches across 1,048,576 outputs were 114,901 on AMD, 6 on Intel and 93 on NVIDIA. The AMD tiled control passed. This is a precision observation, not an indexing diagnosis or a downstream model-quality measurement. It demonstrates why one matching generated token is insufficient to establish numerical equivalence.

NVIDIA 8B initially ran with `zun-flux-pipeline.service` holding about 8 GB VRAM. Those results were substantially slower and variable. The user approved temporarily stopping it; VRAM dropped to 2 MiB and clean 8B throughput returned toward the original study's range. The earlier service-on 8B records are retained only as diagnostics. The clean rerun uses identical binaries, models and timing rules, in a separate directory, and the workflow restored the service afterward. Both benchmark and trace processes exited successfully, and service restoration was verified active.

Measured challenger tokens: AMD 3B 4w `tsweep_dbuf4_t128x128k32g24s32`; AMD 1B int8 `tsweep_dbuf4zpg_t256x128k32g48s32`; Intel 3B int8 `tsweep_dbuf4zpg_xe2_t128x32k64g28s32`; Intel 8B int8 `tsweep_dbuf4zpg_xe2_t128x128k32g48s32`; NVIDIA 3B 4w `tsweep_dbuf4_t128x256k16g41s32`; NVIDIA 8B 4w `tsweep_dbuf4_t256x128k16g22s32`; NVIDIA 3B int8 `tsweep_dbuf4zpgtr_mk32_t128x64k32g24s32`; NVIDIA 8B int8 `tsweep_dbuf4zpgtr_mk32_t128x128k32g84s32`. The original AMD/NVIDIA comparison used supplemental shaders linked into temporary runners. The winning variants are now included in each branch’s normal shader YAML.

The old harness narrative should not be taken as established diagnosis: the inspected correctness matrix includes M=256 cases, despite comments claiming its maximum is 128. Its correctness-only summary incorrectly applies large-performance-case labels to the first small correctness results. The initial run of that existing binary used tiled int8 kernels. The sweep driver also searches for full kernel names in text output, while regular timing rows truncate names. These are concrete coverage/reporting weaknesses; this audit does not establish which one caused every historical rejection. The independent harness checks full names directly from shader timing records.

The included measurements comprise 104 end-to-end runs (78 timed, 26 discarded) across nine device/workload cells. All were valid under the protocol, and 17 model traces confirmed the selected coopmat dispatches. Trace event durations were not validated as GPU time and were not used for performance conclusions. The remaining workload cells and other algorithmic designs were not independently swept, so global optimality is not established.

Full original evidence is retained at `/home/doremy/sarc-acl/wmma-study-2026-09-17/independent` on rocky-ryzen. The adjacent `measurements.jsonl` contains this branch’s winning comparisons. `verification.jsonl` records post-promotion model output and kernel dispatch checks against the rebuilt backend, including explicit old-tile overrides and unaffected workloads. New numerical probes link the normal rebuilt backend without supplemental shader objects.
