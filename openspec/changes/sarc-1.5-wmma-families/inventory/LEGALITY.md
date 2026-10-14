# Static legality and pending device properties

Source commit: `6178971845`. No GPU properties were queried in this run. Old campaign comments or
historical successful runs do not prove current legality. `legality-matrix.csv` expands each deduplicated
candidate against every member of its family: all entries are conditional, `PENDING_QUERY`.

A later device actor must fill these requirements from that specific unit/driver before pruning the
finite candidate list. No candidate gets a legal/illegal verdict from a device name alone. The manifests
are inventory closure, not a claim that all candidates should be timed in all storage modes. Match actual
model storage first; retain supported alternate storage combinations for correctness coverage.

| Requirement | Source fact / later check |
|---|---|
| Family identity | Proposal requires vendor ID, driver ID, complete cooperative-matrix shape/type signature, subgroup-size range and maximum compute shared memory. RDNA3 requires RADV, never infer from silicon or name. These are pending for all eight devices |
| Cooperative matrices | Enumerate `VkCooperativeMatrixPropertiesKHR`: subgroup scope, M/N/K, A/B/C/result types, non-saturating multiply-add. 4w requires fp16 A/B and fp16 or fp32 C/result as recorded. 8da4w requires signed int8 A/B and signed int32 C/result; its fp32 running total is conversion/addition, not a separate fp32-input MMA requirement |
| Subgroup | Required size is in each manifest; confirm the default size matches or compute-stage required-subgroup-size control supports it. Confirm `subgroupSizeControl`, compute stage support, `computeFullSubgroups` and `maxComputeWorkgroupSubgroups` for the requested full-subgroup pipeline. Merely being within min/max is insufficient |
| Workgroup | `SG_GRID_X*SG_GRID_Y*SUBGROUP_SIZE` invocations; this code launches the local size on x. Check `maxComputeWorkGroupInvocations`, `maxComputeWorkGroupSize[0]`, required subgroup count and dispatch-grid limits for the real shape. Some RDNA3 entries require 1024 invocations |
| Shared memory | Actual source declaration sum in each storage variant must be <= `maxComputeSharedMemorySize`; compiled module allocation/alignment is checked later on exact binaries. Do not rely on the selector's partial estimate |
| Shader capabilities | Cooperative matrix and relevant fp16/int8 arithmetic/storage support, subgroup operations, buffer/image format and access support, storage image reads for effective `IMG_A`/`IMG_W`, required descriptor formats and device limits. The wrappers declare features; compilation alone is not legality |
| Graph / shape | No bias, batch=1, fp16 output and M>1; IO and output storage agree; texture3d IO width-packed; registered storage mask; matching fixed int8 layout. Normal fit requires M%tile_M=0, N%tile_N=0, K%tile_K=0, group_size%tile_K=0. Positive dimensions and group size, real packed tensor layout and valid group partitioning are required. Keep route predicate and parent-service condition |
| Static geometry | tile_M divisible by SG_GRID_Y*MMA_M; tile_N by SG_GRID_X*MMA_N; tile_K by MMA_K. FRAG_LAYOUT needs MMA_K=MMA_N=16. A-map checks below must hold |
| Runtime exception | `Dq8caCoopmat.cpp:94-108` reselects a row-major table row with `ignore_alignment=true` after ordinary selection fails. This is a guarded runtime fallback for an already sized row-major graph, not authorization to launch arbitrary unaligned candidates or to change the build-time layout |

Implementation references: `backends/vulkan/runtime/graph/ops/impl/sarc/Select.cpp:72` (fit),
`:141` (table), `Select.h:79` (layout/ignore_alignment),
`Dq8caCoopmat.cpp:195` (build shape) and `:238` (fixed activation layout).
`backends/vulkan/runtime/api/Shader.h` / pipeline construction use the wrapper's required subgroup size;
the F1 release commit `99989d9a3a` is already in the pinned base. No pipeline code was edited here.
The exact source memory and wrapper references for every candidate are in the manifests and hash list.

## Declared shared-memory derivation

Counts are bytes of declared arrays after the variant's wrapper defines; they are source requirements,
not queried limits or compiled SPIR-V measurements. Let m,n,k be workgroup tiles and y the subgroup-grid
height. All variants here have no bias and no register-array accumulator spill buffer. Types count as
uvec4=16, uvec2=8, uint/int/float=4 and float16_t=2 bytes. Two slices means double buffering.

For ordinary 4w (`sarc_linear_q4gsw_coopmat_body.glslh:31-114`,
`sarc_dev_linear_q4gsw_coopmat_bx_body.glslh:32-115`):

- Row-padded A+B: `4*(m*(k+8)+k*(n+8))`.
- Fragment-contiguous A+B: `4*(m*k+k*n)`.
- Column-major B A+B: `4*(m*(k+8)+n*(k+8))`.
- RX 7600 A_V2 / padding: `4*(m*(k+A_PAD_H)+n*(k+BT_PAD_H))`;
  `sarc_dev_rx7600_q4_body.glslh:52-138`. Its final variant is 59,392 bytes.
- Texture drain adds `2*y*MMA_M*n`, or `2*MMA_M*n` for CSH_BAND, or `2*m*n` for CSH_FULL.
  CSH_IN_ASH adds zero; CSH_POOL uses `max(A+B, full drain)`. The wrapper enables FULL/POOL only for
  texture3d, so buffer variants use separate A+B even when YAML flags are true.
- M51 bz's column-major B inside the pool uses the column-major A+B expression, then
  `max(A+B, 2*m*n)` for texture3d; `sarc_dev_m51_linear_q4gsw_coopmat_bz_body.glslh:94-137`.
  Its final texture3d and buffer variants declare 40,960 bytes. The older k16 table row declares
  32,768 bytes texture3d and 20,992 bytes buffer.

For 8da4w (`sarc_linear_dq8ca_coopmat_zpg_body.glslh:24-78`,
`sarc_linear_dq8ca_coopmat_zpgtr_body.glslh:24-81`, and corresponding dev bodies):

- Ordinary double-buffered A+B: `2*k*(m+n)` bytes; raw-uvec4 A changes typing, not capacity.
- Activation/weight parameters add `8*m + 12*n` bytes: izp, ifs, two weight-scale slices and wcorr.
- Texture drain adds `2*y*MMA_M*n`, unless CSH_IN_ASH reuses A. Reused drain must fit the allocated A.
- RX pitch form uses `slices*(k/MMA_K)*(m*A_PITCH+A_PAD+n*B_PITCH)*4` for A+B;
  `sarc_dev_rx7600_dq8ca_zpg_body.glslh:35-121`. Final A_PITCH=6,B_PITCH=4,A_PAD=0, CSH_IN_ASH
  gives 60,160 total bytes. The older 780M afmb1 has smaller A+B but a separate drain and also totals 60,160.
- Orin bf uses one slice only if ONE_SLICE is true; the final bf variant has two. The Xe2 order variants
  add no shared arrays: same capacity as xe2bt, 43,520 texture3d / 35,328 buffer bytes.

For zpg A_MAP_FULL the block map has `(m/4)*(k/4)` blocks. With A_MULTI_BLOCK, require
`WG_SIZE*A_BLOCKS == (m/4)*(k/4)`; without it require WG_SIZE equal to that block count when the guard
is removed. All inventoried variants satisfy their declared mapping. This arithmetic is checked during
source validation; it is independent of whether a target permits that workgroup.

## Selector limitations and unknowns

`Select.cpp:109-131` checks a full-pool estimate or the texture C drain, not all declared A/B/C arrays.
For a CSH_IN_ASH row it does not bound A+B at all. For buffer IO it does not bound shared memory.
Its full-pool estimate assumes row-major B, which undercounts M51 bz's pooled column-major B
(37,888 estimated vs 40,960 declared). Keep these limitations visible; this task does not repair release code.

The historical XTX AMDVLK run declared 40,960 / 60,160 / 61,440-byte linear variants while reporting a
32,768-byte limit (`sarc-1.5-7900xtx-prefill-refine/STATUS.md`, owner-question section). The supplied
proposal resolves the next prerequisite by requiring RADV evidence. It does not resolve actual RADV
limits; the XTX packet requires the query before any such pipeline is used. A target limit below a
candidate's declared requirement statically excludes it, irrespective of past success.

Exact per-unit driver IDs, shapes/types, subgroup features/ranges, shared/workgroup limits and model
storage/metadata remain pending for all devices. Model files, compiled SPIR-V, correctness, timings,
family margin, NVIDIA split/minimality and M51 private evidence are later-stage work. Missing external
evidence has not been replaced by historical prose. Source-only completion does not depend on obtaining it.
