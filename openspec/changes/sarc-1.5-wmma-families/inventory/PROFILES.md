# Final profile closure

All references in this document are repository-relative at commit `6178971845`.
`impl/` below means `backends/vulkan/runtime/graph/ops/impl/`.
Every final environment includes `ET_VK_SARC_UNVERIFIED=1`. Clear unrelated profile, exact-kernel,
variant, force-tiled and disable variables before reconstructing an environment.

| Device / family | Final profile and evidence | Linear closure |
|---|---|---|
| 780M / RDNA3 RADV | `ET_VK_SARC_DEV_PROFILE=780m-final`; `sarc-1.5-780m-prefill-refine/STATUS.md:3`; `sarc/README.md:125` (profile table follows) | `Overrides.cpp:1932` chooses `k780mRefine3` (`:1180`), and `:2742` / `:2754` resolves the final name to c11 / `k780mRefine11` (`:2693`); outer five ordered 4w picks and one 8da4w pick, then refine3, then AMD table |
| RX 7600 / RDNA3 RADV | `ET_VK_SARC_RX7600_PROFILE=rx7600-refine5`; `sarc-1.5-rx7600-prefill-refine/STATUS.md:9`; `sarc/README.md` final-profile table | `Overrides.cpp:2964` / `:2986`: two exact production-NK picks, otherwise prior table; 780M softmax/fused controls below do not add linear preferences |
| RX 7900 XTX / RDNA3 RADV prerequisite | `ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5`; `sarc-1.5-7900xtx-prefill-refine/STATUS.md:18` | `Overrides.cpp:3558` / `:3619`: one 4w pick and four 8da4w picks, then AMD table. This profile was measured under AMDVLK historically; no RADV evidence is inferred |
| B580 / Xe2 | `ET_VK_SARC_DEV_PROFILE=b580-fused1-order-p13`; `sarc-1.5-b580-fused-port/STATUS.md:20` | `Overrides.cpp:2303` -> `kB580Fused1Order_p13` (`:1835`); no 4w preference; 8da4w n13 for N*K >= 24,000,000, i otherwise, then Intel table |
| B70 / Xe2 | `ET_VK_SARC_DEV_PROFILE=b70-fused1-order-i-n26-w16m`; `sarc-1.5-b70-fused-port/STATUS.md:27` | `Overrides.cpp:2310` -> `kB70Fused1Order_i_n26_w16m` (`:1888`); 4w N<=512 k32 preference, then k16; 8da4w n26 for N*K >= 33,554,432, i otherwise, then table |
| RTX 4070 Ti SUPER / NVIDIA Ampere-Ada | `ET_VK_SARC_DEV_PROFILE=4070ti-fused1`; `sarc-1.5-4070ti-fused-port/STATUS.md:3` | `Overrides.cpp:2007` -> `k4070tiRefine1` (`:1264`) has attention preferences only: all linear choices are `table_nvidia.cpp:57-95` |
| Orin Nano / NVIDIA Ampere-Ada | `ET_VK_SARC_DEV_PROFILE=orin-fused1`; `sarc-1.5-orin-fused-port/STATUS.md:3` | `Overrides.cpp:2091` -> `kOrinRefine5` (`:1444`): 8da4w bf, 4w bx for K>8192 and bt for remaining 256-row shapes, otherwise NVIDIA table |
| M51 / Xclipse | `ET_VK_SARC_M51_PROFILE=c8`; `sarc/README.md` final-profile table; `sarc-1.5-m51-prefill-refine/proposal.md` Results / second round | `Overrides.cpp:3278` -> `kM51PicksC8` (`:3252`): bz 4w and zpg_bt 8da4w, then Xclipse rows in `table_amd.cpp`. The public proposal is number-free; private status/raw rows are unavailable here |

The change-directory shorthand in the evidence column is under `openspec/changes/`.
The executable registration, preference and predicate references for every individual route are in
`inventory.csv`; documentation line labels above are navigation aids, with source taking precedence.

The override chain is `table -> dev_select -> select_780m -> select_rx7600 -> select_m51 -> select_7900xtx`
within `impl/sarc_dev/Overrides.cpp`. `dev_select:2414` searches candidates before table rows by suffix,
requiring a table choice. Exact names used by outer layers refer to registered candidates. Their row storage
masks matter even when the YAML contains more storage variants. Rows with unchanged names are not renamed
or collapsed across different bodies/accumulation/storage.

`780m-final` includes all five c11 4w picks even where another pick shadows them on the twelve production
shapes: the later entries can serve other aligned shapes. For example, the fifth pick serves N>=1024,
3072<=K<4096 with N<2048. The inherited wide row can still serve M=128 where a preceding 256-row pick
fails. Its buffer-weight table fallback remains. The 780M table 8da4w row is always replaced by zpg_bt
when the table serves: same alignment and storage, smaller subgroup grid with the same C drain rows.
It therefore is not an extra 780M origin, though the same kernel is a candidate via RX 7600 / XTX.

RX 7600's outer predicate is exactly the twelve NK pairs in `Overrides.cpp:2907`; it does not require
M=2048, so retain both its shape predicate and tile alignment. Shapes outside this set keep the table.
Its full historical attention environment is `ET_VK_SARC_780M_PROFILE=c7` plus
`ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko`.
These controls must be identical across later linear comparison arms. Do not add `780m-final` or c11,
which would add unintended linear choices.

XTX refine5 retains the original zpg table row on 3B wk/wv (N=1024,K=3072) and 8B w2
(N=4096,K=14336), as well as buffer IO and shapes outside its exact picks. Refine6 was rejected and is
not final. The supplied proposal requires RADV first; the historical AMDVLK shared-memory overrun is
not permission to run an illegal pipeline on either driver.

B580 has no hidden 4w refinement inherited from the old fused profile: its order profile explicitly
contains attention and 8da4w entries only. B70's k16 preference has texture3d only; buffer IO retains the
Intel 4w table row. The texture3d table 4w row is shadowed (same k16 alignment, smaller C drain in the
preference); it is not a B70 origin. Both Xe2 devices keep the 256x64k32 table 8da4w row for shapes the
128x128k64 order variants cannot fit, for example N=64 or group_size=32.

For Orin the earlier table must serve `orin_measured`: twelve NK projections, group size 128 and
M<=2048 (`table_nvidia.cpp:35`), texture3d IO. This condition is not removed by its profile. For 4w,
the table's large-K fp32 row and 256-row fp16 row are fully replaced wherever they fit; the 128-row
fp16 row survives M=128 (and the M=256, N%2048!=0 exception). For row-major 8da4w, the runtime's
unaligned fallback calls `select_table` directly (`Dq8caCoopmat.cpp:94-108`), so the shipped g44
zpgtr row remains reachable even though the ordinary aligned profile chooses bf g24.

M51 c8 supersedes the old c3/c5 K=4096 bx pick. The original k16 4w table row remains reachable when
bz k32 cannot fit, including group_size=16 and buffer weights. c8 switches aligned 8da4w graphs from
row-major to 4h4w. Once built, the activation layout is fixed: `select_m51:3329` reopens the layout only
to check service availability, then enforces the actual layout on the pick. This is not conversion at
runtime. For unaligned M with a 4h4w graph it falls back to stock tiled; a graph built row-major can
retain the table zpgtr kernel. Private evidence and compiler dependence are in the packet.

## Non-cooperative fallback boundary

`Q4gswCoopmat.cpp:69-111` chooses `linear_q4gsw_tiled` for M>1 if no SARC choice fits; its cooperative
single-token fallback is decode and outside this inventory. `Dq8caCoopmat.cpp:94-137` retains a row-major
table choice when needed; otherwise it uses `linear_dq8ca_q4gsw_tiled` for prefill. A graph for which
`builds_on_sarc` is false stays on the pre-existing op path entirely (`Select.cpp:167`,
`Q4gswCoopmat.cpp:137`, `Dq8caCoopmat.cpp:211`). Those upstream paths are controls, not additional
WMMA candidates. Bias, fp32, batch!=1, unsupported storage and unaligned cases must continue to be
handled correctly; this stage changes none of them. Decode gate requirements of future measured runs
remain in force; no decode tuning is proposed.
