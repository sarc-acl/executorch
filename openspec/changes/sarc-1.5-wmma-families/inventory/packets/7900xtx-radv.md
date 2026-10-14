# RX 7900 XTX: RADV prerequisite, then RDNA3 matrix

For owner relay with `COMMON.md`, `LEGALITY.md`, `manifests/rdna3-radv.json` and the XTX inventory rows.
No work is launched by this packet. The dependency is a reviewed manifest plus a verified RADV setup and
separately launched device task. Existing historical results are AMDVLK; they do not satisfy this prerequisite.

Use an already authorized/configured RADV installation. If absent, report the dependency to the owner;
this packet authorizes no installation, service, host-policy or driver change. Prove vendor and driver ID
and full property signature from the selected unit/ICD. In particular record the actual shared-memory limit
and exact fp16->fp32 and int8->int32 subgroup MMA shapes. Recompute legality for every candidate first:
source declares up to 61,440 bytes for the inventoried RDNA3 linears, and the afmb1 variant declares 60,160.
Do not run over-limit kernels because the old AMDVLK gate happened to pass them.

Before cross-device work, establish the pristine parent snapshot, six-cell baseline and A/A under RADV,
then the final campaign linear configuration `7900xtx-refine5` under RADV. Its historical complete profile
is `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5`; refine6 was not adopted.
Keep profile attention choices constant across arms; do not add the rejected 780M fused attention. Both
production attention controls and linear dispatch must be recorded. Explain differences from historical
AMDVLK baseline rather than treating it as a RADV number; no upstream-main baseline is requested.

The final linear control picks 4w 128x128k32 column-major B on three 1B shapes and retains the shipped
256x128 tile elsewhere. 8da4w picks four kernels by NK and retains the 128x64k32 table kernel on 3B wk/wv
and 8B w2. Preserve those fallbacks in the control. Exact predicates, variants, source refs and storage masks
are in the inventory. Only after RADV prerequisites pass, evaluate the reviewed legal RDNA3 union on this
unit, six cells, with unchanged attention and the R6/gate evidence in `COMMON.md`.

Return the RADV prerequisite evidence and legality exclusions separately from later cross-device results.
RX 7600/XTX numbers may be published with private infrastructure removed. No family winner is requested.
