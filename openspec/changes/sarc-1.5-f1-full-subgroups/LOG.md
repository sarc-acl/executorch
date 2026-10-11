# sarc-1.5-f1-full-subgroups (Jetson Orin Nano): log

Task: `/mnt/linux-share/hmz-campaigns/f1-orin/TASK.md`. Branch `topic/f1-full-subgroups-orin` (from `da92b4b3c`),
not pushed. Device `duck-naughty` (Jetson Orin Nano, L4T R39.2.1, NVIDIA driver 595.78, Vulkan 1.4.329, 15 W mode as
found), measured over ssh from the workstation `fedora`, cross-built there. Artifacts
`/mnt/linux-share/hmz-campaigns/f1-orin/.artifacts/` (workstation) and `~/hmz-f1-orin/` (device).

## State

(2026-10-11 01:55 UTC) **Done; nothing is running** (device and workstation checked). Steps 1 to 7 exist with
their evidence (`RESULT.md`, `data/`). Verdict by the rule below: **no effect** in both sessions (`orin-fused1`:
geomean -0.05 %, largest cell -0.22 %; no environment: +0.00 %, largest cell +0.05 %; 60 of 60 runs valid in
each, neither repeated). Correctness as the old build in every item; 492 logits vectors and the SDPA records
identical; SPIR-V 0 differences. Open points a reader must know: the validation layer is not installed on the
device (flag proven by a debug print in an uncommitted build, no VUID count); at kernel level 8 decode-size
cases of the stock SDPA kernels read 17 to 25 % slower in the pre-registered 3-process table, which a check made
afterwards (not pre-registered) shows as level-to-level variation present in both arms, cause not measured; two
gate items fail as on the old build (buffer production-diff, SDPA status with no environment). The branch is
committed and not pushed.

## Decision needed from the owner

None.

## Pre-registration (written and committed before any timed run of this task)

Arms: old = build `topic4` (`0bed38090`, the fused-port campaign's final build, bundle copied unchanged), new =
build `f1` (`0f038a736`, same cross recipe, image and shader compiler as `topic4`). Sessions with the campaign's
`session.sh` / `e2e5.sh`, validity rules and clock floor unchanged (`thresholds.txt`, `results/orin/clkmin.json`:
593 MHz; 5 valid runs per arm and cell as in the campaign's closing session `s8-pristine`, arms interleaved, up
to 3 extra pairs a cell replace invalid runs). Per cell: median tok/s of the first 5 valid runs of each arm,
new / old.

- Session A (`sA-fused1`): old against new, both with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1`.
- Session B (`sB-noenv`): old against new, both with no environment.
- **"No effect"**: every cell's difference is inside the campaign's noise band (+-2 %) AND the geomean of the
  six cells is inside +-1 %, in a session. The verdict is given per session. Anything else is reported as an
  effect with its size and sign; nothing is tuned and nothing is worked around.
- A session that ends without 5 valid runs per arm in every cell, or that a tool aborts (foreign GPU process,
  busy lock), is repeated once under a new name and then reported as it is; a complete session is never repeated.
- Kernel level (`test_llama_microbench --linear --regime=prefill --skip-correctness`, both schemes, and `--sdpa`;
  3 processes per arm, arms interleaved, both environments): per case the median of the 3 process medians,
  new / old, with the same +-2 % / +-1 % reading. The verdict of the task is the sessions'; a kernel case outside
  the band is listed with its size beside them.
- Correctness expectation: each gate item of each configuration reads on the new build as on the old one, and
  wherever a tool dumps values (the 41-prompt last-position logits of `logits_dump`; the `[sdpa-error]` records
  of the SDPA correctness test, which does not dump its outputs) old and new are identical. A difference is a
  finding. An item the old build fails in the same way is recorded "as the parent".

## Record

### Step 1: the change (`0f038a736`)

`git am ../f1-full-subgroups.patch`: applied cleanly, no adaptation. `git show --stat`: `Adapter.cpp`,
`Pipeline.cpp`, `Pipeline.h`, `sarc/HOOKS`, +50 / -11.

### Found on the device before anything ran (15:40 UTC)

- `VK_LAYER_KHRONOS_validation` is NOT installed (`/etc/vulkan/explicit_layer.d` empty, no other layer directory,
  no validation package in `dpkg -l`). Step 3 uses the fallback (a debug print in a build that is never committed).
- `vulkaninfo`: `subgroupSizeControl = true`, `computeFullSubgroups = true`, `minSubgroupSize = maxSubgroupSize = 32`.
- No GPU process, GPU load 0, power mode 15 W, no `HOLD` / `ABORTED` / `GPU_GONE` of the campaign.

### Builds (workstation, 15:52 to 16:15 UTC, campaign's `build-orin.sh` from the tools copy, under `lock-desktop-build`)

- The campaign's tools hard-code its own roots; they run from `<artifacts>/tools`, a copy with two constants of
  `common.sh` changed (`tools/mk-tools.sh`, `data/tools.diff`). Device root of this task: `~/hmz-f1-orin`. The
  gpu-lab lock of the device is the campaign's.
- `topic4`: source tree and bundle copied from the campaign's artifacts (`cmp` equal; device copy has the same
  hashes as the campaign's device copy). `f1` = `0f038a736` on `topic4`'s tree (266 changed paths, 4 outside
  `openspec/`), rc 0, `spirv_golden.py` FAIL with `topic4`'s 14 DIFF lines (as the parent). `f1dbg` = `f1` +
  `data/flagproof/flagprint.patch` (not committed). `logits_dump` for `f1` (`build-extra.sh`) waited 15 minutes
  for the build lock, which another task's measurement held shared.
- Recipe hash of `f1.src.txt` differs from `topic4`'s: the campaign added the optional `GLSLC_DIR` switch after
  `topic4`; unset here. SPIR-V: 3240 files per build, 0 differences.
- Static check: 398 lines, 0 violators (`data/local-size.csv`).

### Device chain `chain1` (`tools/f1-chain.sh`, 16:27 UTC to 01:08 UTC; `data/chain.status`)

- Step 2, flag proof (`f1dbg`): 28 pipelines with a required subgroup size, all with stage flags 0x2; 40 without,
  all 0x0.
- Step 3, `sA-fused1` (`gate_sdpa.sh`, 16:33 to 18:36 UTC): `GATE_ACCEPTED`, all steps passed; session 60 of 60
  valid, geomean -0.05 %.
- Step 4, no environment (18:47 to 21:08 UTC): `sBn-noenv` `GATE_ACCEPTED` (0 findings, 0 differing lines);
  36 SDPA passes, each rc 1 with 0 mismatches (stock kernels; old build and parent control the same);
  `sB-noenv` `SESSION_ACCEPTED`, 60 of 60 valid, geomean +0.00 %.
- Step 5, kernel level (21:08 to 22:13 UTC): 36 processes; linear suites inside +-0.6 %; SDPA suite: 8
  decode-size cases outside the band (`RESULT.md` section 7).
- Step 6, logits (22:17 to 01:08 UTC): 246 vectors per configuration, 0 differ.
- No job was aborted; no `ABORTED` / `GPU_GONE`; no foreign GPU process seen.

### After the chain (01:09 to 01:46 UTC), not pre-registered

`decode1` (`tools/f1-decode.sh 10`): 40 processes of `--sdpa --regime=decode`, written after the kernel table
was seen, to tell an effect from variation in the 8 cases. It replaces nothing. Result in `RESULT.md` section 7.

### Rules

No `sudo`, no installation, no clock / power-mode / governor change, no service touched, the second Orin not
contacted. Nothing written to the campaign's working copy or artifacts (read and copied; on the device two
control sessions copied from `~/hmz-sarc-orin-fused/stage/` to `~/hmz-f1-orin/stage/`). One 10-second trial of
the decode filter wrote two small files to the device's `/dev/shm` and removed them. Not pushed.
