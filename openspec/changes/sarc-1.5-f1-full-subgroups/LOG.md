# sarc-1.5-f1-full-subgroups (Jetson Orin Nano): log

Task: `/mnt/linux-share/hmz-campaigns/f1-orin/TASK.md`. Branch `topic/f1-full-subgroups-orin` (from `da92b4b3c`),
not pushed. Device `duck-naughty` (Jetson Orin Nano, L4T R39.2.1, NVIDIA driver 595.78, Vulkan 1.4.329, 15 W mode as
found), measured over ssh from the workstation `fedora`, cross-built there. Artifacts
`/mnt/linux-share/hmz-campaigns/f1-orin/.artifacts/` (workstation) and `~/hmz-f1-orin/` (device).

## State

(2026-10-10 18:45 UTC) **Running on the device: job `chain1` (`tools/f1-chain.sh`, started 16:27 UTC with
`<artifacts>/tools/drun.sh chain1 f1-chain.sh`; status `~/hmz-f1-orin/chain.status`, output
`~/hmz-f1-orin/jobs/chain1.out`).** Done so far: step 1 (`0f038a736`), builds `f1` and `f1dbg` (16:00 / 16:12 UTC),
static check (0 violators), SPIR-V identity (0 differences), flag proof (step 2 of the chain), session A
(`sA-fused1`: `GATE_ACCEPTED` 18:36 UTC, geomean -0.05 %). Still to run in the chain: session B, kernel level,
logits. Then `tools/collect.sh` and `RESULT.md`. Nothing runs on the workstation.

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
