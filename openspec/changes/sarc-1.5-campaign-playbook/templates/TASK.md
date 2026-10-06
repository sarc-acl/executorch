# Campaign: <device name> (device tag `<tag>`)

<!-- How to use this template (delete this comment in the filled-in copy).
     One copy per device, filled in by the coordinator. Replace every <...>. Delete nothing: write "none" or
     "not known, find out and state it" instead. The filled-in file is the LAST part of
     <campaign-root>/CAMPAIGN.md on the host, after RULES.md, LESSONS.md and OWNER-DECISIONS.md (PLAYBOOK.md,
     step 3), so that a decision appended to the end of that file lands under "Owner decisions" below.
     Do not commit a filled-in copy that contains a real host name, user name, absolute path, address, serial
     number or lock id to the public repository: the filled-in task lives on the host and in the coordinator's
     private notes. -->

## 1. Device and driver

- Device: <device name>, <memory> GB, <discrete | integrated | shared memory>.
- Driver: <driver name and version>. Vulkan device index: `ETVK_DEVICE_INDEX=<n>` (<what the other indices
  are; "never measure on them">).
- Cooperative-matrix shapes the device exposes: <fp16 MxNxK list; int8 list | "query them and state them">.
  Subgroup sizes: <min to max, default>. Shared-memory limit: <bytes>.
- Clock policy as found (governor or performance level): <value | "record it before the first session">. Do
  not change it.

## 2. Host facts

- Host: `<host>` (<bare metal | virtual machine with the card passed through | build host that reaches the
  device over ssh>). Login shell: <bash | other: run every script with bash>.
- Working copy: `<campaign-root>/executorch`. Artifact directory: `<artifact-dir>`.
  `SARC_MOUNT_ROOT=<campaign-root>`. Free disk there: <GB>. Write nothing large anywhere else.
- Device lock: `~/.cache/gpu-lab/lock-<lock-uuid>` (create it writable; pass `<lock-uuid>` to `verify.sh --lock`).
- Sensors: temperature from <path or tool>; clock from <path or tool>; power from <path or tool | none>;
  per-client engine time <available through ... | not available>. Sampling period to use: <ms>.
- Container tool: <podman | docker: put a `podman` wrapper that execs `docker` on PATH under the artifact
  directory; do not edit `build.sh`>. Build image `localhost/et-vk-build:rocky10`: <present | build it from the
  sibling's `tools/Containerfile`>.
- Models: `<model-dir>` (<layout: `<model>/exported/*_vulkan_{4w,8da4w}.pte` | flat>), tokenizer `<path>`.
  Read-only. Context length and sizes checked against `kit/MODELS.md` on <date>: <result>.
- RAM: <GB>. <"The six model files do not all fit in the page cache: the first process after a model change is
  a slow load; fill the cache before each cell (owner decision D5)." | "fits">
- Services stopped for this campaign, by the owner, on <date>: <unit names | none>. They are still enabled, so
  a reboot brings them back. Restore with: `<command>`. You never start, stop or change a service; if one comes
  back and takes the GPU, stop measuring and report.
- `sudo`: <not permitted | permitted only for `<exact commands>`>.
- Other users of this host: <display and desktop | CI runners | other agents | a scheduled job at <time>: do
  not measure across it | none>.
- Other checkouts on this host that you never read, build or modify: <paths | none>.

## 3. Starting branch and parent

- Branch: `topic/<tag>-prefill-refine`, created from `origin/<sibling branch>` at `<parent>` (the same vendor's
  finished device; owner decision N5).
- Parent of every comparison: `<parent>` with <no profile | `ET_VK_SARC_UNVERIFIED=1` and no profile>.
- Change directory: `<change>` = `openspec/changes/sarc-1.5-<tag>-prefill-refine/`. Profile names
  `<tag>-refineN`. Sibling whose tools you copy: `openspec/changes/sarc-1.5-<sibling>-prefill-refine/tools/`.

## 4. What already exists for this device

| op | kernel row | where | state |
|---|---|---|---|
| 4w linear | `<kernel name>` | `impl/sarc/table_<vendor>.cpp` | <verified | unverified | none> |
| 8da4w linear | `<kernel name>` | same | <...> |
| attention QK^T / attention x V | `<kernel names>` | same | <...> |
| softmax | `<stock | SARC truncated>` | | <...> |

- Published numbers to compare the baseline with (tok/s, `prompt_2048.txt`): 4w 1B / 3B / 8B about
  <a> / <b> / <c>; 8da4w about <d> / <e> / <f>. Source: `<file>`. Measured with: <commit, toolchain, driver>.
- Known from earlier work on this device (re-measure before relying on it): <percent of roof per kernel;
  share of attention in the prefill; known failures, e.g. "the fp16-accumulate 4w tile fails the production
  diff at K = 14336"; known negative results>.

## 5. Goal and expected range

Push 2048-token Llama prefill tok/s on this device as far as it will go, with kernels chosen for this device.

Expected: <"+20 to +30 % geometric mean: the device already has attention kernel rows and is in the 780M's
position" | "+45 to +65 %: the device still runs the stock attention">. This is an expectation taken from the
finished campaigns, not a target; a smaller result reported with its evidence is a valid outcome.

## 6. Work order

1. **Snapshot.** Build `<parent>` from an exported commit, check the golden, store its unmodified `verify.sh`
   output once as `s0-parent-verify`. No verification pass of the existing rows beyond this (owner decision N2).
2. **Baseline and A/A.** Six cells against the numbers of section 4, 3 % per cell, else explain before
   continuing. A/A session; calibrate the clock floor, the repeat count and the noise band; commit the
   thresholds in `proposal.md` before this session.
3. **Locate.** ETDump per operator family for the six cells; phase timing of the linear kernels.
4. **Port list**, in this order, each as one gated candidate:
   1. <fused attention kernel, from `<sibling>`>
   2. <fp32 softmax without the zero tail, through the softmax-variant hook (owner decision D4)>
   3. <linear kernel chosen per layer shape>
   4. <whole-texel 8da4w weight staging>
5. **Further candidates** only where step 3 points: <...>. No tile sweep of the linear kernels as a first
   move. No sampled parameter search (owner decision N1) <unless: ...>.
6. **Gate and timed session** for every candidate, as in the rules (R6, R7).
7. **Stop rule**: two consecutive gated candidates under 2 % geometric mean each.
8. **Final verification of everything together**, final session, push (R11).
9. **Coordinator hold** in every detached queue from the first one on (`tools/HOLD.md`); test it once and
   record it in `STATUS.md`.

## 7. Device-specific cautions

- <e.g. "The card drives the owner's display: record its other DRM clients and their engine time per run;
  derive the foreign-busy ceiling from the A/A session; 7 repeats if the A/A spread asks for it.">
- <e.g. "Timer resolution: a 1B prefill is about 100 ms and the runner's timer is 1 ms; report it and use
  ETDump dispatch time alongside.">
- <e.g. "If the device disappears or the vendor tool fails: STOP. Do not retry, reboot or reload drivers.
  Write what happened to STATUS.md and end the run.">
- <e.g. "The system driver has no cooperative-matrix support; a user-space driver build is selected per
  process with `VK_ICD_FILENAMES=<path>`; export it for both arms and record the driver commit.">
- <e.g. "Shipped kernel `<name>` is shared with <other device>: its SPIR-V must not change.">
- Profiler tracing: allowed under rule R9 (owner decision N4). <Device-specific note, e.g. what the capture
  tool needs on this host; "untried on this device".>

## 8. Artifact directory

`<artifact-dir>`: `build/<tag>/` and `src/<tag>/` (one per exported commit), `stage/<session>/` (staged
binaries, `env`, raw runs, logs), `logs/` (detached chains and their status files), `venv/`, `tmp/`
(`TMPDIR`), `HOLD` / `HELD` (coordinator hold), `ABORTED` / `GPU_GONE` (markers). Nothing of this goes into the
working copy; small evidence goes to `<change>/results/<tag>/`.

## 9. What "done" means (reviewer)

Done only when all of these hold and you have checked them yourself in this round (rule R12):

- the stop rule of section 6.7 is met by two gated candidates whose numbers you recomputed from `runs.csv`;
- the final stack was verified together on the build of the committed branch head, with no local patch:
  unmodified `verify.sh` against the snapshot, attention tiers 12 passes with 0 mismatches, golden unchanged,
  reference-error evidence present for every arithmetic change and recorded as such;
- the final timed session against the pristine parent exists and its medians match `STATUS.md`;
- files changed outside the dev zone since `<parent>` are exactly the hooks owner decision D4 allows, each its
  own commit; nothing under `sarc/tools`, `sarc/golden`, no tolerance, prompt or threshold file changed;
- every new shader was read for unsynchronised shared writes, by you;
- `proposal.md` and `STATUS.md` are current, `check.sh --no-build` output is reported, the branch is pushed;
- your not-checked list contains none of the above.

## Owner decisions

<!-- Later decisions of the owner for this device are appended here by the coordinator, newest last, each as
     "### Owner decision, <date> (<UTC time>): <title>", with what is decided, why, and what does not change.
     A resumed actor has no memory: a decision that is not here does not exist for it. -->

(none yet)
