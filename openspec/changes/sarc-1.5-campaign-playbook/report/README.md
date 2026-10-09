# The tuning overview report

What this directory is for: `tuning-overview.html` is the owner's one-page view of every prefill campaign:
the final gain per device, the chain of accepted candidates that produced it, the technique-by-device matrix,
where the GPU time went, what did not pay, and where the tuned kernels stand against llama.cpp. It is kept here
so that it stays current as campaigns close, and it is published as a private page from this file (the owner
holds the link). The page's text is Chinese, for the owner; this file and the data keys are English.

## How the file is built

One HTML file, two parts, deliberately separated:

1. **The data block**: `<script id="report-data" type="application/json">` at the top of the script section.
   Every number on the page is in it. Keys:

   | key | what it holds | source of truth |
   |---|---|---|
   | `devices` | one row per device: name, vendor, `final` (geomean gain over the September table, in %; `null` for a device without published figures), `stock` (speed-up over unmodified ExecuTorch, before and after; `null` when not measured), `base`, optional `note` | `results.md` section 1 |
   | `cats` | the six technique categories of figure 1 with their fixed colors (`--s1` to `--s6`; never reorder, never add a seventh) | this file |
   | `chains` | per device, the accepted candidates in order: `c` (category key), `g` (geomean gain over its parent, %), `n` (label) | each campaign's `proposal.md` candidate table, accepted rows only |
   | `reported` | per device, the final stack measured directly against the pristine parent, % | each campaign's final session |
   | `techs` | the ten rows of the matrix; per device a cell `{k, g, v, n, t}`: `k` is `ok`, `tried`, `inherit`, `pending` or `none`; `g` the gain when adopted; `v` a short value when there is no gain to show; `n` a sub-label; `t` the tooltip | `proposal.md` of every campaign |
   | `techs[].ex`, `primer` | the plain-language section before figure 3: per technique `kind`, `what`, `why`, `like` (an analogy), `res` (result in one sentence); `primer` is the list of background terms. Update `res` when a technique lands on a new device | this file; `proposal.md` of the campaign that introduced the technique |
   | `fams`, `timeRows` | kernel families and the before/after dispatch time of 8B cells, ms | `trace-families.csv` / `STATUS.md` traces |
   | `waste` | the searches: hours of device time and end-to-end gain | `LESSONS.md` L6, the campaign's `STATUS.md` |
   | `llama` | tuned 4w tok/s divided by llama.cpp at its best setting, per model, Vulkan and vendor backend | `sarc-1.5-llamacpp-compare/results/cells.csv` |
   | `meta.updated` | the date of the last data change | |
   | `tokps` | per device, per model: `stock4w`, `stock8`, `sarc4w`, `sarc8`, `tuned4w`, `tuned8`, `vk` (best llama.cpp Vulkan Q4_0), `vkk` (best Vulkan Q4_K_M), `vendor` (best SYCL / CUDA Q4_0), `etcuda` (ExecuTorch CUDA 4w, 4070 Ti only), tok/s medians | `sarc-1.5-llamacpp-compare/results/cells.csv`: the max over that backend's `best` and `default` arms per model |
   | `branches` | the lane diagram of figure 6: one entry per branch with its lane geometry (`x0`, `x1`, `fork`, `nodes`), head hash, date, state (`ok`, `run`, `ext`, `base`) and chip text | `git ls-remote origin` and the campaign STATUS files; update heads and chips when a branch moves or a run ends |
   | `roofs` | figure 8, kernel rate against the hardware roofs, for the five devices of the roofline study (780m, b580, b70, 4070ti, orin; a device that was not studied has no entry and is absent from the figure). `study`: campaign id, date, and per device the igpu-roofline branch name and short head. `order`: row order of the figure. `schemes`: `4w`, `8da4w`, `attn` (`plot: false` keeps the fused attention kernel in the data table only). `example`: the device whose chain is shown as the worked example. `devices.<id>`: `isa` (true when the driver's generated instructions were inspected) with `isa_note`, `state` (driver, power and clock state), `width` (subgroup width of roof shaders against kernels), `weight` (how the per-model kernel figure was formed), and `rows.<scheme>`: `unit`, `reg` (register roof), `fed` (fed from shared memory, best reuse), `reuse` (fed at the kernel's own reuse; `null` when no usable row exists, with the reason in `reuse_note`), `kernel` (rate per model, in the order of `models`), `mma` / `in` / `acc` / `sg` (matrix shape, input and accumulator type, kernel subgroup), `name` (dispatched kernel, storage suffix dropped) and `also` (other tiles dispatched by shape). `devices.<id>.chain`: part D of that device's study for its lowest scheme: `steps` with `rate`, `loss` (relative to the previous step unless `loss_basis` is `points`, then points of the register roof) and, where the study states it, `share` of the whole distance | `efficiency.csv` and `STUDY.md` on the igpu-roofline branches `study/et-20261010-<device>` |

2. **The rendering code**: the second `<script>`. It reads the block, draws inline SVG, builds the tooltips
   and the data tables. It has no numbers in it. Change it only for a new figure or a layout fix.

**Devices without published figures, and the column count.** A device whose owner forbids publishing figures
(M51) is a `devices` entry with `"final": null`, a `base` text that says so and a `note` for the tooltip. It has
no entry in `chains`, `reported`, `timeRows`, `llama` or `tokps`, and its `techs` cells carry only `k`, a short
text `v` (for example "已采纳", "不适用"), an optional `n` and a qualitative tooltip `t`: no percentage, speed,
time, driver or board identifier anywhere, in the data block or in the prose. The rendering code shows the text
"不公布数字" on its tile and in the matrix header and draws no bar for it in figure 1. A device that has figures
but no llama.cpp or stock measurement (RX 7900 XTX) keeps those `tokps` fields `null`, which the table prints as
a dash, and has no `llama` entry; an optional `note` on a `devices` or `tokps` entry is shown in the tile tooltip
or beside the device name in figure 7. The matrix of figure 2 takes its column count from the data: the six
original devices in their fixed order, then any further `devices` entries in the order they appear; the script
sets the CSS variable `--ncol` on the matrix and the grid and its `min-width` follow it, so a new device needs
no style change.

Colors, type and theme tokens are in the `<style>` block and follow the validated default palette of the
data-viz method (six categorical slots in fixed order, one sequential blue ramp for the matrix, neutral gray for
"tried, not adopted"). Do not add a seventh categorical color; fold a new technique into an existing category
or make it a matrix row only.

## Updating after a campaign closes or a branch changes

The update is a data edit, not a redesign. Do it with a subagent so that the reading of six branches does not
fill the coordinating session's context; the subagent needs only this directory and the branches.

Prompt for the subagent (fill the angle brackets):

```
Update openspec/changes/sarc-1.5-campaign-playbook/report/tuning-overview.html in the worktree
<path to topic/campaign-playbook/executorch> after <what changed: a campaign closed / a branch was re-gated / a
new device>. Edit ONLY the JSON inside <script id="report-data">; do not touch the rendering script or the
styles. Read README.md in that directory first for the meaning of each key.

For each affected device, read on its branch (fetch origin first; read with `git show origin/<branch>:<path>`,
do not check anything out): openspec/changes/sarc-1.5-<tag>-prefill-refine/proposal.md (candidate table and
final stack) and STATUS.md (final session, traces). Take geomean gains as printed there; recompute from
results/<tag>/sessions/<final>/runs.csv when the two disagree and say which you used.

Then: 1. update `devices.final`, `reported`, `chains`, the device's column in `techs`, `timeRows` if a
trace exists, the device's `tokps` entry from cells.csv, and the branch's head and chip in `branches`; add a device as a new entry in `devices` (keep the order by final gain), `chains`, `reported`, and
a new key in every `techs[].cells`; 2. set `meta.updated`; 3. verify the JSON parses
(`python3 -c 'import json,re,sys; s=open(sys.argv[1]).read(); json.loads(re.search(r"report-data\" type=\"application/json\">\n(.*?)\n</script>", s, re.S).group(1))' <file>`)
and that the rendering script still parses (`node --check` on the second script block); 4. grep the file for
host names, user names and home paths and remove any; 5. report back, in at most 15 lines: which numbers
changed from what to what, with the source line of each, and anything you could not find.
Do not commit, publish or push.
```

After the subagent reports: read its diff (`git diff` of the one file), commit it on `topic/campaign-playbook`
with a message that names the campaign and the numbers that moved, and republish the page to its existing
artifact URL from this file (the Artifact tool with `url` set; a publish without `url` makes a new page and loses
the link). Push on the owner's word.

## Updating the roofs of figure 8

`roofs` is filled from the roofline study, not from the campaign branches. For each studied device read, on the
igpu-roofline branch `study/et-20261010-<device>` (or the branch of a later study), the device's `STUDY.md` and
`efficiency.csv` under its results root (27 rows: 24 linear rows = 3 models x 4 shapes x 2 schemes, plus 3
attention rows).

- `reg`, `fed`, `reuse`: columns `roof_register`, `roof_fed_shared`, `roof_fed_reuse` of the scheme's rows, checked
  against the roof table of `STUDY.md`. Set `reuse` to `null` and write the reason in `reuse_note` when the study
  says there is no matching roof, when the nearest row is far from the kernel's reuse, or when the row reads
  below the kernel (the study then says it is not a ceiling). Say in `reuse_note` when a row is a single run.
- `kernel`: one rate per model. Where `STUDY.md` states a weighting (780M and B580: time-weighted over a layer's
  seven linear calls, i.e. the sum of `rate x kernel_ms` over the sum of `kernel_ms` with `wq_wo`, `wk_wv`,
  `w1_w3` counted twice and `w2` once), use it; where it gives per-shape rows only (B70, 4070 Ti, Orin), take
  the plain mean of the model's four shapes and say so in `weight`. Attention: the one row per model.
- `chain`: copy part D as written, including its basis (which model or mean) and its own loss convention.
- Update `study` (campaign id, date, branch heads) and the date in the header line and in the caption of
  figure 8. The claim sentence and the caption of figure 8 quote a few of these numbers in prose; reread them.
- Never copy a results path into the report: the directory names of the study contain host names.
- A new studied device is a new entry under `roofs.devices` and in `roofs.order`; the rendering code needs no
  change. Do not add a device whose owner forbids publishing figures.

## What the page must keep saying

- Every gain is a geometric mean over 1B, 3B and 8B unless a cell says otherwise, and the figure 1 segments are
  gains over each candidate's own parent, so they multiply, not add.
- A device whose attention kernels and softmax could not be measured apart (RTX 4070 Ti SUPER, Jetson Orin
  Nano) shows the softmax as "within row 1" in the matrix; do not invent a split.
- A technique that was tried and not adopted stays on the page as hatched, with its number. Negative results
  are results.
- The Radeon 780M adopted `fused3sb` in round 3 (2026-10-09); its cell is "ok" with no gain (speed unchanged).
- Every Jetson Orin Nano number is for the 15 W power mode (GPU clock 612 MHz). The higher mode was not measured
  (owner decision 2026-10-09); do not present the expected gain of that mode as a result.
- Figure 8: the NVIDIA devices are not ISA-verified; the fed-at-reuse mark is a yardstick, not a strict ceiling
  (some shapes on B580 and B70 read slightly above it); the three company-side devices were not studied.
- The RX 7900 XTX fused-kernel result stands as measured under AMDVLK, subgroup width unverified; it will not
  be re-measured under RADV (owner decision 2026-10-09).
- The Jetson Orin Nano's second-round numbers come from a cross build whose `glslc` differs from the pinned one; the owner accepted them as measured (2026-10-09). Keep that note in the Orin tooltips.
