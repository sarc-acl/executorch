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
   | `devices` | one row per device: name, vendor, `final` (geomean gain over the September table, in %), `stock` (speed-up over unmodified ExecuTorch, before and after), `base` | `results.md` section 1 |
   | `cats` | the six technique categories of figure 1 with their fixed colors (`--s1` to `--s6`; never reorder, never add a seventh) | this file |
   | `chains` | per device, the accepted candidates in order: `c` (category key), `g` (geomean gain over its parent, %), `n` (label) | each campaign's `proposal.md` candidate table, accepted rows only |
   | `reported` | per device, the final stack measured directly against the pristine parent, % | each campaign's final session |
   | `techs` | the ten rows of the matrix; per device a cell `{k, g, v, n, t}`: `k` is `ok`, `tried`, `inherit`, `pending` or `none`; `g` the gain when adopted; `v` a short value when there is no gain to show; `n` a sub-label; `t` the tooltip | `proposal.md` of every campaign |
   | `fams`, `timeRows` | kernel families and the before/after dispatch time of 8B cells, ms | `trace-families.csv` / `STATUS.md` traces |
   | `waste` | the searches: hours of device time and end-to-end gain | `LESSONS.md` L6, the campaign's `STATUS.md` |
   | `llama` | tuned 4w tok/s divided by llama.cpp at its best setting, per model, Vulkan and vendor backend | `sarc-1.5-llamacpp-compare/results/cells.csv` |
   | `meta.updated` | the date of the last data change | |
   | `tokps` | per device, per model: `stock4w`, `stock8`, `sarc4w`, `sarc8`, `tuned4w`, `tuned8`, `vk` (best llama.cpp Vulkan Q4_0), `vkk` (best Vulkan Q4_K_M), `vendor` (best SYCL / CUDA Q4_0), `etcuda` (ExecuTorch CUDA 4w, 4070 Ti only), tok/s medians | `sarc-1.5-llamacpp-compare/results/cells.csv`: the max over that backend's `best` and `default` arms per model |
   | `branches` | the lane diagram of figure 6: one entry per branch with its lane geometry (`x0`, `x1`, `fork`, `nodes`), head hash, date, state (`ok`, `run`, `ext`, `base`) and chip text | `git ls-remote origin` and the campaign STATUS files; update heads and chips when a branch moves or a run ends |

2. **The rendering code**: the second `<script>`. It reads the block, draws inline SVG, builds the tooltips
   and the data tables. It has no numbers in it. Change it only for a new figure or a layout fix.

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

## What the page must keep saying

- Every gain is a geometric mean over 1B, 3B and 8B unless a cell says otherwise, and the figure 1 segments are
  gains over each candidate's own parent, so they multiply, not add.
- A device whose attention kernels and softmax could not be measured apart (RTX 4070 Ti SUPER, Jetson Orin
  Nano) shows the softmax as "within row 1" in the matrix; do not invent a split.
- A technique that was tried and not adopted stays on the page as hatched, with its number. Negative results
  are results.
- The Radeon 780M's `fused3sb` cell stays "pending" until the branch adopts it.
