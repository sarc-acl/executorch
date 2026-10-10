# linexp-1 / b580-reuse: ExecuTorch side

Experiment, not a campaign: nothing here goes into a campaign branch or report. The running record (`LOG.md`) and
the result page (`RESULTS.md`) are in the igpu-roofline clone of this run, `exp/linexp-1/b580-reuse/`.

- `tools/gen_exp.py`: writes the experiment kernels (`glsl/sarc_dev/sarc_exp_linear_dq8ca_coopmat_zpg*`,
  `impl/sarc_dev/exp/ExpLinear.cpp`). `sh` = fp32 totals in shared memory (correct); `i32` = int32 only
  (**ablation, incorrect by construction**). Reachable only with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_EXP=exp-<x>
  ET_VK_SARC_DQ8CA_VARIANT=<token>`.
- `data/h1t1-*.csv`, `isa/t1/`: compile-only counts of the tree's existing 8da4w rows. `data/h1t2-*.csv`,
  `isa/t2/`: the experiment kernels. Listings are committed as `.cs.log.xz`.
- `data/pdiff/`, `data/pdiff-summary.csv`: production diff (correctness). `data/t3/`, `data/t3-table.csv`: the
  timing sitting. `data/nosel/`, `data/nothing-selected-spirv.txt`, `data/exp1.*.txt`: the build and the "nothing
  selected" check. `data/logs/`: status files and desktop-share samples of every GPU job.
