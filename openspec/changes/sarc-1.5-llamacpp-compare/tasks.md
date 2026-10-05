# Tasks

## 1. Inputs and prior work

- [x] 1.1 Read the August 2026 llama.cpp preparation (scripts, logs, any results, the prompt file) and write
  what it contains into `kit/PRIOR-WORK.md`; verify by listing every file read and every number found
- [x] 1.2 Copy the F16 and Q4_0 GGUF files and their notes to shared storage; verify sha256 against the source
- [x] 1.3 Record for each ExecuTorch model file to be measured its context length, group size and embedding
  quantization (read from the file's metadata) in `kit/MODELS.md`; verify the three values per file are filled
- [x] 1.4 Pin the llama.cpp commit in `kit/VERSIONS.md` and produce Q4_K_M from the F16 files with it; verify
  each note beside the file names source revision, commit and exact command, and that bits per weight are computed

## 2. Kit

- [ ] 2.1 `kit/build-llamacpp.sh` (Vulkan, CUDA; build directory outside the repository; options recorded to
  a text file beside the binaries); verify it produces the completion binary and `llama-bench` on the workstation
- [x] 2.2 `kit/run-llamacpp.sh`: one run of the fixed prompt in a fresh process, tier selected by name, writing
  one JSON record (tokens, prompt-evaluation ms, commit, backend, device, settings); verify the record of a 1B
  run shows the expected token count
- [x] 2.3 Add llama.cpp arms to the session script (interleaving, lock, guard, clock and temperature
  sampling reused from the device's campaign tools); verify with a two-arm dry session on the 1B model that
  both arms' runs are recorded and judged
- [x] 2.4 `kit/aggregate.py`: medians, valid and invalid counts, ratios, bits per weight; verify it reproduces
  by hand-checked arithmetic the medians of the dry session
- [ ] 2.5 `kit/README.md`: how to run a device, the validity rules, the list of disclosed differences; verify
  the documented commands run as written on the pilot device

## 3. Pilot: Arc B580

- [ ] 3.1 Smoke test: 1B, 3B, 8B Q4_0 on llama.cpp Vulkan load, report 2048 prompt tokens and produce sensible
  text; verify from the three run records
- [x] 3.2 Settings screen on 1B for the `best` tier; verify the chosen options and the screen table are
  written to `results/b580/screen.csv` before 3.4 starts
- [x] 3.3 Write down the arms of this device (ExecuTorch stock, sarc, tuned with commits; llama.cpp Vulkan
  Q4_0 and Q4_K_M in three tiers) in `results/b580/ARMS.md`; verify it is committed before 3.4
- [ ] 3.4 Timed session, all arms, three models, both ExecuTorch schemes; verify five valid runs per cell and
  that the ExecuTorch sarc arm is within 3 % of `cells.csv` (or the difference is explained)
- [ ] 3.5 `llama-bench` cross-check for every llama.cpp cell; verify agreement within the threshold fixed in 3.2
- [ ] 3.6 Deliver the B580 table and chart to the owner privately; verify every number against the raw records

## 4. SYCL on Intel

- [ ] 4.1 Container image with the oneAPI toolchain and a SYCL build of the pinned commit; verify the GPU is
  visible inside the container (device listing) or record why not
- [ ] 4.2 Add the SYCL arms on the B580 and measure them in a session with the Vulkan `best` arm as the common
  reference; verify five valid runs per cell, or mark the cells "not run" with the reason

## 5. RTX 4070 Ti SUPER (after its campaign has pushed)

- [ ] 5.1 Build llama.cpp Vulkan and CUDA at the pinned commit; verify both binaries report the device
- [ ] 5.2 ExecuTorch CUDA: upstream plus the export-guard removal only; export 1B, 3B and 8B `4w` and build
  the runner; verify each takes the 2048-token prompt (prompt token count in the run record)
- [ ] 5.3 Arms written down in `results/4070ti/ARMS.md`, then the timed session as in group 3; verify as 3.4
  and 3.5, with the CUDA `8da4w` cells reading "not supported"
- [ ] 5.4 Deliver the table to the owner privately; verify against the raw records

## 6. Arc Pro B70, Radeon 780M, Jetson Orin Nano (each when its device is free)

- [ ] 6.1 Arc Pro B70: Vulkan and SYCL arms, session, delivery; verify as group 3
- [ ] 6.2 Radeon 780M: Vulkan arms, session, delivery; verify as group 3, and record the memory of the 8B cells
- [ ] 6.3 Jetson Orin Nano: Vulkan and CUDA arms of llama.cpp, session, delivery; the ExecuTorch CUDA cells
  read "not supported"; verify as group 3, and record the memory of the 8B cells

## 7. Whole comparison

- [ ] 7.1 `results/cells.csv` for all devices and the private page with one table and chart per device and the
  disclosed differences; verify a second reader recomputes three cells per device from the raw records
- [ ] 7.2 Record what the owner released for publication and what stays private in `results/RELEASE.md`;
  verify nothing else of this change is in a published location
