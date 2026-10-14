# Summary for the owner

One page: what was done in the five campaigns, the results, what worked, what did not, what is left. Details:
`results.md` (numbers and sources), `LESSONS.md` (what happened), `PLAYBOOK.md` (how to repeat it).

## What was done

Five agent-run campaigns, one per device, each an actor and a reviewer working on the GPU host, tuned the
Vulkan kernels of 2048-token Llama prefill (Llama 3.2 1B, 3.2 3B, 3.1 8B; 4w and 8da4w). Each campaign
re-measured the September release as its parent, calibrated noise from an A/A session, located the time,
gated every candidate with the unmodified `verify.sh` and a same-session timed comparison, and stopped when
two consecutive gated candidates gained under 2 %. Afterwards every device was compared with llama.cpp.

## Results (prefill tok/s, geometric mean over the three models)

| device | 4w gain | 8da4w gain | both | x stock, September -> now | state on 2026-10-07 |
|---|---:|---:|---:|---|---|
| Jetson Orin Nano | +65.7 % | +67.8 % | +66.7 % | 4.46 -> 7.43 | finished, pushed (corrected build) |
| Arc B580 | +46.0 % | +77.8 % | +61.1 % | 2.19 -> 3.53 | finished, pushed |
| Arc Pro B70 | +48.8 % | +70.7 % | +59.4 % | 2.17 -> 3.46 | finished, pushed |
| RTX 4070 Ti SUPER | +43.2 % | +49.2 % | +46.2 % | 3.57 -> 5.22 | finished, pushed |
| Radeon 780M | +29.8 % | +36.4 % | +33.1 % | 2.08 -> 2.76 | finished, pushed |

Gains are over the September SARC release; "x stock" is over unmodified ExecuTorch 1.5. All five
campaigns ended by the stop rule, were signed off by the independent reviewer and are pushed.

Against llama.cpp (its warm timer, best screened setting): tuned Vulkan kernels are 1.16 to 1.80 times
llama.cpp Vulkan on AMD and Intel and level on NVIDIA (0.93 to 1.21). Vendor backends are faster than tuned
4w (SYCL by 19 to 39 %, CUDA by 23 to 38 % on the RTX 4070 Ti SUPER, level on the Orin); tuned 8da4w is within
6 % of SYCL. ExecuTorch's own upstream CUDA backend ran 4w at 0.75 to 0.85 times stock Vulkan.

## What worked

- **Porting the attention kernels.** Cooperative-matrix QK^T and attention x V kernels, taken from the 780M and
  adapted per device: +46 to +58 % on the four devices that still ran the stock attention.
- **An fp32 softmax without the zero tail.**
- **Whole-texel weight reads in 8da4w**: +6 to +7 % on Intel, also on the Orin. Phase timing had shown the
  fetch costing more than the multiply.
- **On the 780M, which already had the attention kernels**: one fused attention kernel +13 %, fp32 softmax
  +4 %, a linear kernel per layer shape +3 %, an initial parameter refinement +8 %.
- **The method**: parent from an exported commit, same-session interleaved arms, thresholds fixed before the
  data, a parent `verify.sh` snapshot to compare against, detached queues with a status file, `STATUS.md` after
  every candidate. Work survived a power loss of the control machine and two ended hmz runs.
- **A reviewer that reads the code**: it found a shader data race that every test had passed.

## What did not work

- **Tile sweeps of the linear kernels**: nothing faster on any device.
- **The sampled parameter search** (B70, about 33 hours over four kernel families): 4w tiles 2 to 6 % faster
  per shape, +1.1 % end to end; nothing for 8da4w.
- **Enumerating the attention kernels' parameters**: faster kernels at kernel level on both devices that did it
  (B70 3 to 22 %, 780M 8 to 30 %), nothing end to end: +0.6 and +0.95 % on the B70 (inside the noise band, not
  adopted), and unused on the 780M because its fused kernel replaces those kernels.
- **A softmax that reads its row once** (Orin): slower.
- **Next-token equality as the gate for arithmetic changes**: replaced by your reference-error rule (D3).
- **Process failures**, each once: a campaign closed and pushed before the race was found; three gates spent
  on a runner abort before its cause (slow model load) was looked for; an RGP capture that hung a host; a
  recommended configuration reachable only through an uncommitted patch; a sampler that leaked 160 pollers and
  cost a session; a kill that stopped every campaign; a `HOLD` that was not removed.

## What is left

1. **Nothing is left of the five campaigns**: all closed between 2026-10-05 and 2026-10-07.
2. **Next devices**: Radeon RX 7900 XTX and Radeon RX 7600 (its campaign started on 2026-10-06 on
   `topic/rx7600-prefill-refine`: the fused attention kernel timed +18 % there), from `topic/780m-prefill-refine`, with your
   defaults N1 to N9: port the 780M's second layer, no sampled search, no verification pass first, expect +20
   to +30 %. Then the llama.cpp comparison on each.
3. **Merge** the five topic branches into `dev/1.5`: separate, later work, planned in `MERGE-PLAN.md`. The fused attention entry point
   awaits your review before any promotion (D4.3).
4. **Open points this package could not settle**:
   - how the Q4_0 GGUF files of the comparison were converted is recorded beside the files, on storage the
     next machine cannot reach; it has to make its own files and say so;
   - whether a vendor llama.cpp backend is to be measured on AMD cards;
   - the flow's new review prompt and retry waits, and the patrol script's ssh path, have not run in a
     campaign.
