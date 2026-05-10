# yanwen — pure LLaMA 3.1 8B prefill on AMD 780M iGPU (L=32)

Scope: benchmark stock LLaMA 3.1 8B fp16 prefill on the 780M with the
ExecuTorch Vulkan delegate, using `VulkanPartitioner({})` (no coopmat,
no shader overrides). **L=32 only.**

Read in this order:

1. [`REPORT.md`](REPORT.md) — findings, breakdown, recommendations.
2. [`L32_S128_shader_breakdown.md`](L32_S128_shader_breakdown.md) —
   per-GLSL-shader breakdown for the only usable config (S=128).
   Maps each runtime kernel name to its source `.glsl` + `.yaml` and
   decodes the `(STORAGE, WEIGHT_STORAGE, DTYPE)` variant.
3. [`INSTRUCTIONS.md`](INSTRUCTIONS.md) — how to run the bench / capture
   ETDump / regenerate breakdowns. Written for AI agents (terse, exact
   commands).
4. [`scripts/`](scripts/) — Python scripts: bench, export, analyzers.
5. [`artifacts/L32/`](artifacts/L32/) — symlinked .etdp / .events.tsv /
   memprobe / logs for each tested seq.
6. [`old/`](old/) — superseded reports (kept for context).
