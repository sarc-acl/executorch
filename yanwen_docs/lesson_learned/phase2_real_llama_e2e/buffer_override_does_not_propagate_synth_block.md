# `storage_type_override=BUFFER` does not propagate through a fp16 RMSNorm block

## What was attempted

For Phase 2 I exported a one-block LLaMA-3.1-shaped prefill graph with two
storage variants:

```python
# scripts/export_llama_block.py
options["storage_type_override"] = VkStorageType.BUFFER
options["memory_layout_override"] = VkMemoryLayout.TENSOR_WIDTH_PACKED
```

intent: produce a buffer-everywhere .pte so we can compare buffer coopmat vs
texture coopmat on the same single-block synthetic graph.

## What happened

After export, `executor_runner` reports every linear in the buffer .pte as
`is_buffer=0`:

```text
[VK_LINEAR] Using linear_vec (coop_mat=1, is_buffer=0, has_bias=0)   ×7
[VK_MATMUL] Using matmul_vec (coop_mat=1, is_buffer=0)                ×2
```

i.e. the override did not actually take effect inside the graph; runtime
still treats every linear/matmul output as texture3d. The buffer .pte and the
texture .pte therefore measure the same wallclock (~93 ms) — both go through
Stephen's texture path.

## Why it matters

This is exactly the propagation issue called out in
`yanwen_docs/background/1_previous_story.md` ("Approach A — patch the
partitioner" and "Approach B — rewrite the shader for texture3d I/O"). The
synthetic block I built uses an fp16 RMSNorm that does

```python
def forward(self, x):
    v = x.to(torch.float32)
    rms = ((v.pow(2).mean(-1, keepdim=True)) + eps).sqrt()
    return (v / rms).to(x.dtype) * self.weight
```

The two `_to_dim_order_copy` casts inside RMSNorm act as
buffer/texture3d transition points. The Vulkan partitioner's
`tag_memory_meta_pass` re-promotes the linear inputs/outputs to texture3d
through `sync_primary_io_repr`, regardless of the original
`storage_type_override`. Real LLaMA's buffer .pte does not exhibit this
behaviour because the embedding op (a CPU fallback) emits a buffer-tagged
output that the linear inherits, so buffer storage propagates through the
real LLaMA graph but not through this single-block synth graph.

## Concrete impact on Phase 2

- The synthetic-block buffer-vs-texture comparison cannot be used to study
  the buffer-storage tax in isolation — both arms come out texture3d.
- The real-LLaMA buffer .pte does propagate buffer storage to linears
  (`is_buffer=1` in routing) and is the only valid buffer-coopmat datapoint
  in this round.

## Recommended next action

If a future agent wants to study buffer storage on a synthetic block, the
front-end of the block must include something that forces the partitioner to
keep buffer storage — either an explicit `aten._to_copy` to fp16 with
`memory_format=contiguous_format`, or a CPU-fallback boundary equivalent to
real LLaMA's embedding. Alternatively, modify
`backends/vulkan/_passes/tag_memory_meta_pass.py` to honor
`storage_type_override` more strictly, but per
`yanwen_docs/background/1_previous_story.md` previous attempts at that
direction (Approach A v1/v2) regressed wallclock or partitioning correctness.

Either way, do not assume `storage_type_override=BUFFER` is sufficient — the
runtime routing log (`[VK_LINEAR] ... is_buffer=…`) is the source of truth.
