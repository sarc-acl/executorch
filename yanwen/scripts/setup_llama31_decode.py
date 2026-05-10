#!/usr/bin/env python3
"""
Export LLaMA 3.1 8B fp16 for *real* autoregressive decode (use_kv_cache=True,
static seq_len=1, fixed max_seq_len for KV cache size).

The exported .pte's forward signature is (tokens: [1,1] int64, input_pos: [1] int64),
matching the standard ExecuTorch llama runner convention. KV cache is allocated
as in-graph state buffers, written to position `input_pos` on each forward.

Usage:
    source /home/doremy/sarc-acl/executorch/main/executorch/.venv/bin/activate
    sudo swapon /swapfile
    python yanwen/scripts/setup_llama31_decode.py --n_layers 32 --max_seq_len 1024
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from torch.export import export

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import run_llama31_pure as base  # noqa: E402 — reuses env_check, OOM hardening, etc.


WEIGHTS_DIR = Path("/home/doremy/llama3_1_8b/original")
CKPT = WEIGHTS_DIR / "consolidated.00.pth"
PARAMS = WEIGHTS_DIR / "params.json"
DEFAULT_OUT = Path("/home/doremy/llama31_decode_run")


class DecodeWrapper(torch.nn.Module):
    """
    Wraps the LLaMA Transformer so its forward takes (tokens, input_pos) as
    positional args, returning just the logits tensor.

    torch.export doesn't handle dict kwargs cleanly through the Vulkan
    delegate's lowering path; this flat-positional signature is what the
    standard llama runner expects (see examples/models/llama/runner/native.py).
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, tokens, input_pos):
        result = self.transformer(tokens=tokens, attn_options={"input_pos": input_pos})
        # forward returns either logits or (logits, attn_options_update) depending
        # on whether the cache yielded an update. Decode path returns the tuple
        # since the cache gets updated each step.
        return result[0] if isinstance(result, tuple) else result


def load_decode_model(n_layers: int, max_seq_len: int):
    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs

    with open(PARAMS) as f:
        params = json.load(f)
    original_layers = params.get("n_layers", 32)
    if n_layers < original_layers:
        print(f"[decode] subsetting layers: {n_layers} of {original_layers}")
        params["n_layers"] = n_layers

    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_context_len=max_seq_len,
        max_batch_size=1,
        use_kv_cache=True,
        enable_dynamic_shape=False,  # static [1,1] input + fixed max_seq_len cache
        **params,
    )

    with torch.device("meta"):
        transformer = construct_transformer(model_args)

    print(f"[decode] mmap-loading checkpoint {CKPT}")
    t0 = time.perf_counter()
    checkpoint = torch.load(CKPT, map_location="cpu", mmap=True)  # noqa: TOR102
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    print(f"[decode] checkpoint open in {time.perf_counter()-t0:.1f}s")

    if n_layers < original_layers:
        filtered = {}
        for k, v in checkpoint.items():
            if k.startswith("layers."):
                idx = int(k.split(".")[1])
                if idx < n_layers:
                    filtered[k] = v
            else:
                filtered[k] = v
        checkpoint = filtered

    missing, unexpected = transformer.load_state_dict(
        checkpoint, strict=False, assign=True
    )
    miss_w = [k for k in missing if k.endswith(".weight")]
    if miss_w:
        print(f"[decode] WARNING missing weights: {miss_w[:3]}...")

    transformer = transformer.half().eval()
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"[decode] params: {n_params/1e9:.2f}B fp16 ({n_params*2/1e9:.1f} GiB)")
    return transformer, model_args, checkpoint


def export_decode_pte(n_layers: int, max_seq_len: int, out_dir: Path):
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    tag = f"llama31_8b_{n_layers}L_decode_max{max_seq_len}_fp16"
    pte_path = out_dir / f"{tag}.pte"
    if pte_path.exists():
        print(f"[decode] cached: {pte_path} ({pte_path.stat().st_size/1e9:.2f} GiB)")
        return tag, pte_path

    out_dir.mkdir(parents=True, exist_ok=True)

    transformer, margs, checkpoint = load_decode_model(n_layers, max_seq_len)
    model = DecodeWrapper(transformer).eval()
    example_tokens = torch.randint(0, margs.vocab_size, (1, 1), dtype=torch.int64)
    example_input_pos = torch.tensor([0], dtype=torch.int64)
    example_inputs = (example_tokens, example_input_pos)

    print("[decode] torch.export(strict=False) tokens=[1,1], input_pos=[1]")
    t0 = time.perf_counter()
    with torch.no_grad():
        prog = export(model, example_inputs, strict=False)
    print(f"[decode] torch.export done in {time.perf_counter()-t0:.1f}s")

    del transformer, model, checkpoint
    gc.collect()

    print("[decode] to_edge_transform_and_lower (stock VulkanPartitioner)")
    t0 = time.perf_counter()
    edge = to_edge_transform_and_lower(
        prog,
        compile_config=EdgeCompileConfig(_skip_dim_order=False),
        partitioner=[VulkanPartitioner({})],
    )
    et = edge.to_executorch()
    print(f"[decode] lowered in {time.perf_counter()-t0:.1f}s")

    # OOM-safety: release prog + edge before writing the ~16 GB et.buffer
    del prog, edge
    gc.collect()

    print(f"[decode] writing .pte -> {pte_path}")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    print(f"[decode] .pte size: {pte_path.stat().st_size/1e9:.2f} GiB")
    return tag, pte_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layers", type=int, default=32, choices=range(1, 33))
    ap.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="KV cache capacity (also the max decode position).",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-swap-check", action="store_true")
    args = ap.parse_args()

    base._parent_oom_hardening()
    base._ensure_venv_path()
    base.env_check(args.skip_swap_check)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gc.collect()
    export_decode_pte(args.n_layers, args.max_seq_len, args.out_dir)
    print(f"\n[setup] artifacts in {args.out_dir}/")


if __name__ == "__main__":
    main()
