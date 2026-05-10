#!/usr/bin/env python3
"""
Coopmat variant of run_llama31_pure.py — same methodology, but:
  - executor_runner from pavan-report tree (linear_coopmat shaders compiled in)
  - VulkanPartitioner exports with storage_type_override=BUFFER, which forces
    linear/matmul outputs to buffer storage. At runtime that triggers
    add_linear_coopmat_node() on the 780M (KHR cooperative matrix supported,
    M >= 64). lm_head [1, 128256] has M=1 → falls back to linear_vec.

Usage:
    source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
    sudo swapon /swapfile
    python yanwen/scripts/coopmat/setup_llama31_coopmat.py --n_layers 32 --seq_len 128
    python yanwen/scripts/coopmat/bench_llama31_coopmat.py --n_layers 32 --seq_len 128
"""

import gc
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import run_llama31_pure as base  # noqa: E402

PAVAN_REPO_ROOT = Path("/home/doremy/sarc-acl/executorch/pavan-report/executorch")
base.RUNNER = PAVAN_REPO_ROOT / "cmake-out-vk" / "executor_runner"
base.DEFAULT_OUT = Path("/home/doremy/llama31_pure_run_coopmat")

DEFAULT_OUT = base.DEFAULT_OUT
RUNNER = base.RUNNER
_parent_oom_hardening = base._parent_oom_hardening
_ensure_venv_path = base._ensure_venv_path
env_check = base.env_check
load_model = base.load_model
run_etdump = base.run_etdump
bench_steady_state = base.bench_steady_state
analyze = base.analyze


def export_pte(n_layers: int, seq_len: int, out_dir: Path, want_etrecord: bool):
    import torch
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
        VkStorageType,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export

    tag = f"llama31_8b_{n_layers}L_seq{seq_len}_fp16"
    pte_path = out_dir / f"{tag}.pte"
    etrecord_path = out_dir / f"{tag}.etrecord.bin"
    input_path = out_dir / f"{tag}_input0.bin"

    have_etrecord = etrecord_path.exists()
    if (
        pte_path.exists()
        and input_path.exists()
        and (have_etrecord or not want_etrecord)
    ):
        print(
            f"[export] cached: {pte_path} ({pte_path.stat().st_size/1e9:.2f} GiB)"
            + (" + etrecord" if have_etrecord else " (no etrecord)")
        )
        return tag, pte_path, etrecord_path if have_etrecord else None, input_path

    out_dir.mkdir(parents=True, exist_ok=True)

    model, margs, checkpoint = base.load_model(n_layers, seq_len)
    example_tokens = torch.randint(0, margs.vocab_size, (1, seq_len), dtype=torch.int64)
    example_inputs = (example_tokens,)

    print(f"[export] torch.export(strict=False) tokens={tuple(example_tokens.shape)}")
    t0 = time.perf_counter()
    with torch.no_grad():
        prog = export(model, example_inputs, strict=False)
    print(f"[export] torch.export done in {time.perf_counter()-t0:.1f}s")

    del model, checkpoint
    gc.collect()

    compile_options = {"storage_type_override": VkStorageType.BUFFER}
    print(f"[export] to_edge_transform_and_lower (coopmat: {compile_options})")
    t0 = time.perf_counter()
    edge = to_edge_transform_and_lower(
        prog,
        compile_config=EdgeCompileConfig(_skip_dim_order=False),
        partitioner=[VulkanPartitioner(compile_options)],
    )
    et = edge.to_executorch()
    print(f"[export] lowered in {time.perf_counter()-t0:.1f}s")

    print(f"[export] writing .pte -> {pte_path}")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    size_gb = pte_path.stat().st_size / 1e9
    print(f"[export] .pte size: {size_gb:.2f} GiB")

    print(f"[export] writing input -> {input_path}")
    example_tokens.detach().numpy().astype("int64").tofile(input_path)

    if want_etrecord:
        print(
            f"[export] writing etrecord -> {etrecord_path} (this can OOM without swap)"
        )
        try:
            from executorch.devtools import generate_etrecord

            generate_etrecord(str(etrecord_path), edge, et, prog)
        except Exception as e:
            print(f"[export] WARNING generate_etrecord failed: {e}")

    del prog, edge, et, example_tokens, example_inputs
    gc.collect()
    return (
        tag,
        pte_path,
        etrecord_path if (want_etrecord and etrecord_path.exists()) else None,
        input_path,
    )


base.export_pte = export_pte
