#!/usr/bin/env python3
"""
Int8 (W8A16, vulkan_8w) variant of run_llama31_pure.py — same methodology as
the fp16 baseline, but the exported .pte contains per-channel-symmetric int8
weights folded by PT2E. Runtime dispatches `linear_q8csw_tiled_*` for delegated
linears.

Pipeline (in export_pte below):
  1. Load fp16 model (same as baseline).
  2. torch.export(strict=True).module()  — pre-autograd FX graph.
  3. prepare_pt2e(graph, VulkanQuantizer.set_global(w8a16))  — inserts weight observers.
  4. One forward pass (dummy calibration) so observers see weights.
  5. convert_pt2e(prepared, fold_quantize=True)  — folds int8 scales/zeros.
  6. torch.export(strict=False) the converted graph.
  7. to_edge_transform_and_lower with stock VulkanPartitioner({}) — its pattern
     matcher recognizes (dequant_per_channel + linear) and fuses to
     et_vk.linear_q8csw.default.

Output dir: /home/doremy/llama31_pure_run_int8/
Runner: main tree's cmake-out-vk/executor_runner (no coopmat needed; int8
        shaders are in main).

Usage:
    source /home/doremy/sarc-acl/executorch/main/executorch/.venv/bin/activate
    sudo swapon /swapfile
    python yanwen/scripts/int8/setup_llama31_int8.py --n_layers 32 --seq_len 128
    python yanwen/scripts/int8/bench_llama31_int8.py  --n_layers 32 --seq_len 128
"""

import gc
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import run_llama31_pure as base  # noqa: E402

base.DEFAULT_OUT = Path("/home/doremy/llama31_pure_run_int8")

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
    from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
        get_symmetric_quantization_config,
        VulkanQuantizer,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    tag = f"llama31_8b_{n_layers}L_seq{seq_len}_int8"
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

    # Step 1: pre-autograd FX graph (strict=True so PT2E sees a torch.fx.GraphModule)
    print(
        f"[export] torch.export(strict=True).module() tokens={tuple(example_tokens.shape)}"
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        pre_autograd = export(model, example_inputs, strict=True).module()
    print(f"[export] strict=True capture done in {time.perf_counter()-t0:.1f}s")

    del model, checkpoint
    gc.collect()

    # Step 2: PT2E w8a16 quantizer
    print("[export] VulkanQuantizer(w8a16, per-channel symmetric)")
    quant_config = get_symmetric_quantization_config(is_dynamic=False, weight_bits=8)
    quantizer = VulkanQuantizer().set_global(quant_config)

    # Step 3: prepare (inserts PerChannelMinMaxObserver on weights)
    print("[export] prepare_pt2e")
    t0 = time.perf_counter()
    prepared = prepare_pt2e(pre_autograd, quantizer)
    print(f"[export] prepare_pt2e done in {time.perf_counter()-t0:.1f}s")
    del pre_autograd
    gc.collect()

    # Step 4: calibrate (one forward; observers latch onto fp16 weight statistics)
    # For weight-only quant the activation observers are no-ops (act_spec=None),
    # so the only useful thing this pass does is trigger per-channel weight stats.
    # Slow on CPU (~30–120 s for L=32 fp16), but only run once.
    print("[export] calibration forward (one pass; slow on CPU)")
    t0 = time.perf_counter()
    with torch.no_grad():
        prepared(*example_inputs)
    print(f"[export] calibration done in {time.perf_counter()-t0:.1f}s")

    # Step 5: convert (fold scales/zeros, materialize int8 weights as constants)
    print("[export] convert_pt2e(fold_quantize=True)")
    t0 = time.perf_counter()
    converted = convert_pt2e(prepared, fold_quantize=True)
    print(f"[export] convert_pt2e done in {time.perf_counter()-t0:.1f}s")
    del prepared
    gc.collect()

    # Step 6: final export of the quantized graph
    print("[export] torch.export(strict=False) on quantized graph")
    t0 = time.perf_counter()
    with torch.no_grad():
        prog = export(converted, example_inputs, strict=False)
    print(f"[export] final export done in {time.perf_counter()-t0:.1f}s")
    del converted
    gc.collect()

    # Step 7: lower with stock VulkanPartitioner (auto-detects quantized linear pattern)
    print("[export] to_edge_transform_and_lower (stock VulkanPartitioner)")
    t0 = time.perf_counter()
    edge = to_edge_transform_and_lower(
        prog,
        compile_config=EdgeCompileConfig(_skip_dim_order=False),
        partitioner=[VulkanPartitioner({})],
    )
    et = edge.to_executorch()
    print(f"[export] lowered in {time.perf_counter()-t0:.1f}s")

    if not want_etrecord:
        del prog, edge
        gc.collect()

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

    if "prog" in locals():
        del prog
    if "edge" in locals():
        del edge
    del et, example_tokens, example_inputs
    gc.collect()
    return (
        tag,
        pte_path,
        etrecord_path if (want_etrecord and etrecord_path.exists()) else None,
        input_path,
    )


base.export_pte = export_pte
