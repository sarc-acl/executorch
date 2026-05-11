#!/usr/bin/env python3
"""
W8A8 (static int8 weights + int8 activations) variant of run_llama31_int8.py.

Uses XNNPACKQuantizer's static int8 per-channel config because pavan-report's
VulkanQuantizer doesn't have a static-activation W8A8 mode (its is_dynamic=True
produces dynamic quantization which neither partitioner branch matches; its
is_dynamic=False produces weight-only W8A16, the existing study path).

Pipeline (export_pte below):
  1. Load fp16 model (same as baseline).
  2. torch.export(strict=True).module()  — pre-autograd FX graph.
  3. XNNPACKQuantizer().set_global(static int8 per-channel symmetric).
  4. prepare_pt2e — inserts activation MinMaxObservers on linear sites.
  5. Calibrate with N batches of random tokens — observers latch onto fp16
     activation statistics. Real text would give better accuracy; random is
     fine for wallclock bench (model still runs, just lower output quality).
  6. convert_pt2e(fold_quantize=True) — folds activation+weight scales as
     float constants in the graph.
  7. torch.export(strict=False) the converted graph.
  8. to_edge_transform_and_lower with VulkanPartitioner({}). Pavan-report's
     partitioner pattern `make_q8ta_linear_custom_op` matches
     (input_dq_per_tensor + dq_per_channel_weight + linear + quant_output)
     and produces et_vk.q8ta_linear.default ops.

Runtime: pavan-report's cmake-out-vk/executor_runner dispatches the q8ta
ops to the linear_q8ta_q8csw_tiled (int8→fp) or q8ta_linear (int8→int8)
shaders depending on chain position.

Output dir: /home/doremy/llama31_w8a8_xnn_run/
Runner: pavan-report's cmake-out-vk/executor_runner.

Usage:
    source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
    sudo swapon /swapfile
    python yanwen/scripts/int8/setup_llama31_w8a8_xnn.py --n_layers 32 --seq_len 128
    python yanwen/scripts/int8/bench_llama31_w8a8_xnn.py  --n_layers 32 --seq_len 128
"""

import gc
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import run_llama31_pure as base  # noqa: E402

# Override paths to point at pavan-report's runner (it has q8ta_linear dispatchers).
PAVAN_ROOT = Path("/home/doremy/sarc-acl/executorch/pavan-report/executorch")
base.RUNNER = PAVAN_ROOT / "cmake-out-vk" / "executor_runner"
base.DEFAULT_OUT = Path("/home/doremy/llama31_w8a8_xnn_run")

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
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        get_symmetric_quantization_config,
        XNNPACKQuantizer,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    tag = f"llama31_8b_{n_layers}L_seq{seq_len}_w8a8_xnn"
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

    print(
        f"[export] torch.export(strict=True).module() tokens={tuple(example_tokens.shape)}"
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        pre_autograd = export(model, example_inputs, strict=True).module()
    print(f"[export] strict=True capture done in {time.perf_counter()-t0:.1f}s")

    del model, checkpoint
    gc.collect()

    # XNNPACKQuantizer static int8 per-channel.
    # NOT is_dynamic=True — that produces dynamic per-token quant. Static
    # is what pavan-report's q8ta_q8csw partitioner pattern matches.
    print("[export] XNNPACKQuantizer(static int8 per-channel)")
    quant_config = get_symmetric_quantization_config(is_per_channel=True)
    quantizer = XNNPACKQuantizer().set_global(quant_config)

    print("[export] prepare_pt2e (inserts activation observers)")
    t0 = time.perf_counter()
    prepared = prepare_pt2e(pre_autograd, quantizer)
    print(f"[export] prepare_pt2e done in {time.perf_counter()-t0:.1f}s")
    del pre_autograd
    gc.collect()

    # Calibrate. For a benchmark we don't need accurate activation ranges —
    # any consistent calibration gives a runnable model. Use a handful of
    # random-token batches so each linear's observer sees a reasonable spread.
    n_calib_batches = 4
    print(f"[export] calibration: {n_calib_batches} random batches")
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_calib_batches):
            calib_tokens = torch.randint(
                0, margs.vocab_size, (1, seq_len), dtype=torch.int64
            )
            prepared(calib_tokens)
            print(f"[export]   calib batch {i+1}/{n_calib_batches}")
    print(f"[export] calibration done in {time.perf_counter()-t0:.1f}s")

    print("[export] convert_pt2e(fold_quantize=True)")
    t0 = time.perf_counter()
    converted = convert_pt2e(prepared, fold_quantize=True)
    print(f"[export] convert_pt2e done in {time.perf_counter()-t0:.1f}s")
    del prepared
    gc.collect()

    print("[export] torch.export(strict=False) on quantized graph")
    t0 = time.perf_counter()
    with torch.no_grad():
        prog = export(converted, example_inputs, strict=False)
    print(f"[export] final export done in {time.perf_counter()-t0:.1f}s")
    del converted
    gc.collect()

    print("[export] to_edge_transform_and_lower (VulkanPartitioner)")
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
