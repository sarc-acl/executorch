"""Export the vision models of ExecuTorch's Vulkan CI with the Vulkan partitioner.

Run from the workspace's `main/` directory (so `import executorch` resolves to the read-only source tree
`main/executorch` at 2efa9d0316, as in the earlier MobileNet survey) with
`PYTHONPATH=OUT/pydeps` for torchvision/torchaudio/timm/torchsr:

    cd main && PYTHONPATH=$OUT/pydeps executorch/.venv/bin/python $OUT/export_models.py MODEL PRECISION

Writes into OUT/models/: MODEL-input.bin (float32), MODEL-fp32-reference-<i>.bin (CPU eager FP32 outputs),
MODEL-int8-q8ta-reference-<i>.bin (CPU fake-quantized outputs), MODEL-PRECISION.pte and MODEL-PRECISION.json
(parameter count, PTE size, delegation counts, operators left on CPU).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils._pytree import tree_flatten

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.devtools.backend_debug import get_delegation_info

MODELS = ["mv2", "mv3", "resnet18", "resnet50", "ic3", "ic4", "edsr", "dl3", "w2l", "densenet161"]
PRECISIONS = ["fp32", "fp16", "int8-q8ta"]


def load(model_name):
    """Model and example inputs exactly as examples/vulkan/export.py builds them."""
    from executorch.examples.models import MODEL_NAME_TO_MODEL
    from executorch.examples.models.model_factory import EagerModelFactory

    if model_name in MODEL_NAME_TO_MODEL:
        torch.manual_seed(0)
        model, example_inputs, _, _ = EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[model_name])
        source = f"examples/models {MODEL_NAME_TO_MODEL[model_name]}"
    else:
        # Same as examples/vulkan/export.py init_model() and get_sample_inputs(): torchvision model with
        # random (untrained) weights, input randn(1, 3, 224, 224) after torch.manual_seed(42).
        import torchvision

        assert model_name == "densenet161"
        torch.manual_seed(0)
        model = torchvision.models.densenet161()
        torch.manual_seed(42)
        example_inputs = (torch.randn(1, 3, 224, 224),)
        source = "examples/vulkan/export.py init_model(): torchvision.models.densenet161() (untrained)"
    return model.eval(), example_inputs, source


def save_outputs(outputs, prefix):
    flat, _ = tree_flatten(outputs)
    for i, tensor in enumerate(flat):
        tensor.detach().to(torch.float32).contiguous().numpy().tofile(f"{prefix}-{i}.bin")
    return [list(t.shape) for t in flat]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("precision", choices=PRECISIONS)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / "models"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)

    model, inputs, source = load(args.model)
    info = {
        "model": args.model, "precision": args.precision, "source": source,
        "input_shapes": [list(t.shape) for t in inputs],
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
    }
    with torch.no_grad():
        input_path = out / f"{args.model}-input.bin"
        data = b"".join(t.to(torch.float32).contiguous().numpy().tobytes() for t in inputs)
        if input_path.exists():
            assert input_path.read_bytes() == data, "example input differs from the saved one"
        else:
            input_path.write_bytes(data)
        info["output_shapes"] = save_outputs(model(*inputs), out / f"{args.model}-fp32-reference")

        if args.precision == "int8-q8ta":
            # The int8-q8ta recipe of the earlier MobileNet survey: XNNPACK quantizer, static symmetric
            # per-channel INT8, calibrated here on the example input and five more seeded randn inputs.
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                get_symmetric_quantization_config,
                XNNPACKQuantizer,
            )
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

            quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
            prepared = prepare_pt2e(torch.export.export(model, inputs, strict=True).module(), quantizer)
            prepared(*inputs)
            generator = torch.Generator().manual_seed(1234)
            for _ in range(5):
                prepared(*(torch.randn(t.shape, generator=generator) for t in inputs))
            model = convert_pt2e(prepared)
            save_outputs(model(*inputs), out / f"{args.model}-int8-q8ta-reference")

        exported = torch.export.export(model, inputs, strict=True)
        options = {"force_fp16": True} if args.precision == "fp16" else {}
        lowered = to_edge_transform_and_lower(exported, partitioner=[VulkanPartitioner(options)])
        delegation = get_delegation_info(lowered.exported_program().graph_module)
        info["delegated_nodes"] = delegation.num_delegated_nodes
        info["non_delegated_nodes"] = delegation.num_non_delegated_nodes
        info["delegated_subgraphs"] = delegation.num_delegated_subgraphs
        info["non_delegated_ops"] = {
            op: c.non_delegated for op, c in sorted(delegation.delegation_by_operator.items()) if c.non_delegated
        }
        path = out / f"{args.model}-{args.precision}.pte"
        with path.open("wb") as stream:
            lowered.to_executorch().write_to_file(stream)
        info["pte_bytes"] = path.stat().st_size
    (out / f"{args.model}-{args.precision}.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    sys.exit(main())
