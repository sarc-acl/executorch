"""
v2: more LLaMA-like topology — linear → silu → add (no requant) → linear.
Tests whether linear_q8ta_q8csw (int8→fp, end-of-chain) fires for the last linear.
"""

import torch
import torch.nn as nn
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.exir import to_edge_transform_and_lower
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TinyTransformerBlock(nn.Module):
    """LLaMA-FFN-like: linear → SiLU → linear, with residual add."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(64, 128, bias=False)
        self.up = nn.Linear(64, 128, bias=False)
        self.down = nn.Linear(128, 64, bias=False)

    def forward(self, x):
        return x + self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


def main():
    torch.manual_seed(0)
    model = TinyTransformerBlock().eval()
    example_inputs = (torch.randn(1, 8, 64),)

    exported = export(model, example_inputs, strict=True).module()
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(exported, quantizer)
    for _ in range(8):
        prepared(*(torch.randn_like(example_inputs[0]),))
    converted = convert_pt2e(prepared)
    quantized_exported = export(converted, example_inputs, strict=False)
    to_edge_transform_and_lower(
        quantized_exported,
        partitioner=[VulkanPartitioner({})],
    )
    print(
        "Done — search the log above for 'Operators included in this Vulkan partition'"
    )


if __name__ == "__main__":
    main()
