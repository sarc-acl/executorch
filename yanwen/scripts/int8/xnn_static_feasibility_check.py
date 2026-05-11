"""
Tiny feasibility test: does XNNPACKQuantizer's static int8 config produce a
graph that pavan-report's VulkanPartitioner matches at the linear sites?
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


class TinyLLaMALinears(nn.Module):
    """Two linears matching LLaMA FFN gate/up + down at miniature dims (so
    the calibration is fast). Same arithmetic structure (input → linear → linear)."""

    def __init__(self):
        super().__init__()
        # Pretend dim=64, ffn=128 (tiny). Matches LLaMA's linear shapes
        # structurally.
        self.gate = nn.Linear(64, 128, bias=False)
        self.down = nn.Linear(128, 64, bias=False)

    def forward(self, x):
        return self.down(self.gate(x))


def main():
    torch.manual_seed(0)
    model = TinyLLaMALinears().eval()

    # Dummy input shaped like prefill: [batch=1, seq=8, dim=64]
    example_inputs = (torch.randn(1, 8, 64),)

    # 1. Export to ATen IR
    exported = export(model, example_inputs, strict=True).module()

    # 2. Apply XNNPACK static int8 quantizer
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

    prepared = prepare_pt2e(exported, quantizer)

    # 3. Calibrate with a few random batches
    for _ in range(8):
        prepared(*(torch.randn_like(example_inputs[0]),))

    # 4. Convert to quantized graph
    converted = convert_pt2e(prepared)

    print("=== Quantized graph nodes ===")
    for node in converted.graph.nodes:
        if node.op == "call_function":
            print(
                f"  {node.op}: {node.target} args={[a if not hasattr(a,'op') else f'<{a.op}:{a.name}>' for a in node.args[:3]]}"
            )

    # 5. Re-export the quantized graph
    quantized_exported = export(converted, example_inputs, strict=False)

    # 6. Lower with VulkanPartitioner — does the q8ta_q8csw partitioner match?
    lowered = to_edge_transform_and_lower(
        quantized_exported,
        partitioner=[VulkanPartitioner({})],
    )

    print("\n=== Lowered graph: searching for linear sites ===")
    em = lowered.exported_program()
    delegated_count = 0
    qcsnw_count = 0
    q8ta_q8csw_count = 0
    linear_count = 0
    for node in em.graph.nodes:
        if node.op == "call_function":
            t = str(node.target)
            if (
                "linear" in t.lower()
                or "q8ta" in t.lower()
                or "q8csw" in t.lower()
                or "delegate" in t.lower()
            ):
                print(f"  {t}  (op={node.op})")
            if "linear" in t.lower():
                linear_count += 1
            if "delegate" in t.lower():
                delegated_count += 1
            if "linear_q8ta_q8csw" in t:
                q8ta_q8csw_count += 1
            if "qcsnw" in t.lower():
                qcsnw_count += 1

    print("\n=== Summary ===")
    print(f"  total linear-like nodes: {linear_count}")
    print(f"  delegate_call nodes: {delegated_count}")
    print(f"  linear_q8ta_q8csw matches: {q8ta_q8csw_count}")
    print(f"  linear_qcsnw matches: {qcsnw_count}")


if __name__ == "__main__":
    main()
