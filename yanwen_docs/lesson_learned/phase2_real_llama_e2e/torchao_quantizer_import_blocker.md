# `executorch.examples.models.llama` import fails on locally installed torchao

## What was attempted

For the Phase 2 real-LLaMA E2E study I ran the sibling helper
`/home/doremy/Desktop/samsung/executorch/yanwen_plan/run_real_llama_e2e.py`
against this branch's runtime build:

```bash
python3 yanwen_docs/agent_results/real_llama_e2e_storage_study/scripts/run_real_llama_e2e.py \
  --local --executor_runner cmake-out-vk-etdump/executor_runner --fp16 \
  --n_layers 4 --seq_len 256 --runs 6 --only tex \
  --cache_dir ~/llama3_1_8b ...
```

## What happened

Module import fails before any user code runs:

```text
File ".../torchao/quantization/pt2e/quantize_pt2e.py", line 44, in <module>
    quantizer: Union[Quantizer, torch.ao.quantization.quantizer.quantizer.Quantizer],
AttributeError: module 'torch.ao.quantization' has no attribute 'quantizer'.
Did you mean: 'quantize'?
```

Trace: `run_real_llama_e2e.py` does `from
executorch.examples.models.llama.llama_transformer import construct_transformer`,
which triggers `executorch/examples/models/llama/__init__.py` →
`from .model import Llama2Model` →
`from executorch.extension.llm.export.builder import LLMEdgeManager` →
`from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e`.
That last module evaluates the type annotation
`torch.ao.quantization.quantizer.quantizer.Quantizer` eagerly at import time,
and the locally installed `torch 2.11.0+cpu` does not expose the `quantizer`
sub-module.

## Why it matters

This blocks every Phase 2 / Phase 3 run that imports the LLaMA transformer or
anything else that pulls `executorch.extension.llm.export.builder`, even when
the user has no intent of doing PT2E quantization. fp16 prefill measurement
should not require torchao at all.

## Workaround used

`scripts/run_real_llama_e2e_patched.py` stubs the missing module before the
real script imports executorch, then `runpy.run_path`s the original:

```python
import sys, types, torch
fake_qq = types.ModuleType("torch.ao.quantization.quantizer.quantizer")
class _StubQuantizer: pass
fake_qq.Quantizer = _StubQuantizer
fake_q = types.ModuleType("torch.ao.quantization.quantizer")
fake_q.quantizer = fake_qq
fake_q.Quantizer = _StubQuantizer
sys.modules["torch.ao.quantization.quantizer"] = fake_q
sys.modules["torch.ao.quantization.quantizer.quantizer"] = fake_qq
torch.ao.quantization.quantizer = fake_q
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), "run_real_llama_e2e.py"),
               run_name="__main__")
```

This evaluates the bad annotation against a placeholder class; nothing in our
fp16 prefill path actually calls `prepare_pt2e` or `convert_pt2e`, so the stub
is never exercised at runtime.

## Recommended next action

Pin torchao to a version compatible with the installed torch
(`torch 2.11.0+cpu`), or push for `from __future__ import annotations` on
`torchao/quantization/pt2e/quantize_pt2e.py` so the annotation does not
evaluate at import time. Yanwen's previous-round checkout at
`/home/doremy/Desktop/samsung/executorch/` does not hit this because torchao
is either absent there or pinned differently. Confirm before a longer Phase 3
session.
