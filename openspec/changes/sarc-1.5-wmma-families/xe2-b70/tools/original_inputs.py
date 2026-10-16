#!/usr/bin/env python3
"""original_inputs.py <model root> <receipt>: record the original exported models and tokenizers that the
2026-10-15 23:30 UTC owner decision selects for timing (the revised RoPE inputs are kept but not used)."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
files = {}
for model, stem in [('llama-3.2-1b', 'llama3_2-1b'), ('llama-3.2-3b', 'llama3_2-3b'), ('llama-3.1-8b', 'llama3_1-8b')]:
    names = [f'{model}/original/tokenizer.model'] + [f'{model}/exported/{stem}_vulkan_{q}.pte' for q in ('4w', '8da4w')]
    for name in names:
        with (root / name).open('rb') as handle:
            files[name] = {'size_bytes': (root / name).stat().st_size,
                           'sha256': hashlib.file_digest(handle, 'sha256').hexdigest()}
record = {
    'status': 'ORIGINAL_INPUTS_OWNER_DECISION_2026-10-15T23:30Z',
    'dataset_id': 'original-exported',
    'model_root': str(root),
    'original_model_root': str(root),
    'caveat': 'texture3d images over the 2048 extent reported by ANV (RoPE tables in all six models, 8B '
              'activations, verify.sh 8B production-diff fixtures): pre-existing, identical in both arms, '
              'not a blocker by owner decision',
    'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files': files,
}
Path(sys.argv[2]).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
