#!/usr/bin/env python3
"""Check the B70 admission record after the common revised-input bundle arrives."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(receipt, model_root):
    record = json.loads(receipt.read_text())
    model_root = model_root.resolve(strict=True)
    if record['status'] != 'READY_FOR_CORRECTNESS' or not record['dataset_id'].startswith('revised-'):
        raise ValueError('Revised inputs have not been admitted for correctness work')
    if Path(record['model_root']).resolve(strict=True) != model_root:
        raise ValueError('Configured model root differs from admitted inputs')
    original_root = Path(record['original_model_root']).resolve(strict=True)
    if model_root == original_root:
        raise ValueError('Original inputs cannot be admitted as revised inputs')
    expected = set()
    for model, stem in [('llama-3.2-1b', 'llama3_2-1b'), ('llama-3.2-3b', 'llama3_2-3b'), ('llama-3.1-8b', 'llama3_1-8b')]:
        expected.add(f'{model}/original/tokenizer.model')
        for scheme in ('4w', '8da4w'):
            expected.add(f'{model}/exported/{stem}_vulkan_{scheme}.pte')
    if set(record['files']) != expected:
        raise ValueError('Admission must identify all six models and three tokenizers')
    for name, entry in record['files'].items():
        path = model_root / name
        if path.stat().st_size != entry['size_bytes']:
            raise ValueError(f'Input size changed: {name}')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != entry['sha256']:
            raise ValueError(f'Input hash changed: {name}')
        if name.endswith('.pte') and path.resolve() == (original_root / name).resolve():
            raise ValueError(f'Revised model aliases an original model: {name}')
    for name in ('recipe', 'logical_constant_equivalence', 'runtime_metadata', 'resource_legality', 'selector_coverage', 'gate_resource_legality'):
        entry = record['evidence'][name]
        path = Path(entry['path'])
        if not path.is_absolute():
            raise ValueError(f'Evidence path must be absolute: {name}')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if path.stat().st_size == 0 or digest != entry['sha256']:
            raise ValueError(f'Missing or changed admission evidence: {name}')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('model_root', type=Path)
    args = parser.parse_args()
    result = validate(args.receipt, args.model_root)
    print(f"INPUT_PROVENANCE_OK dataset={result['dataset_id']}; correctness and timing remain separate requirements")
