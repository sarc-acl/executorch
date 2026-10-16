#!/usr/bin/env python3
"""Correctness screen of one candidate's verify.sh output against the control's (owner decision 2026-10-15 23:30).

  screen.py <control verify stage> <candidate verify stage> <out json>

Production diff (rule fixed before any candidate ran): in each of the 12 pdiff logs every shape must be complete
and either `coopmat:PASSED`, or exactly the control's status for that shape with the same kernel dispatched in
linear-<scheme>.json for that (model, projection, storage). A shape that throws, fails or newly falls back with a
changed kernel rejects the candidate; it is then not timed.
Next tokens: the candidate's default-mode logs against the control's on the timed prompt (6 cells, whole text),
prompt_check.txt and r1304.txt (1B, both schemes), and the 1B decode text. Any DIFFER or INVALID item makes the
candidate CORRECTNESS_PENDING; it is still timed. The decode text is recorded, not judged. Dispatched linear kernel names of both arms are recorded."""
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nexttoken

MODELS = ('llama-3.2-1b', 'llama-3.2-3b', 'llama-3.1-8b')
CELLS = [(m, q) for m in ('1b', '3b', '8b') for q in ('4w', '8da4w')]
SHAPE = re.compile(r'^(llama-[\d.]+-\d+b)_prefill_(wq_wo|wk_wv|w1_w3|w2)_linear_\S+_(Buffer|Texture3D)$')


def pdiff(stage, model, scheme, storage):
    path = Path(stage) / 'verify' / f'pdiff-{model}-{scheme}-{storage}.log'
    lines = [l for l in path.read_text(errors='replace').splitlines() if l.startswith('[production-diff]')] if path.exists() else []
    shapes = {}
    for line in lines:
        if ' -> ' not in line and ' threw: ' not in line:
            continue
        name = line.split()[1]
        if ' threw: ' in line:
            shapes[name] = 'threw'
        else:
            correctness = re.search(r'correctness=(\S+)', line)
            shapes[name] = ('fallback' if 'NOT coopmat' in line else 'coopmat') + ':' + (correctness[1].rstrip(',') if correctness else '?')
    count = re.search(r'(\d+) shapes', lines[-1]) if lines else None
    complete = bool(lines) and ('ALL PASSED' in lines[-1] or '] FAILED (' in lines[-1]) and bool(count) and int(count[1]) == len(shapes)
    return shapes, complete


def kernels(stage):
    found = {}
    for scheme in ('4w', '8da4w'):
        path = Path(stage) / 'verify' / f'linear-{scheme}.json'
        try:
            cases = json.loads(path.read_text())['cases']
        except (OSError, ValueError, KeyError):
            continue
        for case in cases:
            name = re.search(r'"kernel_name": "([^"]+)"', str(case.get('kernel', '')))
            found[f"{scheme} {case.get('model')} {case.get('op')} {str(case.get('storage', '')).lower()}"] = {
                'kernel': name[1] if name else str(case.get('kernel')), 'dispatch': case.get('dispatch'),
                'kernel_median_us': case.get('kernel_median_us')}
    return found


def rc_of(stage, log):
    records = [json.loads(l) for l in (Path(stage) / 'invocations.jsonl').read_text().splitlines()]
    hits = [r['rc'] for r in records if r.get('stdout_log', '').endswith('/verify/' + log)]
    return hits[0] if len(hits) == 1 else '?'


def main():
    control, candidate, out = sys.argv[1:4]
    result = {'control': control, 'candidate': candidate, 'pdiff': {}, 'pdiff_reasons': [], 'nexttoken': {}}
    control_kernels, candidate_kernels = kernels(control), kernels(candidate)
    result['control_kernels'] = control_kernels
    result['candidate_kernels'] = candidate_kernels
    result['changed_kernels'] = sorted(k for k in set(control_kernels) | set(candidate_kernels)
                                       if control_kernels.get(k, {}).get('kernel') != candidate_kernels.get(k, {}).get('kernel'))
    if len(control_kernels) != 48 or len(candidate_kernels) != 48:
        result['pdiff_reasons'].append(f'linear json cases control={len(control_kernels)} candidate={len(candidate_kernels)}, required 48')
    for model in MODELS:
        for scheme in ('4w', '8da4w'):
            for storage in ('buffer', 'texture3d'):
                key = f'{model} {scheme} {storage}'
                theirs, control_complete = pdiff(control, model, scheme, storage)
                ours, complete = pdiff(candidate, model, scheme, storage)
                result['pdiff'][key] = ours
                if not complete or not control_complete:
                    result['pdiff_reasons'].append(f'{key}: incomplete log (candidate {complete}, control {control_complete})')
                for name, status in ours.items():
                    if status == 'coopmat:PASSED':
                        continue
                    shape = SHAPE.match(name)
                    case = f'{scheme} {shape[1]} {shape[2]} {shape[3].lower()}' if shape else None
                    same_kernel = case is not None and case in control_kernels and \
                        control_kernels[case].get('kernel') == candidate_kernels.get(case, {}).get('kernel')
                    if status != theirs.get(name) or not same_kernel:
                        result['pdiff_reasons'].append(f'{key} {name}: {status} (control {theirs.get(name)}, same kernel {same_kernel})')
    result['pdiff_pass'] = not result['pdiff_reasons']
    prompts = Path(candidate)
    items = [(f'prefill {m} {q}', 'prompt_2048.txt', '2048', f'prefill-{m}-{q}-default.log', True) for m, q in CELLS]
    for q in ('4w', '8da4w'):
        items += [(f'check 1b {q}', 'prompt_check.txt', '1972', f'check-1b-{q}-default.log', False),
                  (f'unaligned 1b {q}', 'r1304.txt', '1792', f'unaligned-1b-{q}-default.log', False),
                  (f'decode 1b {q}', 'prompt_2048.txt', '2048', f'decode-1b-{q}.log', True)]
    for item, prompt, want, log, timed in items:
        evaluation = nexttoken.evaluate(str(prompts / prompt), want, str(Path(control) / 'verify' / log),
                                        str(Path(candidate) / 'verify' / log), rc_of(control, log), rc_of(candidate, log), timed)
        result['nexttoken'][item] = evaluation
    differing = sorted(k for k, v in result['nexttoken'].items() if v['verdict'] != 'SAME' and not k.startswith('decode'))
    result['nexttoken_not_same'] = differing
    result['verdict'] = 'REJECTED_PDIFF' if not result['pdiff_pass'] else ('CORRECTNESS_PENDING' if differing else 'PASS')
    Path(out).write_text(json.dumps(result, indent=1) + '\n')
    print(result['verdict'], '; '.join(result['pdiff_reasons'][:5]), 'nexttoken not SAME:', ','.join(differing) or '-')


if __name__ == '__main__':
    main()
