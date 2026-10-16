#!/usr/bin/env python3
"""Validate saved finite-session evidence and preserve every original run row."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('artifact_root', type=pathlib.Path)
parser.add_argument('session')
parser.add_argument('--candidate', default='control')
parser.add_argument('--parent-verify', type=pathlib.Path)
parser.add_argument('--inventory', type=pathlib.Path)
args = parser.parse_args()
root = args.artifact_root
stage = root / 'stage' / args.session
cells = [(m, q) for m in ('1b', '3b', '8b') for q in ('4w', '8da4w')]
issues = []
result = {'session': args.session, 'candidate': args.candidate, 'issues': issues}
input_record = json.loads((stage / 'INPUTS.json').read_text())
input_hash = hashlib.sha256((stage / 'INPUTS.json').read_bytes()).hexdigest()
result['input_dataset'] = input_record['dataset_id']
result['input_receipt_sha256'] = input_hash

if (stage / 'verify.out').exists():
    verify = (stage / 'verify.out').read_text()
    if not verify.rstrip().endswith('VERIFY_DONE rc=0'):
        issues.append('verify did not complete with rc0')
    pdiff = re.findall(r'^pdiff .* rc=0 .*ALL PASSED', verify, re.M)
    if len(pdiff) != 12:
        issues.append('verify requires twelve passed production-diff cases')
    for model, scheme in cells:
        for arm in ('default', 'tiled'):
            log = (stage / f'verify/prefill-{model}-{scheme}-{arm}.log').read_text()
            if '"prompt_tokens":2048' not in log or re.search(r'Aborted|corrupted|VK_ERROR_DEVICE_LOST', log):
                issues.append(f'verify prefill {model} {scheme} {arm} invalid')
else:
    issues.append('verify output missing')

if args.parent_verify:
    parent = args.parent_verify.read_text().splitlines()
    candidate = (stage / 'verify.out').read_text().splitlines()
    normalize = lambda line: re.sub(r'(tok_s=)[0-9.eE+-]+', r'\1#', line.rstrip())
    expected_names = []
    for i in range(max(len(parent), len(candidate))):
        a = normalize(parent[i]) if i < len(parent) else '<missing>'
        b = normalize(candidate[i]) if i < len(candidate) else '<missing>'
        if a == b:
            continue
        match_a = re.fullmatch(r'linear (4w|8da4w) rc=(\d+) kernels: (.*)', a)
        match_b = re.fullmatch(r'linear (4w|8da4w) rc=(\d+) kernels: (.*)', b)
        if match_a and match_b and match_a.group(1, 2) == match_b.group(1, 2):
            if not args.inventory:
                issues.append('inventory is required to audit changed dispatch names')
            else:
                allowed = set(re.findall(r'"([^"]+)"', match_a[3]))
                inventory = json.loads(args.inventory.read_text())
                allowed.update(entry['shader'] for entry in inventory['candidates'])
                actual = set(re.findall(r'"([^"]+)"', match_b[3]))
                if not actual <= allowed:
                    issues.append(f'unreviewed kernels in verify: {actual - allowed}')
            expected_names.append({'line': i + 1, 'parent': a, 'candidate': b})
        else:
            issues.append(f'verify line {i + 1} differs: {a} / {b}')
    result['linear_name_differences_to_audit'] = expected_names

raw = stage / 'raw'
if (raw / 'runs.csv').exists():
    text = (raw / 'runs.csv').read_text()
    if not text.endswith('\n'):
        raise ValueError('Unterminated raw CSV record; preserve and diagnose before resuming')
    rows = list(csv.DictReader(text.splitlines()))
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Malformed raw CSV; preserve and diagnose before resuming')
    env_text = (raw / 'env.txt').read_text()
    repeats = int(re.search(r'reps=(\d+)', env_text)[1])
    result['repeats'] = repeats
    properties = json.loads((root / 'preflight/vulkan-device0.json').read_text())['capabilities']['device']['properties']
    identity = properties['VkPhysicalDeviceVulkan11Properties']
    driver = properties['VkPhysicalDeviceVulkan12Properties']
    manifests = {}
    for arm in ('parent', 'cand'):
        arm_dir = stage / arm
        manifests[arm] = {
            'commit': (arm_dir / 'COMMIT').read_text().strip(),
            'binary': str(arm_dir / 'llama_main'),
            'binary_sha256': hashlib.sha256((arm_dir / 'llama_main').read_bytes()).hexdigest(),
            'profile': ';'.join((arm_dir / 'env').read_text().splitlines()),
        }
    target = root / 'runs.csv'
    old_keys = set()
    if target.exists():
        if not target.read_bytes().endswith(b'\n'):
            raise ValueError('Unterminated central CSV record; preserve and diagnose')
        with target.open() as handle:
            old_keys = {row['run_id'] for row in csv.DictReader(handle)}
    extra = ['run_id', 'session', 'candidate', 'device_unit', 'device_uuid', 'driver', 'commit', 'binary', 'binary_sha256', 'profile', 'sensor_evidence', 'input_dataset', 'input_receipt_sha256', 'model_sha256']
    new_file = not target.exists() or target.stat().st_size == 0
    with target.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=extra + list(rows[0]))
        if new_file:
            writer.writeheader()
        for index, row in enumerate(rows):
            key = f'{args.session}:{index}'
            if key in old_keys:
                continue
            arm = manifests[row['build']]
            model, stem = {'1b': ('llama-3.2-1b', 'llama3_2-1b'), '3b': ('llama-3.2-3b', 'llama3_2-3b'), '8b': ('llama-3.1-8b', 'llama3_1-8b')}[row['model']]
            model_entry = input_record['files'][f"{model}/exported/{stem}_vulkan_{row['scheme']}.pte"]
            writer.writerow({**row, **arm, 'run_id': key, 'session': args.session,
                             'candidate': args.candidate, 'device_unit': 'primary:0',
                             'device_uuid': bytes(identity['deviceUUID']).hex(),
                             'driver': driver['driverInfo'],
                             'input_dataset': input_record['dataset_id'],
                             'input_receipt_sha256': input_hash,
                             'model_sha256': model_entry['sha256'],
                             'sensor_evidence': str(raw / pathlib.Path(row['log']).with_suffix('.clk'))})
    timed = [row for row in rows if row['log'].startswith('logs/prefill')]
    result['invalid_runs'] = [row for row in timed if row['valid'] != '1']
    if 'E2E5_OK' not in (raw / 'done.txt').read_text():
        issues.append('timing/next-token session incomplete')
    samples = []
    rng = random.Random(7321)
    summary = []
    for model, scheme in cells:
        values = {}
        for arm in ('parent', 'cand'):
            values[arm] = [float(row['tok_s']) for row in timed if (row['model'], row['scheme'], row['build'], row['valid']) == (model, scheme, arm, '1')][:repeats]
        if min(map(len, values.values())) < repeats:
            issues.append(f'{model} {scheme}: fewer than {repeats} valid runs')
            continue
        control = statistics.median(values['parent'])
        candidate = statistics.median(values['cand'])
        boot = [statistics.median(rng.choices(values['parent'], k=repeats)) / statistics.median(rng.choices(values['cand'], k=repeats)) for _ in range(20000)]
        samples.append(boot)
        ordered = sorted(boot)
        summary.append({'model': model, 'scheme': scheme, 'control_tok_s': control,
                        'candidate_tok_s': candidate, 'slowdown': control / candidate - 1,
                        'slowdown_bootstrap_95': [ordered[499] - 1, ordered[19499] - 1],
                        'valid_control': len(values['parent']), 'valid_candidate': len(values['cand']),
                        'control_spread': (max(values['parent']) - min(values['parent'])) / control,
                        'candidate_spread': (max(values['cand']) - min(values['cand'])) / candidate})
    result['cells'] = summary
    if len(summary) == 6:
        geomean = math.exp(sum(math.log(row['control_tok_s'] / row['candidate_tok_s']) for row in summary) / 6)
        boot = sorted(math.exp(sum(math.log(sample[i]) for sample in samples) / 6) - 1 for i in range(20000))
        result['geomean_slowdown'] = geomean - 1
        result['geomean_slowdown_bootstrap_95'] = [boot[499], boot[19499]]
        result['point_estimates_within_member_margin'] = geomean <= 1.03 and all(row['slowdown'] <= .08 for row in summary)
        result['a_a_requires_seven_repeats'] = any(abs(row['candidate_tok_s'] / row['control_tok_s'] - 1) > .01 or max(row['control_spread'], row['candidate_spread']) > .03 for row in summary)
        result['a_a_consistent'] = abs(1 / geomean - 1) <= .01 and all(abs(row['candidate_tok_s'] / row['control_tok_s'] - 1) <= .02 for row in summary)
    result['uncertainty'] = '20,000 independent-arm bootstrap resamples, seed7321; intervals describe this session only'

(stage / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('invalid_runs', 'cells')}, indent=2))
raise SystemExit(bool(issues))
