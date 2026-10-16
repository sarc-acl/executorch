#!/usr/bin/env python3
"""Recompute the Orin WMMA matrix from the mirrored raw evidence of a matrix queue.

  analyze_matrix.py <mirrored evidence dir> <historical s5-c1 session dir> <results dir>

Writes runs.csv (every row of every session, with provenance), matrix.csv (per candidate and cell: medians of the
first 7 valid runs per arm, ratio = control / candidate tok/s), dispatch.csv (dispatched linear kernel per
candidate and case), nexttoken.csv, control-verify-vs-s5-c1.txt and summary.json. Validity is recomputed from
each run's log and clock samples with runrow.py against the A/A floor; a row whose recomputation disagrees with
its recorded verdict is listed."""
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runrow

CELLS = [(m, q) for m in ('1b', '3b', '8b') for q in ('4w', '8da4w')]
REPS = 7
LIMIT_GEOMEAN, LIMIT_CELL = 1.03, 1.08


def rates_removed(path):
    return [re.sub(r'(tok_s=)\S+', r'\1*', l.rstrip('\n')) for l in open(path)]


def main():
    evidence, historical, results = (Path(x) for x in sys.argv[1:4])
    results.mkdir(parents=True, exist_ok=True)
    plan = json.loads((evidence / 'config.json').read_text())
    device = plan['device']
    state = json.loads((evidence / 'state.json').read_text())
    floor = json.loads((evidence / 'floor.json').read_text())
    clkmin = {cell: v['clkmin_mhz'] for cell, v in floor['cells'].items()}
    summary = {'tag': plan['tag'], 'queue_status': state['status'], 'floor': floor, 'sessions': {}, 'candidates': {},
               'recompute_mismatches': []}
    rows_out = []
    medians = {}
    for step in state['steps']:
        if step['kind'] != 'session':
            continue
        session = Path(step['dir']).name
        raw = evidence / 'steps' / session / 'raw'
        arms = {arm: dict(l.strip().split('=', 1) for l in (evidence / 'steps' / session / arm / 'env').read_text().splitlines() if l.strip())
                for arm in ('parent', 'cand')}
        calibrate = session == 'session-aa'
        valid = {}
        for row in csv.DictReader(open(raw / 'runs.csv')):
            tag = row['log'].split('/')[1].split('-')[0]
            cell = f"{row['model']}-{row['scheme']}"
            check = runrow.evaluate(str(raw / row['log']), str(raw / row['log'].replace('.log', '.clk')),
                                    row['prompt_tokens'] if tag != 'prefill' else '2048',
                                    clkmin[cell] if tag == 'prefill' else 0, row['rc'], row['others'], tag)
            if tag == 'prefill' and (check['valid'] != row['valid'] or check['tok_s'] != row['tok_s']):
                summary['recompute_mismatches'].append(f"{session} {row['log']}: recorded {row['valid']}/{row['tok_s']} "
                                                       f"recomputed {check['valid']}/{check['tok_s']} {check['reason']}")
            env = arms[row['build']]
            rows_out.append({
                'device': device['device'], 'unit': device['unit'], 'driver': device['driver'],
                'commit': device['source_commit'], 'build': device['build'], 'build_manifest_sha256': device['build_manifest_sha256'],
                'binary_llama_main_sha256': device['binary_sha256']['llama_main'],
                'binary_libllama_runner_sha256': device['binary_sha256']['libllama_runner.so'],
                'session': f"{plan['tag']}/{session}", 'candidate': step['candidate'], 'arm': row['build'],
                'profile': env['ET_VK_SARC_DEV_PROFILE'], 'sdpa_fused': env.get('ET_VK_SARC_ORIN_SDPA_FUSED', ''),
                'cell': cell, 'tag': tag, 'rep': row['rep'], 'slot': row['slot'], 'tok_s': row['tok_s'], 'rc': row['rc'],
                'prompt_tokens': row['prompt_tokens'], 'generated_tokens': row['generated_tokens'], 'prefill_ms': row['prefill_ms'],
                'clk_n': row['clk_n'], 'clk_med_mhz': row['clk_med_mhz'], 'clk_min_mhz': row['clk_min_mhz'],
                'power_med_w': row['power_med_w'], 'temp_pre': row['temp_pre'], 'temp_post': row['temp_post'], 'temp_max': row['temp_max'],
                'others': row['others'], 'utc': row['utc'], 'clkmin_in_session': row['clkmin'],
                'valid_recorded': row['valid'], 'reason_recorded': row['reason'],
                'valid_floor_applied': check['valid'] if tag == 'prefill' else '', 'reason_floor_applied': check['reason'] if tag == 'prefill' else '',
                'log': f"steps/{session}/raw/{row['log']}", 'sensor_log': f"steps/{session}/raw/{row['log'].replace('.log', '.clk')}",
                'memory_log': f"steps/{session}/raw/{row['log'].replace('.log', '.mem')}"})
            if tag == 'prefill' and check['valid'] == '1':
                valid.setdefault((cell, row['build']), []).append((int(row['rep']), float(row['tok_s']), float(row['clk_med_mhz'])))
        cells = {}
        for m, q in CELLS:
            cell = f'{m}-{q}'
            arms_runs = {arm: sorted(valid.get((cell, arm), []))[:REPS] for arm in ('parent', 'cand')}
            entry = {f'{arm}_valid': len(valid.get((cell, arm), [])) for arm in ('parent', 'cand')}
            if all(len(v) == REPS for v in arms_runs.values()):
                pm = statistics.median(x[1] for x in arms_runs['parent'])
                cm = statistics.median(x[1] for x in arms_runs['cand'])
                spread = {arm: (max(x[1] for x in v) - min(x[1] for x in v)) / statistics.median(x[1] for x in v)
                          for arm, v in arms_runs.items()}
                entry.update(control_median=pm, candidate_median=cm, ratio=pm / cm, throughput_ratio=cm / pm,
                             control_spread=spread['parent'], candidate_spread=spread['cand'],
                             control_reps=[x[0] for x in arms_runs['parent']], candidate_reps=[x[0] for x in arms_runs['cand']])
            cells[cell] = entry
        ratios = [c['ratio'] for c in cells.values() if 'ratio' in c]
        complete = len(ratios) == 6
        result = {'cells': cells, 'complete': complete}
        if complete:
            result['geomean_ratio'] = math.exp(sum(math.log(r) for r in ratios) / 6)
            result['max_cell_ratio'] = max(ratios)
            if not calibrate:
                result['within_family_limits'] = result['geomean_ratio'] <= LIMIT_GEOMEAN and result['max_cell_ratio'] <= LIMIT_CELL
        nt = list(csv.DictReader(open(raw / 'nexttoken.csv')))
        result['nexttoken_not_same'] = [f"{r['model']} {r['scheme']} {r['prompt']}: {r['verdict']}" for r in nt if r['verdict'] != 'SAME']
        result['nexttoken_rows'] = len(nt)
        summary['sessions'][session] = result
        medians[step['candidate']] = result
    with open(results / 'runs.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    with open(results / 'matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate', 'cell', 'control_valid', 'candidate_valid', 'control_median_tok_s', 'candidate_median_tok_s',
                         'ratio_control_over_candidate', 'throughput_ratio', 'control_spread', 'candidate_spread'])
        for candidate, result in medians.items():
            for cell, c in result['cells'].items():
                writer.writerow([candidate, cell, c['parent_valid'], c['cand_valid'], c.get('control_median', ''), c.get('candidate_median', ''),
                                 c.get('ratio', ''), c.get('throughput_ratio', ''), c.get('control_spread', ''), c.get('candidate_spread', '')])
    dispatch = []
    for step in state['steps']:
        if step['kind'] != 'verify' or step['candidate'] == 'control':
            continue
        screen = json.loads((evidence / 'steps' / Path(step['dir']).name / 'screen.json').read_text())
        for case, k in sorted(screen['candidate_kernels'].items()):
            dispatch.append([step['candidate'], case, screen['control_kernels'][case]['kernel'], k['kernel'],
                             int(k['kernel'] != screen['control_kernels'][case]['kernel']), k['dispatch'],
                             screen['control_kernels'][case]['kernel_median_us'], k['kernel_median_us']])
        summary['candidates'][step['candidate']] = {
            'screen': screen['verdict'], 'pdiff_pass': screen['pdiff_pass'], 'pdiff_reasons': screen['pdiff_reasons'],
            'verify_nexttoken_not_same': screen['nexttoken_not_same'],
            'decode_text': {k: v['verdict'] for k, v in screen['nexttoken'].items() if k.startswith('decode')},
            'changed_cases': screen['changed_kernels'], 'content_check_rc': step['content_check_rc'],
            'timing': step.get('timing', 'timed')}
    with open(results / 'dispatch.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate', 'case', 'control_kernel', 'candidate_kernel', 'changed', 'candidate_dispatch',
                         'control_kernel_median_us', 'candidate_kernel_median_us'])
        writer.writerows(dispatch)
    with open(results / 'nexttoken.csv', 'w', newline='') as f:
        writer = None
        for step in state['steps']:
            if step['kind'] != 'session':
                continue
            for r in csv.DictReader(open(evidence / 'steps' / Path(step['dir']).name / 'raw' / 'nexttoken.csv')):
                r = dict(session=Path(step['dir']).name, **r)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(r))
                    writer.writeheader()
                writer.writerow(r)
    ours = rates_removed(evidence / 'steps/verify-control/verify.out')
    theirs = rates_removed(historical / 'verify.out')
    differing = [f'line {i + 1}: now {a!r} | s5-c1 {b!r}' for i, (a, b) in enumerate(zip(ours, theirs)) if a != b]
    if len(ours) != len(theirs):
        differing.append(f'line counts {len(ours)} vs {len(theirs)}')
    (results / 'control-verify-vs-s5-c1.txt').write_text(
        f'{len(ours)} lines now, {len(theirs)} in s5-c1, rates removed; {len(differing)} differing\n' + '\n'.join(differing) + '\n')
    summary['control_verify_vs_s5_c1_differing'] = differing
    prefill = {}
    for line in open(evidence / 'steps/verify-control/verify.out'):
        m = re.match(r'(\S+) (\S+) default prefill_tok_s=(\S+)', line)
        if m:
            prefill[f'{m[1]}-{m[2]}'] = float(m[3])
    summary['control_verify_default_prefill_tok_s'] = prefill
    (results / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    for session, result in summary['sessions'].items():
        print(session, 'complete' if result['complete'] else 'INCOMPLETE',
              f"geomean {result.get('geomean_ratio', float('nan')):.4f} max {result.get('max_cell_ratio', float('nan')):.4f}",
              'limits', result.get('within_family_limits', '-'), 'nexttoken not SAME:', result['nexttoken_not_same'] or '-')
    print('recompute mismatches:', len(summary['recompute_mismatches']), summary['recompute_mismatches'][:5])
    print('control verify vs s5-c1 differing lines:', len(differing))


if __name__ == '__main__':
    main()
