#!/usr/bin/env python3
"""lean_record.py <artifact root> <change dir> <session...>: small evidence of the lean matrix.

Writes results/matrix-summary.csv (one row per session and cell, valid-run medians from evidence.json),
results/matrix-dispatch.csv (linear kernels each arm dispatched, from the traced ETDump) and the table between
the matrix markers of STATUS.md. Raw rows stay in <artifact root>/runs.csv."""
import csv
import json
import pathlib
import re
import sys

root, change = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sessions = sys.argv[3:]
summary, dispatch, table = [], [], []
for session in sessions:
    stage = root / 'stage' / session
    label = (stage / 'correctness.txt').read_text().strip() if (stage / 'correctness.txt').exists() else ('RUNNING' if stage.exists() else 'NOT_STARTED')
    evidence = json.loads((stage / 'evidence.json').read_text()) if (stage / 'evidence.json').exists() else {}
    names = {}
    if (stage / 'dispatch/dispatch.csv').exists():
        for row in csv.DictReader((stage / 'dispatch/dispatch.csv').open()):
            if 'linear' in row['kernel']:
                dispatch.append({'session': session, **{k: row[k] for k in ('model', 'scheme', 'arm', 'kernel', 'M', 'N', 'K', 'events')}})
                if row['arm'] == 'cand':
                    names.setdefault(row['scheme'], set()).add(re.sub(r'_(texture3d|buffer)_texture2d_half$', '', row['kernel']))
    ratios = []
    for cell in evidence.get('cells', []):
        summary.append({'session': session, 'correctness': label, 'model': cell['model'], 'scheme': cell['scheme'],
                        'control_median_tok_s': f"{cell['control_tok_s']:.2f}",
                        'candidate_median_tok_s': f"{cell['candidate_tok_s']:.2f}",
                        'candidate_over_control': f"{cell['candidate_tok_s'] / cell['control_tok_s']:.4f}",
                        'slowdown': f"{cell['slowdown']:.4f}",
                        'slowdown_boot95_lo': f"{cell['slowdown_bootstrap_95'][0]:.4f}",
                        'slowdown_boot95_hi': f"{cell['slowdown_bootstrap_95'][1]:.4f}",
                        'valid_control': cell['valid_control'], 'valid_candidate': cell['valid_candidate']})
        ratios.append(f"{cell['candidate_tok_s'] / cell['control_tok_s']:.3f}")
    geomean = evidence.get('geomean_slowdown')
    linear = '; '.join(f"{q}: {', '.join(sorted(v))}" for q, v in sorted(names.items())) or '-'
    table.append(f"| {session} | {linear} | {label} | {' / '.join(ratios) or '-'} | "
                 f"{'-' if geomean is None else f'{1 / (1 + geomean):.4f}'} | "
                 f"{'-' if geomean is None else str(evidence.get('point_estimates_within_member_margin'))} |")

results = change / 'results'
results.mkdir(exist_ok=True)
for name, rows in (('matrix-summary.csv', summary), ('matrix-dispatch.csv', dispatch)):
    if rows:
        with (results / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

status = change / 'STATUS.md'
text = status.read_text()
begin, end = '<!-- matrix begin -->', '<!-- matrix end -->'
block = '\n'.join([begin,
                   '| Session | Candidate linear kernels (dispatched, cand arm) | Correctness | Cand/control tok/s, cells 1B 4w, 1B 8da4w, 3B 4w, 3B 8da4w, 8B 4w, 8B 8da4w | Six-cell geomean cand/control | Within 3 %/8 % |',
                   '|---|---|---|---|---:|---|', *table, end])
text = re.sub(re.escape(begin) + '.*?' + re.escape(end), lambda _: block, text, flags=re.S)
status.write_text(text)
print('\n'.join(table))
