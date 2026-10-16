import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('export', type=Path)
args = parser.parse_args()
export = args.export.resolve()
modules = {}
for line in (export / 'MANIFEST').read_text().splitlines():
    name, *fields = line.split()
    fields = dict(field.split('=', 1) for field in fields if '=' in field)
    modules[name] = fields
expected = {'.': (export / 'COMMIT').read_text().strip()}
results = []
errors = []
for name, fields in modules.items():
    commit = expected[name]
    if name != '.' and fields['pinned'] != commit:
        raise ValueError(f'Wrong submodule commit: {name}')
    store = fields['store']
    tree = subprocess.check_output(['git', f'--git-dir={store}', 'rev-parse',
                                    f'{commit}^{{tree}}'], text=True).strip()
    if tree != fields['tree']:
        raise ValueError(f'Wrong tree: {name}')
    entries = subprocess.check_output(['git', f'--git-dir={store}', 'ls-tree',
                                       '-rz', commit]).split(b'\0')
    count = 0
    for entry in filter(None, entries):
        metadata, relative = entry.split(b'\t', 1)
        mode, kind, blob = metadata.decode().split()
        relative = os.fsdecode(relative)
        full_name = relative if name == '.' else f'{name}/{relative}'
        if kind == 'commit':
            expected[full_name] = blob
            continue
        path = export / 'executorch' / full_name
        try:
            data = os.fsencode(os.readlink(path)) if mode == '120000' else path.read_bytes()
            actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if actual != blob:
                errors.append(dict(path=full_name, expected=blob, actual=actual))
            count += 1
        except OSError as exc:
            errors.append(dict(path=full_name, error=str(exc)))
    results.append(dict(path=name, commit=commit, tree=tree, files=count))
if set(expected) != set(modules):
    errors.append(dict(missing=sorted(set(expected) - set(modules)),
                       extra=sorted(set(modules) - set(expected))))
print(json.dumps(dict(modules=results, files=sum(r['files'] for r in results),
                      mismatches=errors, status='PASS' if not errors else 'FAIL'), indent=2))
raise SystemExit(1 if errors else 0)
