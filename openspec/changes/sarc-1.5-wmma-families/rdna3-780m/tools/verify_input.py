import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('inputs', type=Path)
parser.add_argument('snapshot', type=Path)
args = parser.parse_args()
p = json.loads((args.inputs / 'REVIEWED-INPUT.json').read_text())
items = [(args.inputs / 'inventory.tar', p['archive_sha256'])]
items += [(args.snapshot / name, digest) for name, digest in p['files'].items()]
for path, want in items:
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f'FAIL {path.name}: {got} != {want}')
    print(f'PASS {path.name} {got}')
rows = list(csv.DictReader((args.snapshot / 'source-hashes.csv').open()))
for row in rows:
    if row['commit'] != p['source_base']:
        raise SystemExit(f'Unexpected source commit: {row["commit"]}')
    path = Path(row['path'])
    obj = subprocess.check_output(['git', 'show', f'{row["commit"]}:{path}'])
    for data in (obj, path.read_bytes()):
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise SystemExit(f'FAIL source: {path}')
print(f'PASS {len(rows)} pinned git objects and checkout files')
