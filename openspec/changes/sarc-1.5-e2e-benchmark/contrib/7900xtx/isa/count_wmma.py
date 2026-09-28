#!/usr/bin/env python3
"""Count v_wmma_* per ISA file (total, and inside each backward-branch loop body) for the dumps here.
Usage: python3 count_wmma.py */*.isa.txt   (prints CSV: driver,file,opcode,total,per_loop_iteration)"""
import collections, re, sys

LABEL = re.compile(r'^\s*(\.?L?BB[0-9_]+|_L[0-9]+):')
BRANCH = re.compile(r'\bs_(?:cbranch_\w+|branch)\s+(\.?L?BB[0-9_]+|_L[0-9]+)')
WMMA = re.compile(r'^\s*(v_wmma_\S+)', re.M)


def norm(label):
    return label.lstrip('.').replace('LBB', 'BB')


def loops(text):
    blocks = collections.OrderedDict(entry=[])
    cur = 'entry'
    for line in text.splitlines():
        m = LABEL.match(line)
        if m:
            cur = norm(m.group(1))
            blocks[cur] = []
        else:
            blocks[cur].append(line)
    names = list(blocks)
    pos = {n: i for i, n in enumerate(names)}
    out = set()
    for i, n in enumerate(names):
        for line in blocks[n]:
            m = BRANCH.search(line)
            if m and norm(m.group(1)) in pos and pos[norm(m.group(1))] <= i:
                body = '\n'.join(x for k in names[pos[norm(m.group(1))]:i + 1] for x in blocks[k])
                c = len(WMMA.findall(body))
                if c:
                    out.add(c)
    return sorted(out)


print('driver,file,opcode,total,wmma_in_loop_bodies')
for path in sys.argv[1:]:
    text = open(path, errors='replace').read()
    for op, n in sorted(collections.Counter(WMMA.findall(text)).items()):
        print(f"{path.split('/')[-2]},{path.split('/')[-1]},{op},{n},{'/'.join(map(str, loops(text)))}")
