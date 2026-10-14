#!/usr/bin/env python3
"""Regenerate source-only inventory at the pinned base; no builds or device access."""
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = subprocess.check_output(['git', 'rev-parse', '6178971845'], cwd=ROOT, text=True).strip()
IMPL = 'backends/vulkan/runtime/graph/ops/impl/'
GLSL = 'backends/vulkan/runtime/graph/ops/glsl/'
OV = IMPL + 'sarc_dev/Overrides.cpp'
FAMILIES = {'rdna3-radv': ['780m', 'rx7600', '7900xtx'], 'xe2': ['b580', 'b70'],
            'nvidia-ampere-ada': ['4070ti', 'orin'], 'xclipse': ['m51']}
PROFILES = {
    '780m': ('ET_VK_SARC_DEV_PROFILE', '780m-final'),
    'rx7600': ('ET_VK_SARC_RX7600_PROFILE', 'rx7600-refine5'),
    '7900xtx': ('ET_VK_SARC_7900XTX_PROFILE', '7900xtx-refine5'),
    'b580': ('ET_VK_SARC_DEV_PROFILE', 'b580-fused1-order-p13'),
    'b70': ('ET_VK_SARC_DEV_PROFILE', 'b70-fused1-order-i-n26-w16m'),
    '4070ti': ('ET_VK_SARC_DEV_PROFILE', '4070ti-fused1'),
    'orin': ('ET_VK_SARC_DEV_PROFILE', 'orin-fused1'),
    'm51': ('ET_VK_SARC_M51_PROFILE', 'c8'),
}
Q = 'sarc_linear_q4gsw_coopmat_'
D = 'sarc_linear_dq8ca_coopmat_'
XQ = 'sarc_dev_780m_x_linear_q4gsw_coopmat_'
XD = 'sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_'
BT = 'sarc_dev_linear_dq8ca_coopmat_zpg_bt_'
ORDER = 'sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_'
routes = []

def ref(path, needle):
    data = (ROOT / path).read_text()
    pos = data.index(needle)
    return f'{path}:{data.count(chr(10), 0, pos) + 1}@{BASE}'

def add(device, kernel, predicate, symbol, layer='profile', combos=None):
    routes.append(dict(device=device, kernel_base=kernel, predicate=predicate,
                       selection_ref=ref(OV if layer != 'table' else IMPL + 'sarc/table_' +
                                         ('intel' if device in ['b580', 'b70'] else
                                          'nvidia' if device in ['orin', '4070ti'] else 'amd') + '.cpp', symbol),
                       layer=layer, combos=combos))

# Ordered within each layer. A profile layer is evaluated only after its parent served the shape.
for token, pred in [
    ('t256x128k32g18s32f32cbt', 'N>=1024 && K>=4096'),
    ('t256x128k32g24s32f32cbt', 'N>=2048 && 3072<=K<4096'),
    ('t256x128k32g28s32f32cbt', 'N>=8192 && K<3072'),
    ('t128x128k32g24s32f32cbt', 'K<3072'),
    ('t128x256k32g42s32f32cbt', 'N>=1024')]:
    add('780m', XQ + token, pred, 'const Pick780m k780mRefine11[]')
add('780m', XD+'t256x64k64g48s32afmb1', 'N>=2048 && K<=4096', 'const Pick780m k780mRefine11[]')
add('780m', Q+'sweep_t128x256k32g42s32f32c', 'N>=1024; retained if c11 has no fitting pick', 'const Preference k780mRefine3[]', 'inherited-profile')
add('780m', BT+'t128x64k32g22s32', 'all; retained if c11 has no fitting pick', 'const Preference k780mRefine3[]', 'inherited-profile')
add('780m', Q+'t128x128k32g42s32f32c', 'all; retained if neither profile layer fits', '"sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c"', 'table')

PROD = 'NK in {(2048,2048),(512,2048),(8192,2048),(2048,8192),(3072,3072),(1024,3072),(8192,3072),(3072,8192),(4096,4096),(1024,4096),(14336,4096),(4096,14336)} && !gemv'
add('rx7600', 'sarc_dev_rx7600_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap4bp12', PROD, 'const PickRx7600 kRx7600Refine5[]')
add('rx7600', 'sarc_dev_rx7600_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa6pb4csha', PROD, 'const PickRx7600 kRx7600Refine5[]')
for op, kernel in [('4w', Q+'t256x128k32g24s32f32cbt'), ('8da4w', D+'zpg_t128x64k32g42s32')]:
    add('rx7600', kernel, 'all; retained outside production NK set or when pick fails fit', '"rx 7600"', 'table')

for kernel, pred in [
    (Q+'sweep_t128x128k32g42s32f32cbt', 'NK in {(2048,2048),(512,2048),(2048,8192)}'),
    (D+'zpg_sweep_t256x64k32g24s32', 'NK in {(8192,2048),(2048,2048),(8192,3072),(3072,8192),(3072,3072),(14336,4096)}'),
    (D+'zpg_sweep_t128x64k32g22s32', 'NK in {(1024,4096),(4096,4096)}'),
    (XD+'t256x64k64g48s32afmb1', 'NK in {(2048,8192)}'),
    (D+'zpg_sweep_t64x64k32g22s32', 'NK in {(512,2048)}')]:
    add('7900xtx', kernel, pred+' && !gemv', 'const Pick7900xtx kRefine5[]')
for kernel in [Q+'t256x128k32g24s32f32cbt', D+'zpg_t128x64k32g42s32']:
    add('7900xtx', kernel, 'all; retained when no fitting profile pick (includes 3B wk_wv and 8B w2 for 8da4w)', '"7900 xtx"', 'table')

for dev, limit, part in [('b580','K*N>=24000000','n13'), ('b70','K*N>=33554432','n26')]:
    if dev == 'b70':
        for token, pred in [('sweep_t128x128k32g84s16m8flw','N<=512'),('sweep_t128x128k16g82s16m8flib','all; next fitting preference')]:
            add(dev, Q+token, pred, 'const Preference kB70Fused1Order_i_n26_w16m[]')
    add(dev, ORDER+'t128x128k64g84s16m8'+part, limit, 'const Preference '+('kB580Fused1Order_p13[]' if dev=='b580' else 'kB70Fused1Order_i_n26_w16m[]'))
    add(dev, ORDER+'t128x128k64g84s16m8i', 'all; next fitting preference', 'const Preference '+('kB580Fused1Order_p13[]' if dev=='b580' else 'kB70Fused1Order_i_n26_w16m[]'))
    add(dev, Q+'t128x128k16g44s16m8fli', 'all; '+('no linear 4w preference' if dev=='b580' else 'buffer only; texture3d covered by k16g82 preference'), '"bmg g21"' if dev=='b580' else '"bmg g31"', 'table', None if dev=='b580' else ['buffer_texture2d'])
    add(dev, D+'zpg_t256x64k32g48s16m8', 'all; retained when order kernels fail fit, e.g. N%128!=0 or group_size%64!=0', '"bmg g21"' if dev=='b580' else '"bmg g31"', 'table')

for token,pred in [
    ('t256x128k16g42s32ga','N>512; texture3d'),
    ('t128x128k16g24s32ga','all; next fitting texture3d row'),
    ('t128x256k16g42s32ga','N>512; buffer/texture2d'),
    ('t128x128k16g42s32ga','all; next fitting buffer row')]:
    add('4070ti', Q+token, pred, '"'+Q+token+'"', 'table')
add('4070ti', D+'zpgtr_t128x128k64g44s32mk32ra', 'all; row-major layout; no linear preference in k4070tiRefine1', '"'+D+'zpgtr_t128x128k64g44s32mk32ra"', 'table')

add('orin', 'sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32c', 'K>8192; parent table must serve orin_measured', 'const Preference kOrinRefine5[]')
add('orin', Q+'orin_t256x128k16g42s32bt', 'K<=8192 && (M!=256 || N%2048==0); parent table must serve orin_measured', 'const Preference kOrinRefine5[]')
add('orin', D+'zpgtr_orin_bf_t128x128k64g24s32mk32ra', 'all; parent table must serve orin_measured; row-major', 'const Preference kOrinRefine5[]')
add('orin', Q+'t128x128k16g22s32', 'orin_measured && K<=8192; retained when t256 preference cannot fit (e.g. M=128)', '"'+Q+'t128x128k16g22s32"', 'table')

add('orin', D+'zpgtr_t128x128k64g44s32mk32ra', 'orin_measured; runtime row-major unaligned fallback bypasses profile and calls select_table(ignore_alignment=true)', '"'+D+'zpgtr_t128x128k64g44s32mk32ra"', 'table', ['texture3d_texture2d'])

add('m51', 'sarc_dev_m51_linear_q4gsw_coopmat_bz_t128x128k32g42s32f32xp', 'all; preceding selector must serve shape', 'const PickM51 kM51PicksC8[]')
add('m51', BT+'t128x64k32g22s32', 'all; served with layout reopened if needed; final layout must match', 'const PickM51 kM51PicksC8[]')
add('m51', Q+'t128x128k16g22s32f32xp', 'all; retained when bz cannot fit (e.g. group_size=16 or buffer weight)', '"'+Q+'t128x128k16g22s32f32xp"', 'table')
add('m51', D+'zpgtr_t128x64k32g42s32', 'all; retained when zpg_bt cannot fit or fixed row-major layout excludes it', '"'+D+'zpgtr_t128x64k32g42s32"', 'table')

# Parse shader YAML, including inherited defaults and explicit iteration lists.
variants = {}
for path in sorted((ROOT/GLSL).glob('sarc*/*.yaml')):
    if 'linear' not in path.name:
        continue
    data = yaml.safe_load(path.read_text())
    for family, spec in data.items():
        defaults = spec.get('parameter_names_with_default_values', {})
        for entry in spec.get('shader_variants', []):
            params = defaults | entry
            name = entry['NAME']
            if name in variants:
                raise ValueError(name)
            variants[name] = (path, params, family)

source_files = sorted((ROOT/(IMPL+'sarc')).glob('table_*.cpp')) + sorted((ROOT/(IMPL+'sarc_dev')).glob('*.cpp')) + sorted((ROOT/(IMPL+'sarc_dev')).glob('*.inc'))

def row_info(kernel, device, layer):
    for path in source_files:
        text = path.read_text()
        for match in re.finditer('"'+re.escape(kernel)+'"', text):
            # A row has a storage mask and Status before its terminating semicolon/row.
            tail = text[match.end():match.end()+420]
            if layer == 'table':
                device_substr = {'780m':'780m','rx7600':'rx 7600','7900xtx':'7900 xtx','b580':'bmg g21','b70':'bmg g31','4070ti':'4070 ti super','orin':'tegra orin','m51':'xclipse'}[device]
                start = text.rfind('{\"', 0, match.start())
                if not text[start:].startswith('{\"'+device_substr+'\"'):
                    continue
            if 'Status::' not in tail or not re.search(r'kTex3dTex2d|kBufTex2d|kBufBuf', tail.split('Status::')[0]):
                continue
            # Exclude preference literals followed by another registration later.
            if not re.match(r'\s*,\s*(?:\{|tile\(|tile_mnk\(|dq_tile\(|xp_tile\(|k\w+)', tail):
                continue
            tail = tail.split('Status::')[0]
            combos = [c for bit,c in [('kTex3dTex2d','texture3d_texture2d'),('kBufTex2d','buffer_texture2d'),('kBufBuf','buffer_buffer')] if bit in tail]
            return f'{path.relative_to(ROOT)}:{text.count(chr(10), 0, match.start())+1}@{BASE}', combos
    raise ValueError('No row: '+kernel)

def memory(p, body):
    m,n,k,mm,sgy = (p[x] for x in ['WG_TILE_M','WG_TILE_N','WG_TILE_K','MMA_M','SG_GRID_Y'])
    tex = p['IO_STORAGE']=='texture3d'
    if 'WEIGHT_NBITS' not in p:
        if p.get('FRAG_LAYOUT'):
            ab = 4*(m*k+k*n)
        elif p.get('B_COLMAJOR'):
            ab = 4*(m*(k+p.get('A_PAD_H',8))+n*(k+p.get('BT_PAD_H',8)))
        else:
            ab = 4*(m*(k+8)+k*(n+8))
        c = 2*(m if p.get('CSH_FULL') else mm if p.get('CSH_BAND') else sgy*mm)*n if tex else 0
        if p.get('CSH_POOL') and tex:
            total = max(ab,c)
            formula = f'max(AB={ab}, C={c})'
        else:
            c = 0 if p.get('CSH_IN_ASH') else c
            total = ab+c
            formula = f'AB={ab} + C={c}'
        return total, formula
    slabs = k//p['MMA_K']
    ap,bp = p.get('A_PITCH',p['MMA_K']//4),p.get('B_PITCH',p['MMA_K']//4)
    slices = 1 if p.get('ONE_SLICE') else 2
    ab = slices*slabs*(m*ap+p.get('A_PAD',0)+n*bp)*4
    meta = 8*m+12*n
    c = 2*sgy*mm*n if tex and not p.get('CSH_IN_ASH') else 0
    return ab+meta+c, f'AB={ab} + params={meta} + C={c}'

inventory=[]
manifest=defaultdict(dict)
for index,r in enumerate(routes,1):
    family = next(f for f,ds in FAMILIES.items() if r['device'] in ds)
    rr,combos = row_info(r['kernel_base'], r['device'], r['layer'])
    if r['combos'] is not None:
        assert set(r['combos']) <= set(combos)
        combos=r['combos']
    for combo in combos:
        name=r['kernel_base']+'_'+combo+'_half'
        path,p,shader_family=variants[name]
        wrapper = path.with_suffix('.glsl')
        includes=re.findall(r'#include "([^"]+\.glslh)"',wrapper.read_text())
        body_matches=[]
        for inc in includes:
            body_matches += [x for x in (ROOT/GLSL).rglob(Path(inc).name) if 'body' in x.name]
        assert len(body_matches)==1,(name,body_matches)
        body=str(body_matches[0].relative_to(ROOT))
        total,formula=memory(p,body)
        scheme='4w' if 'q4gsw' in r['kernel_base'] else '8da4w'
        mma=f"{p['MMA_M']}x{p['MMA_N']}x{p['MMA_K']}"
        acc='fp32' if p.get('ACC_FP32') else 'fp16-group/fp32-total' if p.get('ACC_GROUP_FP32') else 'fp16'
        if scheme=='8da4w':
            acc='int32-group/fp32-total'
        layout='fp16' if scheme=='4w' else 'row-major' if 'zpgtr' in r['kernel_base'] else '4h4w'
        rid=f'R{index:02}'
        record=dict(route=rid, family=family, device=r['device'], profile=PROFILES[r['device']][1],
                    scheme=scheme,layer=r['layer'],predicate=r['predicate'],kernel_base=r['kernel_base'],shader=name,
                    io=p['IO_STORAGE'],weight=p['WEIGHT_STORAGE'],activation_layout=layout,
                    wg_tile=f"{p['WG_TILE_M']}x{p['WG_TILE_N']}x{p['WG_TILE_K']}",
                    sg_grid=f"{p['SG_GRID_X']}x{p['SG_GRID_Y']}",subgroup=p['SUBGROUP_SIZE'],
                    wg_invocations=p['SG_GRID_X']*p['SG_GRID_Y']*p['SUBGROUP_SIZE'],
                    mma=mma,accumulation=acc,declared_shared_bytes=total,shared_formula=formula,
                    selection_ref=r['selection_ref'],row_ref=rr,yaml_ref=ref(str(path.relative_to(ROOT)),name),
                    wrapper_ref=ref(str(wrapper.relative_to(ROOT)),'#version'),body_ref=ref(body,'const uint NUM_SUBGROUPS'),
                    target_legality='PENDING_QUERY; see LEGALITY.md',commit=BASE)
        inventory.append(record)
        if name not in manifest[family]:
            manifest[family][name]=dict(shader=name,scheme=scheme,kernel_base=r['kernel_base'],
                parameters={k:v for k,v in p.items() if k!='NAME'},declared_shared_bytes=total,shared_formula=formula,
                activation_layout=layout,mma=mma,accumulation=acc,origins=[],row_ref=rr,yaml_ref=record['yaml_ref'],
                wrapper_ref=record['wrapper_ref'],body_ref=record['body_ref'],
                member_legality={d:'PENDING_QUERY: exact MMA types/scope, subgroup control, shared memory, workgroup limits, features, storage and shape; no target queried' for d in FAMILIES[family]})
        manifest[family][name]['origins'].append(dict(device=r['device'],route=rid,layer=r['layer'],predicate=r['predicate'],row_ref=rr,selection_ref=r['selection_ref']))

with (OUT/'inventory.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=inventory[0]);w.writeheader();w.writerows(inventory)
(OUT/'manifests').mkdir(exist_ok=True)
for family,entries in manifest.items():
    (OUT/'manifests'/f'{family}.json').write_text(json.dumps(dict(source_commit=BASE,family=family,members=FAMILIES[family],
        status='SOURCE_INVENTORY_ONLY; conditional candidates, no legality confirmation or performance selection',
        deduplication_key='full shader name within family (body + compile-time parameters + storage); shared origins retained',
        candidates=list(entries.values())),indent=2)+'\n')
# Hash only inputs used by the inventory, with stable repository-relative names.
refs={x.split('@')[0].rsplit(':',1)[0] for r in inventory for k,x in r.items() if k.endswith('_ref')}
refs.update([OV,IMPL+'sarc/Select.cpp',IMPL+'sarc/Select.h',IMPL+'sarc/Dq8caCoopmat.cpp',IMPL+'sarc/Q4gswCoopmat.cpp','sarc/README.md'])
(OUT/'source-hashes.csv').write_text('commit,path,sha256\n'+''.join(f'{BASE},{p},{hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}\n' for p in sorted(refs)))
with (OUT/'legality-matrix.csv').open('w') as f:
    fields=['family','shader','target','status','required_shared_bytes','required_subgroup','required_wg_invocations','required_subgroups','mma','mma_types','shape_requirements','source_ref','commit']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for family,entries in manifest.items():
        for c in entries.values():
            p=c['parameters']
            for target in FAMILIES[family]:
                w.writerow(dict(family=family,shader=c['shader'],target=target,status='PENDING_QUERY',
                    required_shared_bytes=c['declared_shared_bytes'],required_subgroup=p['SUBGROUP_SIZE'],
                    required_wg_invocations=p['SG_GRID_X']*p['SG_GRID_Y']*p['SUBGROUP_SIZE'],
                    required_subgroups=p['SG_GRID_X']*p['SG_GRID_Y'],mma=c['mma'],
                    mma_types='sint8,sint8,sint32,sint32' if c['scheme']=='8da4w' else 'fp16,fp16,fp32,fp32' if p.get('ACC_FP32') else 'fp16,fp16,fp16,fp16',
                    shape_requirements=f"M%{p['WG_TILE_M']}=0; N%{p['WG_TILE_N']}=0; K%{p['WG_TILE_K']}=0; group_size%{p['WG_TILE_K']}=0; row/route/storage/feature checks in LEGALITY.md",
                    source_ref=c['body_ref'],commit=BASE))

print(f'{len(routes)} routes; {len(inventory)} inventory rows')
for family,entries in manifest.items():
    print(f'{family}: {len(entries)} storage-specific candidates; {len({x["kernel_base"] for x in entries.values()})} kernel bases')
