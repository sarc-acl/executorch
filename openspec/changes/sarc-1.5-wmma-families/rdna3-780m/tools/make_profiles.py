import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('manifest', type=Path)
parser.add_argument('include', type=Path)
parser.add_argument('profiles', type=Path)
args = parser.parse_args()
data = args.manifest.read_bytes()
assert hashlib.sha256(data).hexdigest() == 'edcfa39ea7d8fac9c907a6a901754b96aa2d2894302b9c8ad623b90c2780d923'
manifest = json.loads(data)
source_bits = {'780m': 1, 'rx7600': 2, '7900xtx': 4}
storage = {'texture3d': 'kTexture3D', 'texture2d': 'kTexture2D', 'buffer': 'kBuffer'}
lines = ['// rdna3-780m: generated from the reviewed RDNA3 RADV manifest.']
profiles = []
for index, candidate in enumerate(manifest['candidates'], 1):
    name = f'780m-wmma-rdna3-{index:02d}'
    aliases = [alias['kernel_base'] for alias in candidate['dispatch_aliases']]
    assert aliases[0] == candidate['kernel_base'] and len(aliases) <= 2
    alias = json.dumps(aliases[1]) if len(aliases) == 2 else 'nullptr'
    sources = sum(source_bits[device] for device in {o['device'] for o in candidate['origins']})
    op = 'kQ4gswLinear' if candidate['scheme'] == '4w' else 'kDq8caLinear'
    io, weight = candidate['parameters']['IO_STORAGE'], candidate['parameters']['WEIGHT_STORAGE']
    lines.extend([f'{{"{name}", "{aliases[0]}", {alias},',
                  f' Op::{op}, Storage::{storage[io]}, Storage::{storage[weight]},',
                  f' {sources}, {candidate["declared_shared_bytes"]}}},'])
    profiles.append(dict(candidate=f'rdna3-{index:02d}', profile=name,
        environment=dict(ET_VK_SARC_UNVERIFIED='1', ET_VK_SARC_DEV_PROFILE='780m-final',
                         ET_VK_SARC_780M_WMMA_PROFILE=name),
        shader=candidate['shader'], aliases=candidate['dispatch_aliases'],
        sources=sorted({o['device'] for o in candidate['origins']}),
        origins=candidate['origins'], scheme=candidate['scheme'], io=io, weight=weight,
        declared_shared_bytes=candidate['declared_shared_bytes'], parameters=candidate['parameters'],
        fingerprint=candidate['source_fingerprint'],
        origin_binding='Original final-profile selection replay, with target property limits; earlier table/profile service and ordered picks retained',
        acceptance='PENDING_GPU_GATE_AND_MATRIX'))
args.include.write_text('\n'.join(lines) + '\n')
args.profiles.write_text(json.dumps(dict(
    inventory_commit='f393c3ff7ff4fc41e4651a5109a8509097c501f6',
    manifest_sha256=hashlib.sha256(data).hexdigest(), profiles=profiles), indent=2) + '\n')
