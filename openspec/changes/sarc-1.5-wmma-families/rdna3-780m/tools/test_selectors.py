import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('parent', type=Path)
parser.add_argument('topic', type=Path)
parser.add_argument('profile', type=Path)
parser.add_argument('models', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
rel = Path('openspec/changes/sarc-1.5-wmma-families/rdna3-780m')
profiles = json.loads((args.topic / rel / 'results/candidate-profiles.json').read_text())['profiles']
device = json.loads(args.profile.read_text())['capabilities']['device']
properties = device['properties']
sg = properties['VkPhysicalDeviceVulkan13Properties']
default_sg = properties['VkPhysicalDeviceVulkan11Properties']['subgroupSize']
assert device['features']['VkPhysicalDeviceVulkan13Features']['subgroupSizeControl']
shared = properties['VkPhysicalDeviceProperties']['limits']['maxComputeSharedMemorySize']
executables = {}
for name, source in [('parent', args.parent), ('topic', args.topic)]:
    selection = source / 'backends/vulkan/runtime/graph/ops/impl'
    executable = args.output / name
    command = ['c++', '-std=c++17', '-Wall', '-I' + str(source.parent),
               str(args.topic / rel / 'tools/selection_probe.cpp'),
               str(selection / 'sarc/Select.cpp'),
               *map(str, sorted((selection / 'sarc').glob('table_*.cpp'))),
               str(selection / 'sarc_dev/Overrides.cpp'), '-o', str(executable)]
    with (args.output / (name + '-compile.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    executables[name] = executable

nk = set()
for model in json.loads(args.models.read_text()):
    for op in model['linear_configurations']:
        inp, out = op['args'][0], op['args'][-1]
        if inp['dims'][-2] > 1:
            nk.add((out['dims'][-1], inp['dims'][-1]))
production = sorted(nk)
nk.update([(512, 3072), (768, 3072), (1024, 1536), (128, 2048)])
shapes = []
for op, (n, k), m, (io, weight), group in itertools.product(
        range(2), sorted(nk), (1, 128, 256, 2048, 2032),
        ((0, 1), (0, 0), (2, 1)), (16, 32, 128)):
    shapes.append([op, m, n, k, group, io, weight, 0, 1, 0, 1, int(m == 1), 1, 0])
for op in range(2):
    base = [op, 2048, 2048, 2048, 128, 2, 1, 0, 1, 0, 1, 0, 1, 0]
    for index, value in [(8, 0), (9, 1), (10, 2), (11, 1), (12, 0), (13, 1), (4, 0), (7, 2)]:
        shape = base.copy()
        shape[index] = value
        shapes.append(shape)
for op, head in itertools.product((2, 3), (64, 128)):
    m, n, k = (2048, 2560, head) if op == 2 else (2048, head, 2560)
    shapes.append([op, m, n, k, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
stdin = ''.join(' '.join(map(str, [index, *shape])) + '\n' for index, shape in enumerate(shapes))
(args.output / 'shapes.txt').write_text(stdin)
clean = {k: v for k, v in os.environ.items() if not k.startswith(
    ('ET_VK_SARC_', 'ET_VK_FORCE_', 'ET_VK_DISABLE_', 'ETVK_FORCE_'))}
checks = 0


def probe(binary, device_name, extra, label):
    command = [str(executables[binary]), device_name, str(int('radeon' in device_name)),
               str(default_sg), str(sg['minSubgroupSize']), str(sg['maxSubgroupSize']),
               '1', str(shared)]
    result = subprocess.run(command, input=stdin, text=True, capture_output=True,
                            env={**clean, **extra}, check=True)
    (args.output / (label + '.stdout')).write_text(result.stdout)
    (args.output / (label + '.stderr')).write_text(result.stderr)
    lines = result.stdout.splitlines()
    assert len(lines) == len(shapes) + 1
    return lines[0], [line.split('|')[1:] for line in lines[1:]]


control_env = dict(ET_VK_SARC_UNVERIFIED='1', ET_VK_SARC_DEV_PROFILE='780m-final')
control_softmax, control = probe('parent', 'radeon 780m', control_env, 'control-parent')
assert control_softmax == 'softmax|780m_r3'
assert probe('topic', 'radeon 780m', control_env, 'control-topic') == (control_softmax, control)
checks += len(shapes)
origins = {'780m': control}
origin_envs = {
    'rx7600': dict(ET_VK_SARC_UNVERIFIED='1', ET_VK_SARC_RX7600_PROFILE='rx7600-refine5', ET_VK_SARC_780M_PROFILE='c7'),
    '7900xtx': dict(ET_VK_SARC_UNVERIFIED='1', ET_VK_SARC_7900XTX_PROFILE='7900xtx-refine5')}
for origin, env in origin_envs.items():
    name = 'radeon rx 7600' if origin == 'rx7600' else 'radeon rx 7900 xtx'
    origins[origin] = probe('parent', name, env, 'origin-' + origin)[1]
for name in ('radeon 780m', 'radeon rx 7600', 'radeon rx 7900 xtx', 'unknown'):
    for unverified in ('0', '1'):
        env = dict(ET_VK_SARC_UNVERIFIED=unverified)
        assert probe('parent', name, env, name.replace(' ', '-') + unverified + '-old') == probe(
            'topic', name, env, name.replace(' ', '-') + unverified + '-new')
        checks += len(shapes)

rows = []
for profile in profiles:
    softmax, actual = probe('topic', 'radeon 780m', profile['environment'], profile['profile'])
    assert softmax == control_softmax
    aliases = {a['kernel_base'] for a in profile['aliases']}
    expected_op = 0 if profile['scheme'] == '4w' else 1
    expected_io = 2 if profile['io'] == 'texture3d' else 0
    expected_weight = 1 if profile['weight'] == 'texture2d' else 0
    bound = changed = production_bound = 0
    for index, shape in enumerate(shapes):
        op, m, n, k, group, io, weight, layout, half, bias, batch, gemv, width, ignore = shape
        expected = control[index]
        selected = [origins[o][index] for o in profile['sources']
                    if origins[o][index][0] in aliases]
        eligible = (control[index][0] != '-' and op == expected_op and m > 1 and n > 0
                    and k > 0 and group > 0 and not ignore and io == expected_io
                    and weight == expected_weight and profile['declared_shared_bytes'] <= shared)
        if eligible and selected:
            expected = [profile['aliases'][0]['kernel_base'], *selected[0][1:]]
            bound += 1
            production_bound += int(m == 2048 and (n, k) in production and group == 128 and io == 2
                                    and half and not bias and batch == 1 and not gemv and width)
        assert actual[index] == expected, (profile['profile'], shape, actual[index], expected)
        changed += actual[index] != control[index]
        checks += 1
    for name in ('radeon rx 7600', 'unknown'):
        expected = probe('parent', name, control_env, profile['profile'] + '-' + name.replace(' ', '-') + '-old')
        assert probe('topic', name, profile['environment'], profile['profile'] + '-' + name.replace(' ', '-') + '-new') == expected
        checks += len(shapes)
    assert bound, profile['profile']
    rows.append(dict(candidate=profile['candidate'], source_bound_shapes=bound,
                     changed_shapes=changed, production_bound_shapes=production_bound))
summary = dict(status='PASS_CPU_SELECTION', checks=checks, fixture_shapes=len(shapes),
               source_property_basis='Target queried limits replayed under source device names solely for predicate/service binding; no external-device legality or measurements inferred',
               profiles=rows, gpu_model_execution=False,
               monitoring_visibility='own_user_plus_device_busy_other_users_unobservable',
               monitoring_owner_decision='2026-10-15 03:57 UTC')
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
