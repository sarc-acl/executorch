#!/usr/bin/env python3
"""Join reviewed modules, queried properties and serialized model resource bounds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('build', type=Path)
    args = parser.parse_args()
    a = args.artifacts
    physical = (a / 'legality/current-vulkaninfo.txt').read_text()
    original = (a / 'preflight/vulkaninfo.txt').read_text()
    def value(name, text=physical):
        match = re.search(r'^\s*' + re.escape(name) + r'\s*=\s*(.+)$', text, re.M)
        if not match:
            raise ValueError('missing property: ' + name)
        return match[1].strip()
    for name in ('deviceUUID', 'driverVersion', 'vendorID', 'driverID'):
        assert value(name) == value(name, original), name
    def enabled(name):
        assert value(name) == 'true', name
    for name in ('cooperativeMatrix', 'subgroupSizeControl', 'computeFullSubgroups',
                 'shaderFloat16', 'shaderInt8', 'storageBuffer16BitAccess',
                 'storageBuffer8BitAccess', 'vulkanMemoryModel'):
        enabled(name)
    assert 'VK_KHR_vulkan_memory_model ' in physical
    assert 'VK_KHR_cooperative_matrix ' in physical
    assert int(value('subgroupSize')) == 32
    assert int(value('minSubgroupSize')) == int(value('maxSubgroupSize')) == 32
    matrices = json.loads((a / 'preflight/coopmat.json').read_text())['matrices']
    requirements = {x['shader']: x for x in json.loads((a / 'recovery/parent-20261015-pinned1/module-requirements/requirements.json').read_text())}
    candidates = json.loads(args.inventory.read_text())['candidates']
    capabilities = {'Shader', 'Float16', 'Int8', 'GroupNonUniform',
                    'StorageBuffer16BitAccess', 'StorageBuffer8BitAccess',
                    'VulkanMemoryModel', 'CooperativeMatrixKHR'}
    entries = []
    for candidate in candidates:
        p = candidate['parameters']; module = requirements[candidate['shader']]
        data = (args.build / 'build/vulkan_compute_shaders' / (candidate['shader'] + '.spv')).read_bytes()
        assert hashlib.sha256(data).hexdigest() == module['module_sha256']
        assert set(module['capabilities']) <= capabilities
        assert module['spirv_val_rc'] == 0 and module['declared_matches_module']
        assert module['shared_type_bytes'] <= int(value('maxComputeSharedMemorySize'))
        groups = p['SG_GRID_X'] * p['SG_GRID_Y']; threads = groups * p['SUBGROUP_SIZE']
        assert threads <= int(value('maxComputeWorkGroupInvocations'))
        assert groups <= int(value('maxComputeWorkgroupSubgroups'))
        axes = re.search(r'maxComputeWorkGroupSize: count = 3\s+(\d+)\s+(\d+)\s+(\d+)', physical)
        assert axes and threads <= int(axes[1])
        matching = [x for x in matrices if x['M'] == p['MMA_M'] and x['N'] == p['MMA_N'] and x['K'] == p['MMA_K'] and x['scope'] == 3 and not x['saturating']]
        if candidate['scheme'] == '4w':
            accumulator = 1 if p.get('ACC_FP32') else 0
            assert any(x['A'] == x['B'] == 0 and x['C'] == x['result'] == accumulator for x in matching)
        else:
            assert any(x['A'] == x['B'] == 3 and x['C'] == x['result'] == 5 for x in matching)
        entries.append({'shader': candidate['shader'], 'module_sha256': module['module_sha256'],
                        'static_module_device_requirements': 'PASS', 'shared_bytes': module['shared_type_bytes'],
                        'workgroup_invocations': threads,
                        'coverage': 'E2E_TEXTURE3D' if p['IO_STORAGE'] == 'texture3d' else 'ALTERNATE_STORAGE_CORRECTNESS'})
    formats = json.loads((a / 'resource-checks/image-properties.json').read_text())['results']
    def image(extent, fmt, image_type):
        matches = [x for x in formats if x['format'] == fmt and x['image_type'] == image_type and x['tiling'] == 0 and x['usage'] == 15]
        assert len(matches) == 1 and matches[0]['rc'] == 0
        assert all(0 < v <= limit for v, limit in zip(extent, matches[0]['max_extent'])), extent
    for fmt in ('R16G16B16A16_SFLOAT', 'R32G32B32A32_SINT'):
        image([1, 1, 1], fmt, 2)
    resource = json.loads((a / 'resource-checks/allocation-precheck.json').read_text())
    models = json.loads((a / 'resource-checks/model-resources.json').read_text())
    buffers_limit = int(value('maxStorageBufferRange'))
    checks = []
    for model, previous in zip(models, resource['models']):
        assert model['file'] == previous['file']
        for x in previous['images']:
            image(x['packed_extent'], x['format'], {1: 2, 2: 1}[x['storage']])
        unresolved = previous['unresolved_default_values']
        assert len(unresolved) == 1 and unresolved[0]['dims'] == [1] and unresolved[0]['dtype'] == 3
        image([1, 1, 1], 'R32G32B32A32_SINT', 2)
        weights = []
        max_buffer_bound = 0
        for delegate in model['delegates']:
            for t in delegate['tensors']:
                # A conservative four-byte bound covers these serialized dtypes,
                # packed activations, norm intermediates and staging buffers.
                assert t['dtype'] in (1, 2, 3, 4), t
                bound = math.prod(t['dims']) * 4
                assert bound < buffers_limit, t
                max_buffer_bound = max(bound, max_buffer_bound)
            for op in delegate['linear_embedding_nodes']:
                if 'linear_' not in op['op']:
                    continue
                arg = op['args'][3 if 'dq8ca' in op['op'] else 1]
                n, k_half = arg['dims']; k = 2 * k_half
                # QuantizedLinear.cpp: N/8 by K/4 texels, int32 RGBA.
                extent = [(k + 3) // 4, (n + 7) // 8, 1]
                image(extent, 'R32G32B32A32_SINT', 1)
                weights.append(extent)
        # SDPA.cpp bounds the attention buffer by maxStorageBufferRange.
        # Even 64 heads at the full validated context fit this device's range.
        context = model['methods']['get_max_context_len'][0]
        attention_bound = 64 * context * context * 4
        assert attention_bound < buffers_limit
        checks.append({'file': model['file'], 'explicit_images': len(previous['images']),
                       'resolved_default_scalar_extent': [1, 1, 1], 'linear_weight_images': len(weights),
                       'max_linear_weight_extent': [max(x[i] for x in weights) for i in range(3)],
                       'max_serialized_buffer_bound': max_buffer_bound,
                       'conservative_attention_buffer_bound': attention_bound,
                       'static_resource_bounds': 'PASS',
                       'actual_allocation_and_correctness': 'PENDING_GPU_GATE'})
    report = {'source_commit': json.loads((args.build / 'manifest.json').read_text())['source_commit'],
              'property_signature': {k: value(k) for k in ('deviceUUID', 'vendorID', 'driverID', 'driverVersion', 'subgroupSize', 'maxComputeSharedMemorySize')},
              'candidates': entries, 'models': checks,
              'scope': 'Static launch prerequisites; full runtime allocation/correctness remains a gate result',
              'route_predicates': 'Committed selector preserves storage, shape, group, alignment and parent-service boundaries; CPU evidence separately reviewed',
              'logical_feature_execution': 'PENDING_GPU_GATE'}
    (a / 'legality/join.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: 11 static module/device joins and six model resource-bound joins; GPU gates pending')

if __name__ == '__main__':
    main()
