import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('manifest', type=Path)
p.add_argument('device_profile', type=Path)
p.add_argument('matrices', type=Path)
p.add_argument('models', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
manifest = json.loads(a.manifest.read_text())
d = json.loads(a.device_profile.read_text())['capabilities']['device']
props, features = d['properties'], d['features']
limits = props['VkPhysicalDeviceProperties']['limits']
sg = props['VkPhysicalDeviceVulkan13Properties']
assert props['VkPhysicalDeviceProperties']['vendorID'] == 0x1002
assert props['VkPhysicalDeviceVulkan12Properties']['driverID'] == 'VK_DRIVER_ID_MESA_RADV'
for struct, keys in {
    'VkPhysicalDeviceCooperativeMatrixFeaturesKHR': ['cooperativeMatrix'],
    'VkPhysicalDeviceVulkan11Features': ['storageBuffer16BitAccess'],
    'VkPhysicalDeviceVulkan12Features': ['storageBuffer8BitAccess', 'shaderFloat16', 'shaderInt8'],
    'VkPhysicalDeviceVulkan13Features': ['subgroupSizeControl', 'computeFullSubgroups'],
}.items():
    for key in keys:
        assert features[struct][key], (struct, key)
assert 'VK_SHADER_STAGE_COMPUTE_BIT' in sg['requiredSubgroupSizeStages']
assert 'VK_SHADER_STAGE_COMPUTE_BIT' in props['VkPhysicalDeviceCooperativeMatrixPropertiesKHR']['cooperativeMatrixSupportedStages']
for fmt in ('R16G16B16A16_SFLOAT', 'R8G8B8A8_UINT', 'R8G8B8A8_SINT', 'R32G32B32A32_SINT'):
    flags = d['formats']['VK_FORMAT_' + fmt]['VkFormatProperties']['optimalTilingFeatures']
    for flag in ('SAMPLED_IMAGE', 'STORAGE_IMAGE', 'TRANSFER_SRC', 'TRANSFER_DST'):
        assert 'VK_FORMAT_FEATURE_' + flag + '_BIT' in flags, (fmt, flag)
matrices = json.loads(a.matrices.read_text())['matrices']
models = json.loads(a.models.read_text())
fields = ['candidate', 'fingerprint', 'shader', 'scheme', 'declared_shared_bytes', 'workgroup_invocations',
          'queried_requirements', 'timed_shape_fit_count', 'production_storage', 'remaining', 'origin_routes']
with a.output.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    for ix, candidate in enumerate(manifest['candidates'], 1):
        q = candidate['parameters']
        failures = []
        invocations = q['SG_GRID_X'] * q['SG_GRID_Y'] * q['SUBGROUP_SIZE']
        if candidate['declared_shared_bytes'] > limits['maxComputeSharedMemorySize']:
            failures.append('shared-memory')
        if invocations > min(limits['maxComputeWorkGroupInvocations'], limits['maxComputeWorkGroupSize'][0]):
            failures.append('workgroup')
        if q['SG_GRID_X'] * q['SG_GRID_Y'] > sg['maxComputeWorkgroupSubgroups']:
            failures.append('subgroup-count')
        if not sg['minSubgroupSize'] <= q['SUBGROUP_SIZE'] <= sg['maxSubgroupSize']:
            failures.append('subgroup-size')
        operand, accumulator = (0, 1) if candidate['scheme'] == '4w' else (3, 5)
        if not any(m['MSize'] == q['MMA_M'] and m['NSize'] == q['MMA_N'] and m['KSize'] == q['MMA_K']
                   and m['AType'] == m['BType'] == operand and m['CType'] == m['ResultType'] == accumulator
                   and m['scope'] == 3 and not m['saturatingAccumulation'] for m in matrices):
            failures.append('mma-types')
        shapes = set()
        for model in models:
            if ('_8da4w.pte' in model['path']) != (candidate['scheme'] == '8da4w'):
                continue
            for op in model['linear_configurations']:
                args = op['args']
                inp, out = args[0], args[-1]
                if inp['dims'][-2] <= 1:
                    continue
                group = args[3] if candidate['scheme'] == '4w' else args[6]
                m, n, k = inp['dims'][-2], out['dims'][-1], inp['dims'][-1]
                if m % q['WG_TILE_M'] or n % q['WG_TILE_N'] or k % q['WG_TILE_K'] or group % q['WG_TILE_K']:
                    continue
                # Actual packed weights are chosen at graph initialization; this is the source prediction.
                weight = 'buffer' if (k + 3)//4 > limits['maxImageDimension2D'] or (n + 7)//8 > limits['maxImageDimension2D'] else 'texture2d'
                if q['IO_STORAGE'] == 'texture3d' and q['WEIGHT_STORAGE'] == weight:
                    shapes.add((m, n, k))
        w.writerow(dict(candidate=f'rdna3-{ix:02d}', fingerprint=candidate['source_fingerprint'],
                        shader=candidate['shader'], scheme=candidate['scheme'],
                        declared_shared_bytes=candidate['declared_shared_bytes'], workgroup_invocations=invocations,
                        queried_requirements='EXCLUDED:'+';'.join(failures) if failures else 'PASS_DECLARED_REQUIREMENTS',
                        timed_shape_fit_count=len(shapes),
                        production_storage='PREDICTED_MATCH' if shapes else 'ALTERNATE_CORRECTNESS_ONLY',
                        remaining='exact-module-allocation;runtime-storage;origin-predicate-binding;gates;timing',
                        origin_routes=';'.join(sorted({o['route'] for o in candidate['origins']}))))
