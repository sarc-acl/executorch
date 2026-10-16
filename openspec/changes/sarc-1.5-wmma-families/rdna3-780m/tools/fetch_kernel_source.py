import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.parse
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('url')
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
expected = subprocess.check_output(
    ['rpm', '-q', '--qf', '%{SOURCERPM}', 'kernel-core-' + os.uname().release], text=True)
name = Path(urllib.parse.urlparse(args.url).path).name
assert name == expected
package = args.output / name
digest = hashlib.sha256()
with urllib.request.urlopen(args.url, timeout=30) as response, package.open('xb') as output:
    while block := response.read(1024 * 1024):
        digest.update(block)
        output.write(block)
signature = subprocess.run(['rpm', '-K', str(package)], text=True, capture_output=True)
(args.output / 'signature.txt').write_text(signature.stdout + signature.stderr)
assert signature.returncode == 0 and 'NOKEY' not in signature.stdout + signature.stderr
payload = args.output / 'payload.cpio'
with payload.open('xb') as output:
    subprocess.run(['rpm2cpio', str(package)], stdout=output, check=True)
listing = subprocess.check_output(['cpio', '-it', '--quiet', '-F', str(payload)], text=True)
(args.output / 'contents.txt').write_text(listing)
assert all(not Path(path).is_absolute() and '..' not in Path(path).parts for path in listing.splitlines())
sources = args.output / 'sources'
sources.mkdir()
subprocess.run(['cpio', '-idm', '--quiet', '--no-absolute-filenames', '-F', str(payload)],
               cwd=sources, check=True)
suffixes = (
    'drivers/gpu/drm/amd/pm/swsmu/inc/pmfw_if/smu13_driver_if_v13_0_4.h',
    'drivers/gpu/drm/amd/pm/swsmu/smu13/smu_v13_0_4_ppt.c',
    'drivers/gpu/drm/amd/pm/swsmu/smu13/smu_v13_0.c',
    'drivers/gpu/drm/amd/include/kgd_pp_interface.h')
selected = []
for archive in sorted(sources.glob('*.tar.*')):
    with tarfile.open(archive) as tar:
        for member in tar:
            for suffix in suffixes:
                if member.isfile() and member.name.endswith('/' + suffix):
                    destination = args.output / 'selected' / archive.name / suffix
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    data = tar.extractfile(member).read()
                    destination.write_bytes(data)
                    selected.append(dict(archive=archive.name, member=member.name,
                                         selected=str(destination), sha256=hashlib.sha256(data).hexdigest()))
record = dict(status='SOURCE_RETRIEVED_INTERPRETATION_PENDING', url=args.url,
              running_kernel=os.uname().release, source_rpm=expected,
              sha256=digest.hexdigest(), signature=signature.stdout.strip(), selected=selected,
              patches_applied=False, gpu_model_execution=False,
              monitoring_visibility='own_user_plus_device_busy_other_users_unobservable',
              monitoring_owner_decision='2026-10-15 03:57 UTC')
(args.output / 'source.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
