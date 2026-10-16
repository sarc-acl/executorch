#!/usr/bin/env python3
"""SDPA tier check of one finalist (R7), the criteria of the Orin campaign's gate_check.py sdpa unchanged.

  sdpa_check.py <sdpa stage> <env file>

Tiers all, extended, full 12 passes each, peaked and fused 3 passes each (an orin-fused profile), every pass with
the tier's case count, every case PASSED with mismatches=0 and a [sdpa-kernels] line with pairing=ok. A case the
fused node serves names that kernel and no QK^T / softmax / attn*V kernel; every other case runs both coopmat
kernels and the environment's softmax. Banners: the environment's profile, softmax and fused pair in every pass.
Run-local difference: the historical checker derives the fused pair from the profile name (orin-fused1 only); the
matrix arms name it through ET_VK_SARC_ORIN_SDPA_FUSED, which takes precedence in SdpaOrinFused.cpp, so the pair
is read from that variable here. Prints findings and ACCEPT or REJECT; exit 0 only on ACCEPT."""
import glob
import os
import re
import sys

TIERS = [('all', 4, 12), ('extended', 8, 12), ('full', 4, 12), ('peaked', 5, 3), ('fused', 5, 3)]


def main():
    directory, envfile = sys.argv[1:3]
    env = dict(l.strip().split('=', 1) for l in open(envfile) if '=' in l)
    profile = env.get('ET_VK_SARC_DEV_PROFILE')
    soft = env.get('ET_VK_SARC_SOFTMAX_VARIANT') or ('orin_g64' if (profile or '').startswith('orin-fused') else None)
    fused = {}
    for token in filter(None, env.get('ET_VK_SARC_ORIN_SDPA_FUSED', '').split(',')):
        d, m, n = map(int, re.search(r'_d(\d+)_t(\d+)x(\d+)', token).groups())
        fused[d] = ('sarc_dev_orin_sdpa_' + token, m, n)
    want_soft = 'sarc_sdpa_attn_weights_softmax_buffer_half' + (f'_{soft}' if soft else '')
    findings, names = [], set()
    for tier, ncase, npass in TIERS:
        logs = sorted(glob.glob(os.path.join(directory, f'cand-{tier}-r*.log')))
        if len(logs) != npass:
            findings.append(f'{tier}: {len(logs)} pass logs, required {npass}')
        for f in logs:
            b = os.path.basename(f)
            text = open(f, errors='replace').read()
            lines = text.splitlines()
            if set(re.findall(r'\[sarc_dev\] profile active: (\S+)', text)) != ({profile} if profile else set()):
                findings.append(f'{b}: profile banner differs from {profile}')
            if set(re.findall(r'\[sarc_dev\] softmax variant: (\S+)', text)) != ({soft} if soft else set()):
                findings.append(f'{b}: softmax banner differs from {soft}')
            wf = ' '.join(v[0] for _, v in sorted(fused.items()))
            if set(re.findall(r'\[sarc_dev\] orin fused attention: (.+)', text)) != ({wf} if wf else set()):
                findings.append(f'{b}: fused attention banner differs from {wf or None}')
            cor = [l for l in lines if l.startswith('[sdpa-correctness]') and ' S=' in l]
            tot = [l for l in lines if l.startswith(f'[sdpa-correctness] tier={tier} ')]
            ker = {l.split()[1]: l for l in lines if l.startswith('[sdpa-kernels]')}
            if len(tot) != 1 or f'cases_run={ncase}' not in tot[0].split():
                findings.append(f'{b}: summary line {tot}')
            if len(cor) != ncase or len(ker) != ncase:
                findings.append(f'{b}: {len(cor)} cases, {len(ker)} [sdpa-kernels] lines, required {ncase}')
            for l in cor:
                name = l.split()[1]
                k = ker.get(name, '')
                g = re.search(r' S=(\d+) input_pos=(\d+) D=(\d+) ', l)
                S, pos, D = map(int, g.groups()) if g else (0, 0, 0)
                kernel, m, n = fused.get(D, (None, 1, 1))
                serves = kernel is not None and S >= m and S % m == 0 and S % n == 0 and pos % n == 0
                if not (l.rstrip().endswith(' PASSED') and re.search(r' mismatches=0/\d+', l)):
                    findings.append(f'{b}: {l.strip()[:160]}')
                if not k.rstrip().endswith('pairing=ok'):
                    findings.append(f'{b}: {name}: {k.strip()[:200]}')
                if serves:
                    if not re.search(r' fused=(?:"kernel_name": ")?' + re.escape(kernel) + r'_buffer_buffer_half\b', k):
                        findings.append(f'{b}: {name}: fused kernel is not {kernel}')
                    if not re.search(r' qk=\? softmax=\? av=\? ', k) or 'qk_coopmat=NO' not in l or 'av_coopmat=NO' not in l:
                        findings.append(f'{b}: {name}: an SDPA kernel ran beside the fused node')
                else:
                    if not ('qk_coopmat=yes' in l and 'av_coopmat=yes' in l):
                        findings.append(f'{b}: {l.strip()[:160]}')
                    if ' fused=- ' not in k and ' fused=' in k:
                        findings.append(f'{b}: {name}: a fused kernel ran where none fits')
                    if f'"{want_soft}"' not in k and f'softmax={want_soft} ' not in k:
                        findings.append(f'{b}: softmax kernel is not {want_soft}')
                names.update(re.findall(r'(?:qk|softmax|av|fused)=(?:"kernel_name": ")?([A-Za-z0-9_?-]+)', k))
    for finding in findings:
        print(finding)
    print('sdpa kernels dispatched:', ', '.join(sorted(names)) or 'none')
    print(f'sdpa: {"ACCEPT" if not findings else "REJECT"} ({len(findings)} findings)')
    sys.exit(1 if findings else 0)


if __name__ == '__main__':
    main()
