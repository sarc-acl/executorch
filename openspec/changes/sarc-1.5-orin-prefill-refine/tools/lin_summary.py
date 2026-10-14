import json,sys,glob,os,collections,statistics as st
d=sys.argv[1]; res=collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(d+'/*.json')):
    b=os.path.basename(f)[:-5]; q,rest=b.split('-',1); tok,rep=rest.rsplit('-r',1)
    try: cs=json.load(open(f))['cases']
    except Exception as e: continue
    for c in cs:
        if str(c.get('storage','')).lower()!='texture3d': continue
        res[tok][(c['model'][-2:],c['op'])].append((c['kernel_median_us'], c.get('kernel','')[:60]))
keys=[(m,o) for m in ('1b','3b','8b') for o in ('wq_wo','wk_wv','w1_w3','w2')]
base=res.get('base')
print('token', *[f'{m}:{o}' for m,o in keys], 'geomean_vs_base', 'kernel')
import math
for tok,v in res.items():
    row=[]; ratios=[]
    for k in keys:
        if k in v:
            t=st.median(x[0] for x in v[k]); row.append(f'{t/1000:.2f}')
            if base and k in base: ratios.append(st.median(x[0] for x in base[k])/t)
        else: row.append('-')
    g=math.exp(sum(map(math.log,ratios))/len(ratios)) if ratios else 0
    kern=set(x[1] for k in v for x in v[k])
    print(tok, *row, f'{g:.3f}', ('' if all('sarc' in k for k in kern) else 'FALLBACK:'+';'.join(sorted(kern))[:80]))
