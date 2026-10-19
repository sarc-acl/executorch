import glob,re,statistics,sys,math
row=re.compile(r"^et_vk\.linear_dq8ca\S*\s+linear_dq8ca_q4gsw_M(\d+)_K(\d+)_N(\d+)_Texture3D\s+\[\S+\]\s+([\d.]+) μs")
def load(d,p,a):
    t={}
    for f in glob.glob(f"{d}/{p}{a}_r*.log"):
        for l in open(f):
            m=row.match(l)
            if m: t.setdefault(tuple(map(int,m.groups()[:3])),[]).append(float(m[4]))
    return {k:statistics.median(v) for k,v in t.items()}
for d,p in [(sys.argv[1],"perf_"),(sys.argv[2],"b580_")]:
    h0,h1=load(d,p,"h0"),load(d,p,"h1")
    print("==",d.split('/')[-1] or d, p)
    for T in [0,10e6,17e6,20e6,24e6,28e6,1e18]:
        lr=[];worst=1
        for k in h0:
            best=min(h0[k],h1[k]); ch=h1[k] if k[1]*k[2]>=T else h0[k]
            lr.append(math.log(ch/best)); worst=max(worst,ch/best)
        print(f"T={T/1e6:8.0f}e6  geomean loss vs best {100*(math.exp(sum(lr)/len(lr))-1):.2f}%  worst {100*(worst-1):.1f}%")
