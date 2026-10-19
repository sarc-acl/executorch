import glob,re,statistics,sys
d,p=sys.argv[1],sys.argv[2]
row=re.compile(r"^et_vk\.linear_dq8ca\S*\s+(linear_dq8ca_q4gsw_M(\d+)_K(\d+)_N(\d+)_Texture3D)\s+\[\S+\]\s+([\d.]+) μs")
def load(a):
    t={}
    for f in glob.glob(f"{d}/{p}{a}_r*.log"):
        for l in open(f):
            m=row.match(l)
            if m: t.setdefault((int(m[2]),int(m[3]),int(m[4])),[]).append(float(m[5]))
    return t
h0,h1=load("h0"),load("h1")
rows=[]
for k in h0:
    a,b=h0[k],h1[k]
    rows.append((k[1]*k[2],k,statistics.median(a)/statistics.median(b),min(a)/min(b),(max(a)-min(a))/statistics.median(a)))
for kn,k,r,rmin,spr in sorted(rows):
    print(f"M{k[0]:5} K{k[1]:6} N{k[2]:6} KN={kn/1e6:6.1f}e6  h0/h1 med {r:.3f} min {rmin:.3f}  h0 spread {spr*100:4.1f}%")
