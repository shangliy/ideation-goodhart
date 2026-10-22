# Stage 11 conflict share and siding recomputed with the three-way prior (stage 12) instead of the complement-coded rating (post hoc, no API calls).
# Usage: NB201_LOADER=... FC_TAB=... python3 s11_three_way.py
import os,sys
sys.path.insert(0,os.getcwd())
import numpy as np, s4, s11, s12
s4.load(); s12.load()
def decompose3(rec,ev,sp):
    claim=s11.s2.SP[sp]['claim']; n=conf=side=0
    for x in rec['hyps']:
        c=claim(x)
        if not c: continue
        cell,d=c[:4],c[4]; v=s12.P3[sp].get(cell,{}).get('sonnet'); n+=1
        if cell not in ev or v is None or v[0]==v[1]: continue
        pd=1 if v[0]>v[1] else -1
        if np.sign(ev[cell])!=pd: conf+=1; side+= np.sign(ev[cell])==d
    return conf/max(n,1), (side/conf if conf else np.nan)
s11.decompose=decompose3
for arm,reps in (('F',s11.NB_REPS),('G',s11.NB_REPS),('H',s11.FC_REPS)):
    R=[x for x in (s11.score_pair(arm,r) for r in reps) if x]
    g=lambda k,i: np.array([x[k][i] for x in R],float)
    print(f"{arm} n={len(R)}: conflict share {g('conf',0).mean():.2f} -> {g('conf',1).mean():.2f}; side with evidence {np.nanmean(g('side',0)):.2f} -> {np.nanmean(g('side',1)):.2f}; precision {g('prec',0).mean():.2f} -> {g('prec',1).mean():.2f}")
