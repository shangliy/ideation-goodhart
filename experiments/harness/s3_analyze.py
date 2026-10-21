# Stage-3 analysis (claim D: the precision-surprise frontier), pre-registered in PREREG_s3.md.
import json,os
import numpy as np
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); O=lambda f: json.load(open(os.path.join(HERE,'out2',f)))
rng=np.random.default_rng(0)
def perm(a,b):  # one-sided p for mean(a) > mean(b)
    a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
    return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
def holm(ps):
    o=np.argsort(ps); adj=np.empty(len(ps)); run=0
    for i,k in enumerate(o): run=max(run,min(1,(len(ps)-i)*ps[k])); adj[k]=run
    return adj
def stats(runs): return dict(n=len(runs),sv=float(np.mean([r['sv'] for r in runs])),prec=float(np.mean([r['prec'] for r in runs])),
                             prior=float(np.nanmean([r['prior'] for r in runs])),svs=[r['sv'] for r in runs])
H={'sonnet':{},'opus':{}}; CMP={'sonnet':[],'opus':[]}
S1=json.load(open(os.path.join(HERE,'s1_results.json')))['rows']
for cfg in sorted({tuple(r['cfg']) for r in S1}): H['sonnet'][f's1{cfg}']=stats([r for r in S1 if tuple(r['cfg'])==cfg])
F=O('final.json')
for nm in ('BASE','final_J','final_V'): H['sonnet'][f's2 {nm}']=stats(F[f'nb|{nm}']['runs'])
if os.path.exists(os.path.join(HERE,'out2','fresh_SV2.json')):
    f2=O('fresh_SV2.json'); same=f2['harness']==F['nb|BASE']['harness']
    if not same: H['sonnet']['s2 final_SV2']=stats(f2['runs'])
    print('SV2 final harness is BASE' if same else 'SV2 final harness: '+json.dumps(f2['harness'])[:300])
for cfg in H['sonnet']:
    if cfg.startswith('s1') and cfg!='s1(0, 0, 0, 0)': CMP['sonnet'].append((cfg,'s1(0, 0, 0, 0)'))
    if cfg.startswith('s2') and cfg!='s2 BASE': CMP['sonnet'].append((cfg,'s2 BASE'))
if os.path.exists(os.path.join(HERE,'out2','frontier_opus.json')):
    for nm,v in O('frontier_opus.json').items(): H['opus'][nm]=stats(v['runs'])
    CMP['opus']=[(nm,'BASE') for nm in H['opus'] if nm!='BASE']
verdict={}; d1=[]; d2=[]
for g in ('sonnet','opus'):
    if not CMP[g]: continue
    print(f'\n== generator {g} ==\nharness                       n   SV    prec  prior')
    for nm,s in H[g].items(): print(f"{nm:28s} {s['n']:2d}  {s['sv']:.2f}  {s['prec']:.2f}  {s['prior']:.1f}")
    ps=[perm(H[g][a]['svs'],H[g][b]['svs']) for a,b in CMP[g]]; adj=holm(np.array(ps))
    for (a,b),p,q in zip(CMP[g],ps,adj): print(f'  SV {a} > {b}: diff {H[g][a]["sv"]-H[g][b]["sv"]:+.2f}  p {p:.3f}  holm {q:.3f}')
    sig=any(q<.05 for q in adj); close=all(H[g][a]['sv']<=H[g][b]['sv']+0.5 for a,b in CMP[g])
    d1.append('REFUTE' if sig else ('SUPPORT' if close else 'INCONCLUSIVE'))
    rho=spearmanr([s['prior'] for s in H[g].values()],[s['prec'] for s in H[g].values()])[0]; print(f'  D2 Spearman(mean prior, precision) across {len(H[g])} harnesses: {rho:.2f}')
    d2.append('SUPPORT' if rho>=.5 else ('REFUTE' if rho<=0 else 'INCONCLUSIVE'))
verdict['D1']='REFUTE' if 'REFUTE' in d1 else ('SUPPORT' if d1 and all(x=='SUPPORT' for x in d1) and len(d1)==2 else 'INCONCLUSIVE')
verdict['D2']='REFUTE' if 'REFUTE' in d2 else ('SUPPORT' if len(d2)==2 and all(x=='SUPPORT' for x in d2) else 'INCONCLUSIVE')
verdict['D']='SUPPORT' if verdict['D1']==verdict['D2']=='SUPPORT' else ('REFUTE' if verdict['D1']=='REFUTE' else 'NOT SUPPORTED')
print('\nper-generator D1',d1,'D2',d2,'\npre-registered verdicts:',verdict)
json.dump(dict(harnesses={g:{k:{a:b for a,b in v.items() if a!='svs'} for k,v in H[g].items()} for g in H},verdicts=verdict),open(os.path.join(HERE,'s3_results.json'),'w'),indent=1)
