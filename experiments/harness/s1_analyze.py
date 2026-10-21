# Stage-1 analysis, pre-registered in PREREG_s1.md.
import json,os,glob,collections
import numpy as np
from scipy.stats import spearmanr
from s1 import H,truth,to_claim,OUT,PANEL,GEN,CELLS,L9
P=collections.defaultdict(dict)
for f in glob.glob(OUT+'/prior_*.json'):
    _,md,i=os.path.basename(f)[:-5].split('_'); v=json.load(open(f))['v']
    for n,c in enumerate(CELLS[int(i):int(i)+30]): P[c][md]=float(v[str(n)])
def prior(cl,excl=GEN):
    c,d=cl[:4],cl[4]; v=[p if d>0 else 100-p for m,p in P.get(c,{}).items() if m!=excl]; return np.mean(v) if v else np.nan
rows=[]
for f in sorted(glob.glob(OUT+'/run_*.json')):
    r=json.load(open(f)); cl=[to_claim(x) for x in r['hyps']]; on=[c for c in dict.fromkeys(c for c in cl if c)]
    val=[c for c in on if truth(c[:4])==c[4]]; sv=[c for c in val if prior(c)<50]
    j=os.path.join(OUT,f"judge_c{''.join(map(str,r['cfg']))}_r{r['rep']}.json"); js=json.load(open(j))['v'] if os.path.exists(j) else {}
    rows.append(dict(cfg=tuple(r['cfg']),rep=r['rep'],on=len(on),off=10-len(on),valid=len(val),sv=len(sv),prec=len(val)/max(len(on),1),
                     prior=float(np.nanmean([prior(c) for c in on])) if on else np.nan,tok=r['tok'],nov=js.get('novelty'),qual=js.get('quality')))
print(len(rows),'runs; cells rated',len(P),'of',len(H))
allc=[(c+(d,)) for c in H for d in (1,-1)]; base_sv=np.mean([truth(c[:4])==c[4] and prior(c)<50 for c in allc])
print(f'random sampling: precision {np.mean([truth(c[:4])==c[4] for c in allc]):.3f}, SV rate {base_sv:.3f}')
by=collections.defaultdict(list)
for r in rows: by[r['cfg']].append(r)
print('cfg (R,G,C,S)  n  SV  valid  prec  prior  off  tok(k)  nov  qual')
for cfg in L9:
    L=by.get(cfg,[]);
    if not L: continue
    m=lambda k: np.nanmean([x[k] for x in L if x[k] is not None]) if any(x[k] is not None for x in L) else np.nan
    print(cfg,len(L),f"{m('sv'):.2f} {m('valid'):.2f} {m('prec'):.2f} {m('prior'):.1f} {m('off'):.1f} {m('tok')/1e3:.0f} {m('nov'):.1f} {m('qual'):.1f}")
sv=np.array([r['sv'] for r in rows],float); rng=np.random.default_rng(0); verdict={}
names='RGCS'; K1=[]
for fi,nm in enumerate(names):
    lv=np.array([r['cfg'][fi] for r in rows]); means=[sv[lv==l].mean() for l in range(3)]
    stat=np.var(means); null=[np.var([p[lv==l].mean() for l in range(3)]) for p in (rng.permutation(sv) for _ in range(5000))]
    p=float(np.mean(np.array(null)>=stat)); K1.append((nm,p,max(means)-min(means)))
    print(f'factor {nm}: SV by level {np.round(means,2)}  prec by level {np.round([np.mean([r["prec"] for r in rows if r["cfg"][fi]==l]) for l in range(3)],2)}  '
          f'prior by level {np.round([np.nanmean([r["prior"] for r in rows if r["cfg"][fi]==l]) for l in range(3)],1)}  perm p={p:.3f}')
verdict['K1']='SUPPORT' if any(p<0.05 and d>=1 for _,p,d in K1) else ('REFUTE' if all(p>0.2 for _,p,_ in K1) else 'INCONCLUSIVE')
J=[r for r in rows if r['nov'] is not None]
if len(J)>10:
    rho=spearmanr([r['nov'] for r in J],[r['sv'] for r in J]); rp=spearmanr([r['nov'] for r in J],[r['prior'] for r in J]); rq=spearmanr([r['qual'] for r in J],[r['valid'] for r in J])
    print(f'K2: rho(judge novelty, SV)={rho[0]:.2f} p={rho[1]:.3f}; rho(novelty, mean prior)={rp[0]:.2f} p={rp[1]:.3f}; rho(quality, valid)={rq[0]:.2f} p={rq[1]:.3f}  (n={len(J)})')
    verdict['K2']='PREMISE HOLDS' if rho[0]<=0.2 else ('JUDGE IS A GOOD PROXY' if rho[0]>=0.5 else 'INCONCLUSIVE')
rank=lambda key: sorted(by,key=lambda c: -np.mean([key(x) for x in by[c]]))
r1=rank(lambda x: x['sv']); r2=rank(lambda x: x['sv']/max(x['tok'],1)*1e3)
print('K3 top by SV:',r1[:3],' top by SV/1k tok:',r2[:3])
print('pre-registered verdicts:',verdict)
json.dump(dict(rows=[dict(r,cfg=list(r['cfg'])) for r in rows],verdict=verdict),open('s1_results.json','w'),indent=1)
