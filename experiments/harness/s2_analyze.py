# Stage-2 analysis, pre-registered in PREREG_s2.md. Reads out2/final.json and out2/train_*.json.
import json,os,itertools
import numpy as np
import s2
F=json.load(open(os.path.join(s2.OUT,'final.json'))); rng=np.random.default_rng(0)
def col(key,name,sp='nb'): return np.array([r[key] for r in F[f'{sp}|{name}']['runs']],float)
def perm(a,b):  # one-sided p for mean(a) > mean(b)
    a,b=a[~np.isnan(a)],b[~np.isnan(b)]; x=np.r_[a,b]; d=a.mean()-b.mean()
    null=[(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x)) for _ in range(5000)]; return float(np.mean(np.array(null)>=d))
def distinct(name,sp='nb'): return {tuple(map(lambda v: tuple(v) if isinstance(v,list) else v,c)) for r in F[f'{sp}|{name}']['runs'] for c in r['claims']}
NAMES=['BASE','final_J','final_V','final_SV']
for arm in s2.FIT:
    T=json.load(open(os.path.join(s2.OUT,f'train_{arm}.json')))
    print(f"train {arm}: final score {T['final_score']}; accepted:",[ (x['round'],x.get('score')) for x in T['log'] if x.get('status')=='accepted'])
    print('   final harness',json.dumps(T['final']))
res={}
for sp in ('nb',):
    print(f'\n== {sp} fresh runs ==\nharness     n  judge  nov  qual  valid  prec  SV   prior  distinct  tok(k)')
    for nm in NAMES:
        R=F[f'{sp}|{nm}']['runs']; m=lambda k: float(np.nanmean([r[k] if r[k] is not None else np.nan for r in R]))
        res[f'{sp}|{nm}']=dict(n=len(R),judge=m('judge'),valid=m('valid'),prec=m('prec'),sv=m('sv'),prior=m('prior'),distinct=len(distinct(nm,sp)),tok=m('tok'))
        print(f"{nm:10s} {len(R):2d}  {m('judge'):5.1f}  {m('nov'):4.1f}  {m('qual'):4.1f}  {m('valid'):5.2f}  {m('prec'):.2f}  {m('sv'):.2f}  {m('prior'):5.1f}  {len(distinct(nm,sp)):3d}      {m('tok')/1e3:.0f}")
N=lambda k,nm: res[f'nb|{nm}'][k]; v={}
pJ=perm(col('judge','final_J'),col('judge','BASE')); pJsv=perm(col('sv','final_J'),col('sv','BASE'))
v['P1']='SUPPORT' if pJ<.05 and N('sv','final_J')<=N('sv','BASE')+0.3 else ('REFUTE' if pJsv<.05 else 'INCONCLUSIVE')
v['P2']=('SUPPORT' if N('prior','final_V')>=N('prior','BASE')+3 and N('distinct','final_V')<=N('distinct','BASE') else
         'REFUTE' if N('prior','final_V')<=N('prior','BASE')-3 else 'INCONCLUSIVE')
p1=perm(col('sv','final_SV'),col('sv','BASE')); p2=perm(col('sv','final_SV'),col('sv','final_V'))
v['P3']=('SUPPORT' if p1<.10 and p2<.10 and N('prec','final_SV')>=0.8*N('prec','BASE') else
         'REFUTE' if N('sv','final_SV')<=N('sv','BASE') else 'INCONCLUSIVE')
v['main line']='SUPPORT' if v['P1']=='SUPPORT' and v['P3']=='SUPPORT' else 'NOT SUPPORTED'
print(f"\np-values: judge J>BASE {pJ:.3f}; SV J>BASE {pJsv:.3f}; SV SV>BASE {p1:.3f}; SV SV>V {p2:.3f}")
# (The CTR pooled hypergeometric check of the original study needs the private CTR space and is omitted here.)
print('pre-registered verdicts:',v)
json.dump(dict(summary=res,verdicts=v,p=dict(judge_J=pJ,sv_J=pJsv,sv_SV_base=p1,sv_SV_V=p2)),open('s2_results.json','w'),indent=1)
