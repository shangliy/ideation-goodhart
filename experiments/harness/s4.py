# Stage 4: does a measurement knob push the precision-surprise frontier? (see PREREG_s4.md)
# Usage: python3 s4.py show | run | analyze | confirm | confirm-analyze        (S2_GEN=opus switches the generator; calls cached in out2/)
import sys,os,json,random,concurrent.futures as cf
import numpy as np
import s2,s1
from s2 import step,show,score,OUT,GEN,SUF
from scipy.stats import spearmanr
DS=('C10','C100','IN16'); T={}; CELLS=[]
def load():  # measurements use the test split of the NB201 table (truth uses val); needed only to run or show, not to analyze
    global edits,apply
    if T: return
    sys.path.insert(0,os.environ['NB201_LOADER']); from bench.nb201 import table,edits,apply
    T.update({ds:table(ds,'test') for ds in DS}); CELLS.extend(sorted(T['C10']))
ARMS=[('R',20),('R',100),('R',400),('A',20),('A',100)]; REPS=range(100,108)
def measure(p,e): c=apply(p,e); return [round(T[ds][c]-T[ds][p],2) for ds in DS]
def mtext(M): return "\n".join(f"{i+1}. parent ({', '.join(f'e{j}={o}' for j,o in enumerate(p))}); edit: edge {e[0]} -> {e[1]}; "
                               f"change C10 {d[0]:+.2f}, C100 {d[1]:+.2f}, IN16 {d[2]:+.2f}" for i,(p,e,d) in enumerate(M))
INTRO=("Before proposing, you ran {k} real experiments on this benchmark. Each experiment applied one edit to one parent cell and measured the change in "
       "test accuracy (percentage points) on the three datasets; these are real measurements and can serve as evidence:\n")
def propose_prompt(M,rep):
    return (s1.SETUP+f"\n(Independent attempt #{rep+1}.)\n\n"+INTRO.format(k=len(M))+mtext(M)+
            "\n\nPropose 10 conditional hypotheses about this benchmark that you believe are true.\n"+s1.fmt(10))
def design_prompt(k,rep):
    return (s1.SETUP+f"\n(Independent attempt #{rep+1}.)\n\nYou may run {k} experiments before proposing hypotheses. An experiment applies one edit to one "
            "parent cell you choose and returns the change in test accuracy (percentage points) on the three datasets. Design the experiments that will best "
            "reveal conditional effects; you will see the results and then propose 10 hypotheses.\n"
            'End with JSON only: {"experiments": [{"parent": ["<op edge 0>", ..., "<op edge 5>"], "edit_edge": <0-5>, "edit_op": "<op>"}, ...]} '
            f"with exactly {k} experiments.")
def parse_probe(x):
    try:
        p=tuple(str(o).strip() for o in x['parent']); e=(int(x['edit_edge']),str(x['edit_op']).strip())
        return (p,e) if len(p)==6 and p in T['C10'] and e in edits(p) else None
    except Exception: return None
def run(mode,k,rep):
    load(); tag=f'm{mode}{k}_nb_r{rep}{SUF}'; f=os.path.join(OUT,f'run_{tag}.json')
    if os.path.exists(f): return json.load(open(f))
    tok=0; bad=0
    if mode=='R':
        rng=random.Random(rep*1000+k); M=[]
        for _ in range(k):
            p=rng.choice(CELLS); e=rng.choice(edits(p)); M.append((p,e,measure(p,e)))
    else:
        d=step(f'{tag}_design',GEN,design_prompt(k,rep),'experiments',lambda v: isinstance(v,list) and len(v)>=k//2)
        if not d['ok']: return None
        tok+=d['tok']; P=[parse_probe(x) for x in d['v'][:k]]; bad=sum(x is None for x in P)
        M=[(p,e,measure(p,e)) for p,e in (x for x in P if x)]
    r=step(f'{tag}_gen',GEN,propose_prompt(M,rep),'hypotheses',lambda v: isinstance(v,list) and len(v)>=5)
    if not r['ok']: return None
    rec=dict(h=dict(measure=mode,k=k),sp='nb',rep=rep,hyps=r['v'][:10],tok=tok+r['tok'],bad=bad,probes=[[list(p),list(e)] for p,e,_ in M])
    json.dump(rec,open(f,'w')); return rec
def covered(c,probes):  # was the proposed claim's cell touched directly: same edit on a parent satisfying the condition
    j,b,k,cc=c[:4]; return any(e[0]==j and e[1]==b and p[k]==cc for p,e in probes)
def analyze():
    rng=np.random.default_rng(0); out={}; verdict={}; m1=[]; m2=[]
    def perm(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
        return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
    for gen in ('sonnet','opus'):
        suf='' if gen=='sonnet' else '_'+gen
        base=json.load(open(os.path.join(OUT,'final.json')))['nb|BASE']['runs'] if gen=='sonnet' else json.load(open(os.path.join(OUT,'frontier_opus.json')))['BASE']['runs']
        S={'k=0':[dict(sv=r['sv'],prec=r['prec'],prior=r['prior'],tok=r['tok'],cov=np.nan,bad=0) for r in base]}
        s2.GEN=gen  # leave-one-out prior excludes this generator
        for mode,k in ARMS:
            L=[]
            for rep in REPS:
                f=os.path.join(OUT,f'run_m{mode}{k}_nb_r{rep}{suf}.json')
                if not os.path.exists(f): continue
                rec=json.load(open(f)); d=score(rec); cl=[tuple(c) for c in d['claims']]
                d['cov']=float(np.mean([covered(c,rec['probes']) for c in cl])) if cl else np.nan; d['bad']=rec['bad']; L.append(d)
            S[f'{mode}-{k}']=L
        print(f'\n== generator {gen} ==\narm     n   SV    prec  prior  covered  bad  tok(k)')
        for nm,L in S.items():
            if L: print(f"{nm:6s} {len(L):2d}  {np.mean([x['sv'] for x in L]):.2f}  {np.mean([x['prec'] for x in L]):.2f}  {np.nanmean([x['prior'] for x in L]):5.1f}  "
                        f"{np.nanmean([x['cov'] for x in L]) if not all(np.isnan(x['cov']) for x in L) else float('nan'):.2f}     {np.mean([x['bad'] for x in L]):4.1f}  {np.mean([x['tok'] for x in L])/1e3:.0f}")
        b=S['k=0']; names=[n for n in S if n!='k=0' and S[n]]
        ps=[perm([x['sv'] for x in S[n]],[x['sv'] for x in b]) for n in names]
        o=np.argsort(ps); adj=np.empty(len(ps)); run_=0
        for i,ix in enumerate(o): run_=max(run_,min(1,(len(ps)-i)*ps[ix])); adj[ix]=run_
        bsv,bprec=np.mean([x['sv'] for x in b]),np.mean([x['prec'] for x in b])
        for n,p,q in zip(names,ps,adj): print(f'  SV {n} > k=0: diff {np.mean([x["sv"] for x in S[n]])-bsv:+.2f}  p {p:.3f}  holm {q:.3f}')
        hit=[n for n,q in zip(names,adj) if q<.05 and np.mean([x['prec'] for x in S[n]])>=0.8*bprec]
        m1.append('SUPPORT' if hit else ('REFUTE' if all(np.mean([x['sv'] for x in S[n]])<=bsv+0.3 for n in names) else 'INCONCLUSIVE'))
        ks=[0]*len(b)+[k for _,k in ARMS if _=='R' for _x in S[f'R-{k}']]; sv=[x['sv'] for x in b]+[x['sv'] for _,k in ARMS if _=='R' for x in S[f'R-{k}']]
        rho,p=spearmanr(ks,sv); print(f'  M2 Spearman(k, SV) over random arms: rho {rho:.2f} p {p:.3f}'); m2.append((rho,p))
        for k in (20,100):
            a,r_=S.get(f'A-{k}'),S.get(f'R-{k}')
            if a and r_: print(f"  M3 k={k}: active-random SV {np.mean([x['sv'] for x in a])-np.mean([x['sv'] for x in r_]):+.2f} (p {perm([x['sv'] for x in a],[x['sv'] for x in r_]):.3f}), "
                                f"precision {np.mean([x['prec'] for x in a])-np.mean([x['prec'] for x in r_]):+.2f}")
        out[gen]={n:dict(n=len(L),sv=float(np.mean([x['sv'] for x in L])),prec=float(np.mean([x['prec'] for x in L])),prior=float(np.nanmean([x['prior'] for x in L])),
                         tok=float(np.mean([x['tok'] for x in L]))) for n,L in S.items() if L}
    verdict['M1']='SUPPORT' if 'SUPPORT' in m1 else ('REFUTE' if all(x=='REFUTE' for x in m1) else 'INCONCLUSIVE')
    verdict['M2']='SUPPORT' if all(r>0 and p<.05 for r,p in m2) else ('REFUTE' if all(r<=0 for r,_ in m2) else 'INCONCLUSIVE')
    print('\nper-generator M1',m1,'\npre-registered verdicts:',verdict)
    json.dump(dict(arms=out,verdicts=verdict),open(os.path.join(s2.s1.HERE,'s4_results.json'),'w'),indent=1)
CREPS=range(108,116)
def confirm():  # Opus confirmation (PREREG_s4.md addendum): 8 new reps for k=0, R-100, A-100
    J=[('k0',0,r) for r in CREPS]+[(mo,100,r) for r in CREPS for mo in ('R','A')]
    f=lambda j: s2.run(dict(s2.BASE),'nb',j[2]) if j[0]=='k0' else run(*j)
    with cf.ThreadPoolExecutor(6) as ex:
        for j,rec in zip(J,ex.map(f,J)): print(GEN,j,'ok' if rec else 'FAIL',flush=True)
def confirm_analyze():
    rng=np.random.default_rng(0); s2.GEN='opus'
    def perm(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
        return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
    def runs(arm,reps):
        L=[]
        for r in reps:
            f=os.path.join(OUT,f'run_nb_{s2.hid(s2.BASE)}_r{r}_opus.json' if arm=='k0' else f'run_m{arm}100_nb_r{r}_opus.json')
            if os.path.exists(f): L.append(score(json.load(open(f))))
        return L
    out={}
    for label,reps in (('pooled 16',list(REPS)+list(CREPS)),('new 8 only',CREPS)):
        R={a:runs(a,reps) for a in ('k0','R','A')}; m=lambda a,k: float(np.mean([x[k] for x in R[a]]))
        ps=[perm([x['sv'] for x in R[a]],[x['sv'] for x in R['k0']]) for a in ('R','A')]; adj=[min(1,2*min(ps)),min(1,max(max(ps),2*min(ps)))]
        adj=[adj[0],adj[1]] if ps[0]<=ps[1] else [adj[1],adj[0]]
        print(f'== {label} ==')
        for a in ('k0','R','A'): print(f"  {a:3s} n={len(R[a])} SV {m(a,'sv'):.2f} prec {m(a,'prec'):.2f} prior {m(a,'prior'):.1f}")
        for a,p,q in zip(('R','A'),ps,adj): print(f"  SV {a}-100 > k0: diff {m(a,'sv')-m('k0','sv'):+.2f}  p {p:.3f}  holm {q:.3f}")
        hit=[a for a,q in zip(('R','A'),adj) if q<.05 and m(a,'prec')>=0.8*m('k0','prec')]
        v='SUPPORT' if hit else ('REFUTE' if all(m(a,'sv')<=m('k0','sv')+0.3 for a in ('R','A')) else 'INCONCLUSIVE')
        print('  verdict:',v); out[label]=dict(p=ps,holm=adj,verdict=v,**{a:dict(n=len(R[a]),sv=m(a,'sv'),prec=m(a,'prec'),prior=m(a,'prior')) for a in R})
    json.dump(out,open(os.path.join(s2.s1.HERE,'s4_confirm.json'),'w'),indent=1)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'show'
    if m=='show':
        load(); rng=random.Random(0); M=[]
        for _ in range(3):
            p=rng.choice(CELLS); e=rng.choice(edits(p)); M.append((p,e,measure(p,e)))
        print(propose_prompt(M,100)); print('\n----- design prompt tail -----\n',design_prompt(20,100)[-700:])
    if m=='run':
        J=[(mo,k,r) for r in REPS for mo,k in ARMS]
        with cf.ThreadPoolExecutor(6) as ex:
            for (mo,k,r),rec in zip(J,ex.map(lambda j: run(*j),J)): print(GEN,mo,k,r,'ok' if rec else 'FAIL',flush=True)
    if m=='analyze': analyze()
    if m=='confirm': confirm()
    if m=='confirm-analyze': confirm_analyze()
