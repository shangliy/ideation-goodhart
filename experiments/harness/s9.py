# Stage 9: replicate A (judge Goodhart), D (BASE vs judge harness) and M (designed measurements) on a second public space,
# the HPOBench tabular MLP benchmark (see PREREG_s9.md, fc_space.py).
# Usage: python3 s9.py show | prior  |  S2_GEN=sonnet|opus python3 s9.py run  |  python3 s9.py analyze      (calls cached in out2/)
import sys,os,json,random,collections,concurrent.futures as cf
import numpy as np
import s2
from fc_space import P,NAME,BANDS,TASKS,band,load,space,EDITS
TAU=0.5; REPS=range(200,224); PANEL=('haiku','sonnet','opus'); K=100
from fc_space import SRC
if os.path.exists(SRC):  # full table: needed to run measurements; analysis only needs the derived claim space
    H=space(); L,KEYS,X,_,RAW=load()
    json.dump(dict(levels=L,cells={'|'.join(k):v for k,v in H.items()}),open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'fc_space.json'),'w'))
else:
    _d=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'fc_space.json'))); L=_d['levels']; H={tuple(k.split('|')):v for k,v in _d['cells'].items()}; KEYS=RAW=None
def truth(cell):
    d=np.array(H[cell]['diffs']); return int(np.sign(d[0])) if np.all(np.sign(d)==np.sign(d[0])) and np.min(np.abs(d))>=TAU else 0
def g(v): return f'{v:.3g}'
SETUP="\n".join([
 "Benchmark: a fully tabulated hyperparameter benchmark for a multilayer perceptron classifier (ReLU MLP trained with Adam for 243 epochs) on three tabular "
 "OpenML classification datasets: phoneme, segment, vehicle. Fitness is validation accuracy, averaged over 5 training seeds.",
 "Hyperparameters and their grids (every combination was trained):",
 *[f"- {p} = {NAME[p]}: "+", ".join(g(v) for v in L[p]) for p in P],
 "An edit moves one hyperparameter j one grid step up or down. A condition is 'hyperparameter k is in band c' where for depth the bands low/mid/high mean "
 "1/2/3 layers, and for the other hyperparameters low = the 3 smallest grid values, mid = the middle 4, high = the 3 largest (k may equal j).",
 "A hypothesis states that the edit's effect on validation accuracy is HIGHER (or LOWER) when the configuration satisfies the condition than when it does not, "
 "consistently across the three datasets and by a practically meaningful margin. It is about the interaction, not the edit's average effect."])
PN=' | '.join(P)
def fmt(n): return ('End with JSON only: {"hypotheses": [{"edit_param": "<'+PN+'>", "edit_dir": "up" or "down", "cond_param": "<'+PN+'>", '
                    f'"cond_band": "low" or "mid" or "high", "direction": "higher" or "lower", "why": "<one sentence>"}}, ...]}} with exactly {n} distinct hypotheses.')
def to_claim(x):
    try:
        j,dd,k,c=str(x['edit_param']).strip(),str(x['edit_dir']).strip().lower(),str(x['cond_param']).strip(),str(x['cond_band']).strip().lower()
        d=1 if str(x['direction']).lower().startswith('h') else -1
    except Exception: return None
    return (j,dd,k,c,d) if (j,dd,k,c) in H else None
CELLS=sorted(H); random.Random(0).shuffle(CELLS)
def ctext(c): j,dd,k,cc=c; return f"moving {j} one step {dd} has a HIGHER effect when {k} is in the {cc} band than when it is not"
def prior_job(model,i):
    B=CELLS[i:i+30]
    p=(SETUP+"\n\nFor each numbered claim, give your probability (0-100) that it holds (consistent across the three datasets, practically meaningful margin). "
       "If you think the effect is instead LOWER under the condition, give a low number.\n"+"\n".join(f"{n}. {ctext(c)}" for n,c in enumerate(B))+
       '\nEnd with JSON only: {"p": {"0": <0-100>, ...}} covering every claim.')
    return s2.step(f'fcprior_{model}_{i}',model,p,'p',lambda v: isinstance(v,dict) and all(str(n) in v for n in range(len(B))))
FCP=collections.defaultdict(dict)
def load_prior():
    for md in PANEL:
        for i in range(0,len(CELLS),30):
            f=os.path.join(s2.OUT,f'fcprior_{md}_{i}.json')
            if os.path.exists(f):
                for n,c in enumerate(CELLS[i:i+30]): FCP[c][md]=float(json.load(open(f))['v'][str(n)])
load_prior()
def fc_prior(cl): v=[p if cl[4]>0 else 100-p for m,p in FCP[cl[:4]].items() if m!=s2.GEN]; return float(np.mean(v)) if v else np.nan
s2.SP['fc']=dict(setup=SETUP,fmt=fmt,claim=to_claim,valid=lambda c: truth(c[:4])==c[4],prior=fc_prior)
# --- measurement: one real training seed, test split (truth uses the 5-seed val mean) ---
GRID={p:L[p] for p in P}
def measure(par,j,dd,rng):
    ch=list(par); ch[j]+=1 if dd=='up' else -1
    a,b='|'.join(map(str,par)),'|'.join(map(str,ch)); out=[]
    for t in TASKS:
        s=rng.randrange(5); out.append(round(100*(RAW[t]['cells'][b]['test'][s]-RAW[t]['cells'][a]['test'][s]),2))
    return out
def ptxt(par): return ', '.join(f'{p}={g(GRID[p][i])}' for p,i in zip(P,par))
def mtext(M): return "\n".join(f"{n+1}. config ({ptxt(par)}); edit: {P[j]} one step {dd}; change phoneme {d[0]:+.2f}, segment {d[1]:+.2f}, vehicle {d[2]:+.2f}"
                               for n,(par,j,dd,d) in enumerate(M))
def design_prompt(rep):
    return (SETUP+f"\n(Independent attempt #{rep+1}.)\n\nYou may run {K} experiments before proposing hypotheses. An experiment applies one edit to one configuration "
            "you choose, trains both once (a single training seed) and returns the change in test accuracy (percentage points) on the three datasets. Design the "
            "experiments that will best reveal conditional effects; you will see the results and then propose 10 hypotheses.\n"
            'End with JSON only: {"experiments": [{"config": {"alpha": <grid value>, "batch_size": ..., "depth": ..., "learning_rate_init": ..., "width": ...}, '
            f'"edit_param": "<{PN}>", "edit_dir": "up" or "down"}}, ...]}} with exactly {K} experiments.')
def propose_prompt(M,rep):
    return (SETUP+f"\n(Independent attempt #{rep+1}.)\n\nBefore proposing, you ran {len(M)} real experiments on this benchmark. Each experiment applied one edit to one "
            "configuration, trained both once (a single training seed) and measured the change in test accuracy (percentage points) on the three datasets; these are "
            "real measurements and can serve as evidence:\n"+mtext(M)+"\n\nPropose 10 conditional hypotheses about this benchmark that you believe are true.\n"+fmt(10))
def parse_probe(x):
    try:
        par=tuple(int(np.argmin([abs(np.log(v)-np.log(float(x['config'][p]))) for v in GRID[p]])) for p in P)
        j=P.index(str(x['edit_param']).strip()); dd=str(x['edit_dir']).strip().lower(); s=1 if dd=='up' else -1 if dd=='down' else None
        return (par,j,dd) if s and 0<=par[j]+s<len(GRID[P[j]]) else None
    except Exception: return None
def run_meas(rep):
    tag=f'fcA{K}_r{rep}{s2.SUF}'; f=os.path.join(s2.OUT,f'run_{tag}.json')
    if os.path.exists(f): return json.load(open(f))
    d=s2.step(f'{tag}_design',s2.GEN,design_prompt(rep),'experiments',lambda v: isinstance(v,list) and len(v)>=K//2)
    if not d['ok']: return None
    Pr=[parse_probe(x) for x in d['v'][:K]]; rng=random.Random(rep); M=[(par,j,dd,measure(par,j,dd,rng)) for par,j,dd in (x for x in Pr if x)]
    r=s2.step(f'{tag}_gen',s2.GEN,propose_prompt(M,rep),'hypotheses',lambda v: isinstance(v,list) and len(v)>=5)
    if not r['ok']: return None
    rec=dict(h=dict(measure='A',k=K),sp='fc',rep=rep,hyps=r['v'][:10],tok=d['tok']+r['tok'],bad=sum(x is None for x in Pr),probes=[[list(a),j,dd] for a,j,dd,_ in M])
    json.dump(rec,open(f,'w')); return rec
def J(): return json.load(open(os.path.join(s2.OUT,'train_J.json')))['final']
def job(arm,rep):
    if arm=='A': return run_meas(rep)
    rec=s2.run(dict(s2.BASE) if arm=='BASE' else J(),'fc',rep)
    if rec: s2.judge(rec)
    return rec
def runs(gen,arm):
    s2.GEN=gen; suf='' if gen=='sonnet' else '_'+gen; Lr=[]
    for rep in REPS:
        f=os.path.join(s2.OUT,f'run_fcA{K}_r{rep}{suf}.json' if arm=='A' else f'run_fc_{s2.hid(s2.BASE if arm=="BASE" else J())}_r{rep}{suf}.json')
        if not os.path.exists(f): continue
        rec=json.load(open(f)); d=s2.score(rec)
        if arm!='A':
            jf=os.path.join(s2.OUT,f"judge_fc_{s2.hid(rec['h'])}_r{rep}{suf}.json")
            if os.path.exists(jf): v=json.load(open(jf))['v']; d['judge']=float(v['novelty'])+float(v['quality'])
        Lr.append(d)
    return Lr
def analyze():
    rng=np.random.default_rng(0); out={}; V={}
    def perm(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
        return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
    def ci(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); bs=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(5000)]
        return [float(np.percentile(bs,5)),float(np.percentile(bs,95))]
    vc=[c for c in H if truth(c)]; print(f'space: {len(H)} cells, {2*len(H)} claims, {len(vc)} valid; random precision {len(vc)/(2*len(H)):.3f}')
    for gen in ('sonnet','opus'):
        s2.GEN=gen; sv_rate=np.mean([fc_prior((*c,truth(c)))<50 for c in vc]) if vc else np.nan
        R={a:runs(gen,a) for a in ('BASE','J','A')}; gv=lambda a,k: [x[k] for x in R[a] if k in x]; m=lambda a,k: float(np.mean(gv(a,k))) if gv(a,k) else float('nan')
        print(f'\n== {gen} == (valid claims with LOO prior < 50: {sv_rate:.2f})\narm   n   SV    prec  prior  judge  tok(k)')
        for a in R: print(f"{a:4s} {len(R[a]):2d}  {m(a,'sv'):.2f}  {m(a,'prec'):.2f}  {m(a,'prior'):5.1f}  {m(a,'judge'):5.1f}  {m(a,'tok')/1e3:.0f}")
        pA=perm(gv('A','sv'),gv('BASE','sv')); pJ=perm(gv('BASE','sv'),gv('J','sv')); ps=np.array([pA,pJ]); lo,hi=np.argsort(ps); adj=np.empty(2)
        adj[lo]=min(1,2*ps[lo]); adj[hi]=max(adj[lo],ps[hi]); ciA=ci(gv('A','sv'),gv('BASE','sv'))
        pP=perm(gv('BASE','prec'),gv('J','prec')); dP=m('J','prec')-m('BASE','prec'); pj=perm(gv('J','judge'),gv('BASE','judge'))
        print(f"  M  SV A-100 - BASE {m('A','sv')-m('BASE','sv'):+.2f} 90% CI [{ciA[0]:+.2f},{ciA[1]:+.2f}] p {pA:.3f} holm {adj[0]:.3f}")
        print(f"  A  SV J - BASE {m('J','sv')-m('BASE','sv'):+.2f} p(J<BASE) {pJ:.3f} holm {adj[1]:.3f}; precision {dP:+.2f} p {pP:.3f}; judge {m('J','judge')-m('BASE','judge'):+.2f} p {pj:.3f}")
        v=dict(M='SUPPORT' if adj[0]<.05 and m('A','prec')>=0.8*m('BASE','prec') else ('REFUTE' if ciA[1]<0.5 else 'INCONCLUSIVE'),
               A_prec='SUPPORT' if dP<=-0.2 and pP<.05 else ('REFUTE' if dP>-0.1 else 'INCONCLUSIVE'),
               A_judge='SUPPORT' if pj<.05 else ('REFUTE' if m('J','judge')<=m('BASE','judge') else 'INCONCLUSIVE'),
               A_SV='SUPPORT' if adj[1]<.05 else ('REFUTE' if m('J','sv')>=m('BASE','sv') else 'INCONCLUSIVE'))
        V[gen]=v; out[gen]=dict(arms={a:dict(n=len(R[a]),sv=m(a,'sv'),prec=m(a,'prec'),prior=m(a,'prior'),judge=m(a,'judge'),tok=m(a,'tok')) for a in R},
                                ci_A=ciA,p=dict(A=pA,J=pJ,holm=adj.tolist(),prec=pP,judge=pj),verdicts=v,sv_rate=float(sv_rate))
    print('\npre-registered verdicts:',V)
    json.dump(dict(gens=out,verdicts=V,space=dict(cells=len(H),valid=len(vc))),open(os.path.join(s2.s1.HERE,'s9_results.json'),'w'),indent=1)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'show'
    if m=='show':
        print(SETUP+'\n\n'+fmt(10)); rng=random.Random(0); M=[((3,4,1,5,4),3,'up',measure((3,4,1,5,4),3,'up',rng))]; print('\n'+mtext(M))
        print('\nprobe parse test:',parse_probe({'config':{'alpha':1e-4,'batch_size':32,'depth':2,'learning_rate_init':0.001,'width':128},'edit_param':'width','edit_dir':'up'}))
        print(len(H),'cells; valid',sum(truth(c)!=0 for c in H))
    if m=='prior':
        Jb=[(md,i) for md in PANEL for i in range(0,len(CELLS),30)]
        with cf.ThreadPoolExecutor(6) as ex:
            for (md,i),r in zip(Jb,ex.map(lambda j: prior_job(*j),Jb)): print('prior',md,i,'ok' if r['ok'] else 'FAIL',flush=True)
    if m=='run':
        Jb=[(a,r) for r in REPS for a in ('BASE','J','A')]
        with cf.ThreadPoolExecutor(6) as ex:
            for (a,r),rec in zip(Jb,ex.map(lambda j: job(*j),Jb)): print(s2.GEN,a,r,'ok' if rec else 'FAIL',flush=True)
    if m=='analyze': analyze()
