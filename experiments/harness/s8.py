# Stage 8: design-only harness (P-100): design experiments knowing no results will come, then propose (see PREREG_s8.md)
# Usage: S2_GEN=sonnet|opus python3 s8.py show | run  |  python3 s8.py analyze        (calls cached in out2/)
import sys,os,json,concurrent.futures as cf
import numpy as np
import s2,s4,s7
REPS=range(116,140); OUT=s2.OUT; K=100
def design_prompt(rep):
    t=s4.design_prompt(K,rep)
    return t.replace("You may run {k} experiments before proposing hypotheses. An experiment applies one edit to one parent cell you choose and returns the change in test "
                     "accuracy (percentage points) on the three datasets. Design the experiments that will best reveal conditional effects; you will see the results and "
                     "then propose 10 hypotheses.".format(k=K),
                     f"Before proposing hypotheses, write out {K} experiments. An experiment applies one edit to one parent cell you choose. You will NOT see the results "
                     "of these experiments; writing them out is meant to help you think about which conditional effects are worth hypothesizing. Design the experiments "
                     "that would best reveal conditional effects; afterwards you will propose 10 hypotheses.")
def run(rep):
    s4.load(); tag=f'mP{K}_nb_r{rep}{s2.SUF}'; f=os.path.join(OUT,f'run_{tag}.json')
    if os.path.exists(f): return json.load(open(f))
    d=s2.step(f'{tag}_design',s2.GEN,design_prompt(rep),'experiments',lambda v: isinstance(v,list) and len(v)>=K//2)
    if not d['ok']: return None
    P=[s4.parse_probe(x) for x in d['v'][:K]]; probes=[[list(p),list(e)] for p,e in (x for x in P if x)]
    r=s2.step(f'{tag}_gen',s2.GEN,s7.prompt('D0',rep,probes),'hypotheses',lambda v: isinstance(v,list) and len(v)>=5)
    if not r['ok']: return None
    rec=dict(h=dict(measure='P',k=K),sp='nb',rep=rep,hyps=r['v'][:10],tok=d['tok']+r['tok'],bad=sum(x is None for x in P),probes=probes)
    json.dump(rec,open(f,'w')); return rec
def analyze():
    rng=np.random.default_rng(0); out={}; ps=[]
    def perm(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
        return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
    def ci(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); bs=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(5000)]
        return [float(np.percentile(bs,5)),float(np.percentile(bs,95))]
    for gen in ('opus','sonnet'):
        s2.GEN=gen; suf='' if gen=='sonnet' else '_opus'
        names={'BASE':lambda r: f'run_nb_{s2.hid(s2.BASE)}_r{r}{suf}.json','A':lambda r: f'run_mA100_nb_r{r}{suf}.json','P':lambda r: f'run_mP{K}_nb_r{r}{suf}.json'}
        if gen=='opus': names['D0']=lambda r: f'run_mA100D0_nb_r{r}_opus.json'
        R={}
        for n,fn in names.items():
            L=[]
            for r in REPS:
                f=os.path.join(OUT,fn(r))
                if os.path.exists(f): rec=json.load(open(f)); d=s2.score(rec); d['bad']=rec.get('bad',0); L.append(d)
            R[n]=L
        g=lambda n,k: [x[k] for x in R[n]]; m=lambda n,k: float(np.mean(g(n,k)))
        print(f'\n== {gen} ==\narm   n   SV    prec  prior  tok(k)  bad')
        for n in R: print(f"{n:4s} {len(R[n]):2d}  {m(n,'sv'):.2f}  {m(n,'prec'):.2f}  {m(n,'prior'):5.1f}  {m(n,'tok')/1e3:5.0f}  {m(n,'bad'):.1f}")
        p=perm(g('P','sv'),g('BASE','sv')); c=ci(g('P','sv'),g('BASE','sv')); ps.append(p)
        out[gen]=dict(arms={n:dict(n=len(R[n]),sv=m(n,'sv'),prec=m(n,'prec'),prior=m(n,'prior'),tok=m(n,'tok'),bad=m(n,'bad')) for n in R},p=p,ci=c,
                      desc={o:dict(diff=m('P','sv')-m(o,'sv'),ci=ci(g('P','sv'),g(o,'sv'))) for o in R if o not in ('P','BASE')})
        print(f"  P - BASE: SV {m('P','sv')-m('BASE','sv'):+.2f} 90% CI [{c[0]:+.2f},{c[1]:+.2f}] p {p:.3f}")
        for o,dd in out[gen]['desc'].items(): print(f"  P - {o}: SV {dd['diff']:+.2f} [{dd['ci'][0]:+.2f},{dd['ci'][1]:+.2f}] (descriptive)")
    ps=np.array(ps); lo,hi=np.argsort(ps); adj=np.empty(2); adj[lo]=min(1,2*ps[lo]); adj[hi]=max(adj[lo],ps[hi]); V={}
    for (gen,o),q in zip(out.items(),adj):
        a=o['arms']; o['holm']=float(q)
        V[gen]=o['verdict']='SUPPORT' if q<.05 and a['P']['prec']>=0.8*a['BASE']['prec'] else ('REFUTE' if o['ci'][1]<0.5 else 'INCONCLUSIVE')
        print(f'{gen}: holm {q:.3f} -> {V[gen]}')
    print('\npre-registered verdicts:',V)
    json.dump(out,open(os.path.join(s2.s1.HERE,'s8_results.json'),'w'),indent=1)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'show'
    if m=='show': t=design_prompt(116); i=t.find('Before proposing hypotheses'); print(t[i-60:])
    if m=='run':
        with cf.ThreadPoolExecutor(6) as ex:
            for r,rec in zip(REPS,ex.map(run,REPS)): print(s2.GEN,'P',r,'ok' if rec else 'FAIL',flush=True)
    if m=='analyze': analyze()
