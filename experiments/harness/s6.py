# Stage 6: does the measurement gain survive noisy or low-fidelity measurements? (see PREREG_s6.md)
# Reuses the stage-5 Opus A-100 experiment designs; only the returned values change.
# Usage: S2_GEN=opus python3 s6.py show | run | analyze        (calls cached in out2/)
import sys,os,json,concurrent.futures as cf
import numpy as np
import s2,s4
REPS=range(116,140); OUT=s2.OUT; DS=s4.DS
SIG={'N1':(0.42,0.85,1.27),'N2':(0.85,1.70,2.55)}; ARMS=('N1','N2','L')
def clean_rec(rep): return json.load(open(os.path.join(OUT,f'run_mA100_nb_r{rep}_opus.json')))
def values(arm,rep,probes):
    s4.load(); P=[(tuple(p),tuple(e)) for p,e in probes]
    if arm=='L':
        from bench.nb201 import table
        E={ds:table(ds,'e12') for ds in DS}
        return [(p,e,[round(E[ds][s4.apply(p,e)]-E[ds][p],2) for ds in DS]) for p,e in P]
    rng=np.random.default_rng(rep)
    return [(p,e,[round(T+rng.normal(0,s),2) for T,s in zip(s4.measure(p,e),SIG[arm])]) for p,e in P]
def run(arm,rep):
    tag=f'mA100{arm}_nb_r{rep}_opus'; f=os.path.join(OUT,f'run_{tag}.json')
    if os.path.exists(f): return json.load(open(f))
    c=clean_rec(rep); M=values(arm,rep,c['probes'])
    r=s2.step(f'{tag}_gen','opus',s4.propose_prompt(M,rep),'hypotheses',lambda v: isinstance(v,list) and len(v)>=5)
    if not r['ok']: return None
    rec=dict(h=dict(measure='A',k=100,noise=arm),sp='nb',rep=rep,hyps=r['v'][:10],tok=r['tok'],bad=c['bad'],probes=c['probes'])
    json.dump(rec,open(f,'w')); return rec
def analyze():
    rng=np.random.default_rng(0); s2.GEN='opus'
    def perm(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
        return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
    def ci(a,b,paired=False):
        a,b=np.asarray(a,float),np.asarray(b,float)
        if paired: d=a-b; bs=[rng.choice(d,len(d)).mean() for _ in range(5000)]
        else: bs=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(5000)]
        return [float(np.percentile(bs,5)),float(np.percentile(bs,95))]
    def load(name):
        D={}
        for rep in REPS:
            f=os.path.join(OUT,f'run_nb_{s2.hid(s2.BASE)}_r{rep}_opus.json' if name=='BASE' else
                           f'run_mA100_nb_r{rep}_opus.json' if name=='A' else f'run_mA100{name}_nb_r{rep}_opus.json')
            if os.path.exists(f): D[rep]=s2.score(json.load(open(f)))
        return D
    R={n:load(n) for n in ('BASE','A')+ARMS}; v=lambda n,k: [R[n][r][k] for r in sorted(R[n])]; m=lambda n,k: float(np.mean(v(n,k)))
    print('arm   n   SV    prec  prior')
    for n in R: print(f"{n:4s} {len(R[n]):2d}  {m(n,'sv'):.2f}  {m(n,'prec'):.2f}  {m(n,'prior'):5.1f}")
    ps=np.array([perm(v(x,'sv'),v('BASE','sv')) for x in ARMS]); o=np.argsort(ps); adj=np.empty(3); run_=0
    for i,ix in enumerate(o): run_=max(run_,min(1,(3-i)*ps[ix])); adj[ix]=run_
    out={}; bprec=m('BASE','prec')
    for x,p,q in zip(ARMS,ps,adj):
        c=ci(v(x,'sv'),v('BASE','sv')); reps=sorted(set(R[x])&set(R['A']))
        pa=lambda k: ([R[x][r][k] for r in reps],[R['A'][r][k] for r in reps])
        csv,cpr=ci(*pa('sv'),paired=True),ci(*pa('prec'),paired=True)
        verdict='SUPPORT' if q<.05 and m(x,'prec')>=0.8*bprec else ('REFUTE' if c[1]<0.5 else 'INCONCLUSIVE')
        print(f"  {x} - BASE: SV {m(x,'sv')-m('BASE','sv'):+.2f} 90% CI [{c[0]:+.2f},{c[1]:+.2f}] p {p:.3f} holm {q:.3f} prec {m(x,'prec'):.2f} -> {verdict}")
        print(f"     paired vs clean A-100 (n={len(reps)}): SV {np.mean(np.subtract(*pa('sv'))):+.2f} [{csv[0]:+.2f},{csv[1]:+.2f}], "
              f"prec {np.mean(np.subtract(*pa('prec'))):+.2f} [{cpr[0]:+.2f},{cpr[1]:+.2f}]")
        out[x]=dict(n=len(R[x]),sv=m(x,'sv'),prec=m(x,'prec'),prior=m(x,'prior'),p=float(p),holm=float(q),ci_vs_base=c,
                    paired_vs_clean=dict(sv=csv,prec=cpr),verdict=verdict)
    out['BASE']=dict(sv=m('BASE','sv'),prec=bprec); out['A']=dict(sv=m('A','sv'),prec=m('A','prec'))
    print('\npre-registered verdicts:',{x:out[x]['verdict'] for x in ARMS})
    json.dump(out,open(os.path.join(s2.s1.HERE,'s6_results.json'),'w'),indent=1)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'show'
    assert m=='analyze' or s2.GEN=='opus','set S2_GEN=opus'
    if m=='show':
        c=clean_rec(116)
        for a in ARMS: M=values(a,116,c['probes'][:3]); print(a,'\n'+s4.mtext(M))
        print('\n--- clean ---\n'+s4.mtext(values('N1',116,c['probes'][:3]) and [(tuple(p),tuple(e),s4.measure(tuple(p),tuple(e))) for p,e in c['probes'][:3]]))
        print('\n--- prompt head ---\n'+s4.propose_prompt(values('N1',116,c['probes']),116)[:1500])
    if m=='run':
        J=[(a,r) for r in REPS for a in ARMS]
        with cf.ThreadPoolExecutor(6) as ex:
            for (a,r),rec in zip(J,ex.map(lambda j: run(*j),J)): print(a,r,'ok' if rec else 'FAIL',flush=True)
    if m=='analyze': analyze()
