# Stage 7: shuffled-results and design-only controls for the measurement gain (see PREREG_s7.md)
# Usage: S2_GEN=opus python3 s7.py show | run | analyze        (calls cached in out2/)
import sys,os,json,random,concurrent.futures as cf
import numpy as np
import s2,s4,s6
REPS=s6.REPS; OUT=s2.OUT; ARMS=('S','D0')
D0_INTRO=("Before proposing, you designed the following {k} experiments on this benchmark. Each applies one edit to one parent cell; "
          "their results are not available:\n")
def ptext(P): return "\n".join(f"{i+1}. parent ({', '.join(f'e{j}={o}' for j,o in enumerate(p))}); edit: edge {e[0]} -> {e[1]}" for i,(p,e) in enumerate(P))
def prompt(arm,rep,probes):
    P=[(tuple(p),tuple(e)) for p,e in probes]
    if arm=='D0':
        import s1
        return (s1.SETUP+f"\n(Independent attempt #{rep+1}.)\n\n"+D0_INTRO.format(k=len(P))+ptext(P)+
                "\n\nPropose 10 conditional hypotheses about this benchmark that you believe are true.\n"+s1.fmt(10))
    s4.load(); V=[s4.measure(p,e) for p,e in P]; random.Random(rep).shuffle(V)
    return s4.propose_prompt([(p,e,v) for (p,e),v in zip(P,V)],rep)
def run(arm,rep):
    tag=f'mA100{arm}_nb_r{rep}_opus'; f=os.path.join(OUT,f'run_{tag}.json')
    if os.path.exists(f): return json.load(open(f))
    c=s6.clean_rec(rep)
    r=s2.step(f'{tag}_gen','opus',prompt(arm,rep,c['probes']),'hypotheses',lambda v: isinstance(v,list) and len(v)>=5)
    if not r['ok']: return None
    rec=dict(h=dict(measure='A',k=100,control=arm),sp='nb',rep=rep,hyps=r['v'][:10],tok=r['tok'],bad=c['bad'],probes=c['probes'])
    json.dump(rec,open(f,'w')); return rec
def analyze():
    rng=np.random.default_rng(0); s2.GEN='opus'
    def load(n):
        D={}
        for rep in REPS:
            f=os.path.join(OUT,f'run_nb_{s2.hid(s2.BASE)}_r{rep}_opus.json' if n=='BASE' else f'run_mA100_nb_r{rep}_opus.json' if n=='A' else f'run_mA100{n}_nb_r{rep}_opus.json')
            if os.path.exists(f): D[rep]=s2.score(json.load(open(f)))
        return D
    R={n:load(n) for n in ('BASE','A','N1','N2','L')+ARMS}; v=lambda n,k: [R[n][r][k] for r in sorted(R[n])]; m=lambda n,k: float(np.mean(v(n,k)))
    print('arm   n   SV    prec  prior')
    for n in R:
        if R[n]: print(f"{n:4s} {len(R[n]):2d}  {m(n,'sv'):.2f}  {m(n,'prec'):.2f}  {m(n,'prior'):5.1f}")
    def flip(d): d=np.asarray(d,float); return float(np.mean([(d*rng.choice([-1,1],len(d))).mean()>=d.mean() for _ in range(5000)]))
    def perm(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
        return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
    D={x:[R['A'][r]['sv']-R[x][r]['sv'] for r in sorted(set(R['A'])&set(R[x]))] for x in ARMS}
    ps=np.array([flip(D[x]) for x in ARMS]); lo,hi=np.argsort(ps); adj=np.empty(2); adj[lo]=min(1,2*ps[lo]); adj[hi]=max(adj[lo],ps[hi])
    out={}
    for x,p,q in zip(ARMS,ps,adj):
        bs=[rng.choice(D[x],len(D[x])).mean() for _ in range(5000)]; ci=[float(np.percentile(bs,5)),float(np.percentile(bs,95))]
        pb=perm(v(x,'sv'),v('BASE','sv'))
        verdict='CONTENT' if q<.05 else ('DESIGN' if ci[1]<0.5 and pb<.05 else 'INCONCLUSIVE')
        print(f"  A - {x}: SV {np.mean(D[x]):+.2f} 90% CI [{ci[0]:+.2f},{ci[1]:+.2f}] p {p:.3f} holm {q:.3f};  {x} > BASE p {pb:.3f}  -> {verdict}")
        out[x]=dict(n=len(R[x]),sv=m(x,'sv'),prec=m(x,'prec'),prior=m(x,'prior'),diff_A=float(np.mean(D[x])),ci=ci,p=float(p),holm=float(q),p_vs_base=pb,verdict=verdict)
    print('\npre-registered verdicts:',{x:out[x]['verdict'] for x in ARMS})
    json.dump(out,open(os.path.join(s2.s1.HERE,'s7_results.json'),'w'),indent=1)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'show'
    assert m=='analyze' or s2.GEN=='opus','set S2_GEN=opus'
    if m=='show':
        c=s6.clean_rec(116)
        for a in ARMS: t=prompt(a,116,c['probes']); i=t.find('Before proposing'); print(f'--- {a} ---\n'+t[i:i+700]+'\n...\n'+t[-400:])
    if m=='run':
        J=[(a,r) for r in REPS for a in ARMS]
        with cf.ThreadPoolExecutor(6) as ex:
            for (a,r),rec in zip(J,ex.map(lambda j: run(*j),J)): print(a,r,'ok' if rec else 'FAIL',flush=True)
    if m=='analyze': analyze()
