# Stage 5: larger-sample confirmation of A (judge Goodhart) and M (measurement), see PREREG_s5.md.
# Usage: S2_GEN=sonnet|opus python3 s5.py run  |  python3 s5.py analyze       (calls cached in out2/)
import sys,os,json,concurrent.futures as cf
import numpy as np
import s2,s4
REPS=range(116,140); OUT=s2.OUT
def J(): return json.load(open(os.path.join(OUT,'train_J.json')))['final']
def job(arm,rep):
    judge=s2.GEN=='opus' and arm in ('BASE','J')
    rec=s2.run(dict(s2.BASE),'nb',rep) if arm=='BASE' else s2.run(J(),'nb',rep) if arm=='J' else s4.run('A',100,rep)
    if rec and judge: s2.judge(rec)
    return rec
def runs(gen,arm):
    suf='' if gen=='sonnet' else '_'+gen; s2.GEN=gen; L=[]
    for rep in REPS:
        f=os.path.join(OUT,f'run_nb_{s2.hid(s2.BASE if arm=="BASE" else J())}_r{rep}{suf}.json' if arm!='A' else f'run_mA100_nb_r{rep}{suf}.json')
        if not os.path.exists(f): continue
        rec=json.load(open(f)); d=s2.score(rec)
        jf=os.path.join(OUT,f"judge_nb_{s2.hid(rec['h'])}_r{rep}{suf}.json") if arm!='A' else ''
        if jf and os.path.exists(jf): v=json.load(open(jf))['v']; d['judge']=float(v['novelty'])+float(v['quality'])
        L.append(d)
    return L
def analyze():
    rng=np.random.default_rng(0); out={}; V={}
    def perm(a,b):  # one-sided p for mean(a) > mean(b)
        a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
        return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
    def ci(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); bs=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(5000)]
        return [float(np.percentile(bs,5)),float(np.percentile(bs,95))]
    for gen in ('sonnet','opus'):
        R={a:runs(gen,a) for a in ('BASE','J','A')}; g=lambda a,k: [x[k] for x in R[a] if k in x]; m=lambda a,k: float(np.mean(g(a,k)))
        print(f'\n== {gen} (new runs only) ==\narm   n   SV    prec  prior  judge  tok(k)')
        for a in R: print(f"{a:4s} {len(R[a]):2d}  {m(a,'sv'):.2f}  {m(a,'prec'):.2f}  {m(a,'prior'):5.1f}  {m(a,'judge') if g(a,'judge') else float('nan'):5.1f}  {m(a,'tok')/1e3:.0f}")
        pA=perm(g('A','sv'),g('BASE','sv')); pJ=perm(g('BASE','sv'),g('J','sv'))  # M: A > BASE; A-SV: J < BASE
        ps=np.array([pA,pJ]); o=np.argsort(ps); adj=np.empty(2); adj[o[0]]=min(1,2*ps[o[0]]); adj[o[1]]=max(adj[o[0]],ps[o[1]])
        ciA=ci(g('A','sv'),g('BASE','sv')); ciJ=ci(g('J','sv'),g('BASE','sv')); pP=perm(g('BASE','prec'),g('J','prec')); dP=m('J','prec')-m('BASE','prec')
        print(f"  SV A-100 - BASE {m('A','sv')-m('BASE','sv'):+.2f}  90% CI [{ciA[0]:+.2f},{ciA[1]:+.2f}]  p {pA:.3f} holm {adj[0]:.3f}")
        print(f"  SV J - BASE     {m('J','sv')-m('BASE','sv'):+.2f}  90% CI [{ciJ[0]:+.2f},{ciJ[1]:+.2f}]  p(J<BASE) {pJ:.3f} holm {adj[1]:.3f}")
        print(f"  precision J - BASE {dP:+.2f}  p {pP:.3f}")
        v={}
        v['M']='SUPPORT' if adj[0]<.05 and m('A','prec')>=0.8*m('BASE','prec') else ('REFUTE' if ciA[1]<0.5 else 'INCONCLUSIVE')
        v['A-SV']='SUPPORT' if pJ<.05 else ('REFUTE' if m('J','sv')>=m('BASE','sv') else 'INCONCLUSIVE')
        v['prec_ok']=dP<=-0.2 and pP<.05; v['prec_refute']=dP>-0.1
        if g('J','judge') and g('BASE','judge'):
            pj=perm(g('J','judge'),g('BASE','judge')); print(f"  judge J - BASE {m('J','judge')-m('BASE','judge'):+.2f}  p {pj:.3f}")
            v['A-judge']='SUPPORT' if pj<.05 else ('REFUTE' if m('J','judge')<=m('BASE','judge') else 'INCONCLUSIVE')
        V[gen]=v; out[gen]=dict(arms={a:dict(n=len(R[a]),sv=m(a,'sv'),prec=m(a,'prec'),prior=m(a,'prior'),tok=m(a,'tok'),judge=m(a,'judge') if g(a,'judge') else None) for a in R},
                                ci_A=ciA,ci_J=ciJ,p=dict(A=pA,J=pJ,holm=adj.tolist(),prec=pP),verdicts=v)
    verdict={'A-prec':'SUPPORT' if all(V[x]['prec_ok'] for x in V) else ('REFUTE' if any(V[x]['prec_refute'] for x in V) else 'INCONCLUSIVE'),
             'A-judge (opus)':V['opus'].get('A-judge'),'A-SV':{x:V[x]['A-SV'] for x in V},'M':{x:V[x]['M'] for x in V}}
    print('\npre-registered verdicts:',verdict)
    json.dump(dict(gens=out,verdicts=verdict),open(os.path.join(s2.s1.HERE,'s5_results.json'),'w'),indent=1)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'run'
    if m=='run':
        jobs=[(a,r) for r in REPS for a in ('BASE','J','A')]
        with cf.ThreadPoolExecutor(6) as ex:
            for (a,r),rec in zip(jobs,ex.map(lambda j: job(*j),jobs)): print(s2.GEN,a,r,'ok' if rec else 'FAIL',flush=True)
    if m=='analyze': analyze()
