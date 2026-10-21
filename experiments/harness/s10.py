# Stage 10: a non-Claude generator (GPT via the Codex CLI) on both public spaces, see PREREG_s10.md.
# Usage: python3 s10.py run nb|fc  |  python3 s10.py prior  |  python3 s10.py analyze      (calls cached in out/ and out2/)
import os,sys,json,concurrent.futures as cf
os.environ['S2_GEN']='gpt'
import numpy as np
import s1,s2,s4,s5,s9
GEN='gpt'; s2.GEN=GEN; s2.SUF='_'+GEN
def job(sp,arm,rep):
    if sp=='fc': return s9.job(arm,rep)
    rec=s2.run(dict(s2.BASE),'nb',rep) if arm=='BASE' else s2.run(s5.J(),'nb',rep) if arm=='J' else s4.run('A',100,rep)
    if rec and arm!='A': s2.judge(rec)
    return rec
def gpt_prior():
    # descriptive only: written under names the LOO panels never load (out/xprior_gpt_*, out2/fcprior_gpt_*), so earlier SV numbers do not move
    J=[('nb',i) for i in range(0,len(s1.CELLS),30)]+[('fc',i) for i in range(0,len(s9.CELLS),30)]
    def one(sp,i):
        if sp=='fc': return s9.prior_job(GEN,i)
        B=s1.CELLS[i:i+30]
        p=(s1.SETUP+"\n\nFor each numbered claim, give your probability (0-100) that it holds (consistent across the three datasets, practically meaningful margin). "
           "If you think the effect is instead LOWER under the condition, give a low number.\n"+"\n".join(f"{n}. {s1.ctext(c)}" for n,c in enumerate(B))+
           '\nEnd with JSON only: {"p": {"0": <0-100>, ...}} covering every claim.')
        return s1.step(f'xprior_{GEN}_{i}',GEN,p,'p',lambda v: isinstance(v,dict) and all(str(n) in v for n in range(len(B))))
    with cf.ThreadPoolExecutor(4) as ex:
        for (sp,i),r in zip(J,ex.map(lambda j: one(*j),J)): print('prior',sp,i,'ok' if r['ok'] else 'FAIL',flush=True)
def verdicts(R,rng):
    def perm(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
        return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
    def ci(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); bs=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(5000)]
        return [float(np.percentile(bs,5)),float(np.percentile(bs,95))]
    gv=lambda a,k: [x[k] for x in R[a] if k in x]; m=lambda a,k: float(np.mean(gv(a,k))) if gv(a,k) else float('nan')
    print('arm   n   SV    prec  prior  judge  tok(k)')
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
    return v,dict(arms={a:dict(n=len(R[a]),sv=m(a,'sv'),prec=m(a,'prec'),prior=m(a,'prior'),judge=m(a,'judge'),tok=m(a,'tok')) for a in R},
                  ci_A=ciA,p=dict(A=pA,J=pJ,holm=adj.tolist(),prec=pP,judge=pj),verdicts=v)
def prior_agreement():
    # descriptive: correlation of GPT's prior with the Claude panel mean, and with truth (directional claims)
    out={}
    for sp,cells,panel,load,truth in (('nb',s1.CELLS,s2.NBP,lambda i: os.path.join(s1.OUT,f'xprior_gpt_{i}.json'),s1.truth),
                                      ('fc',s9.CELLS,s9.FCP,lambda i: os.path.join(s2.OUT,f'fcprior_gpt_{i}.json'),s9.truth)):
        g={}
        for i in range(0,len(cells),30):
            if os.path.exists(load(i)):
                for n,c in enumerate(cells[i:i+30]): g[c]=float(json.load(open(load(i)))['v'][str(n)])
        C=[c for c in g if panel.get(c[:4] if sp=='nb' else c)]
        if not C: continue
        pm=lambda c: (lambda d: np.mean([d[k] for k in ('haiku','sonnet','opus') if k in d]))(panel[c[:4] if sp=='nb' else c])
        G=np.array([g[c] for c in C]); P=np.array([pm(c) for c in C]); T=np.array([truth(c[:4] if sp=='nb' else c) for c in C])
        r=lambda a,b: float(np.corrcoef(a,b)[0,1])
        out[sp]=dict(n=len(C),r_gpt_panel=r(G,P),r_gpt_truth=r(G,T!=0),r_panel_truth=r(P,T!=0))
        print(sp,out[sp])
    return out
def analyze():
    rng=np.random.default_rng(0); res={}
    print('== NB201, gpt ==')
    v,o=verdicts({a:s5.runs(GEN,a) for a in ('BASE','J','A')},rng); res['nb']=o
    print('== HPOBench, gpt ==')
    v2,o2=verdicts({a:s9.runs(GEN,a) for a in ('BASE','J','A')},rng); res['fc']=o2
    print('\npre-registered verdicts:',{'nb':v,'fc':v2})
    res['prior']=prior_agreement()
    json.dump(res,open(os.path.join(s1.HERE,'s10_results.json'),'w'),indent=1)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'analyze'
    if m=='run':
        sp=sys.argv[2]; reps=s5.REPS if sp=='nb' else s9.REPS
        J=[(a,r) for r in reps for a in ('BASE','J','A')]
        with cf.ThreadPoolExecutor(4) as ex:
            for (a,r),rec in zip(J,ex.map(lambda j: job(sp,*j),J)): print(GEN,sp,a,r,'ok' if rec else 'FAIL',flush=True)
    if m=='prior': gpt_prior()
    if m=='analyze': analyze()
