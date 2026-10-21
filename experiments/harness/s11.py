# Stage 11: causal test of the Sonnet x NB201 failure (framing and prior-consistency), see PREREG_s11.md.
# Reuses Sonnet's own measure-first probes and changes only the propose prompt.
# Usage: NB201_LOADER=... python3 s11.py show | run | analyze      (calls cached in out2/)
import os,sys,json,random,collections,concurrent.futures as cf
os.environ.setdefault('S2_GEN','sonnet')
import numpy as np
import s1,s2,s4,s9
from fc_space import band
NB_REPS=range(117,140); FC_REPS=s9.REPS; P_=list(s9.P)
NOTE_F=(" Each result comes from a single parent cell; the same edit can have a very different effect on other cells, so a single result is weak evidence "
        "about how the edit behaves in general.")
NOTE_G=("Propose only hypotheses that the experiments support AND that you also find plausible from your own knowledge of neural architectures; if the "
        "experiments and your knowledge disagree about a claim, do not propose it.\n")
def nb_probes(rep):
    r=json.load(open(os.path.join(s2.OUT,f'run_mA100_nb_r{rep}.json')))
    return [(tuple(p),(e[0],e[1]),s4.measure(tuple(p),(e[0],e[1]))) for p,e in r['probes']]
def fc_probes(rep):
    r=json.load(open(os.path.join(s2.OUT,f'run_fcA100_r{rep}.json'))); rng=random.Random(rep)
    return [(tuple(p),j,dd,s9.measure(tuple(p),j,dd,rng)) for p,j,dd in r['probes']]
def prompt(arm,rep):
    if arm in ('F','G'):
        M=nb_probes(rep); intro=s4.INTRO.format(k=len(M))
        if arm=='F': intro=intro.replace("can serve as evidence:\n","can serve as evidence."+NOTE_F+"\n")
        ask="\n\nPropose 10 conditional hypotheses about this benchmark that you believe are true.\n"+(NOTE_G if arm=='G' else '')
        return s1.SETUP+f"\n(Independent attempt #{rep+1}.)\n\n"+intro+s4.mtext(M)+ask+s1.fmt(10)
    M=fc_probes(rep); p=s9.propose_prompt(M,rep)
    old="configuration, trained both once (a single training seed) and measured"
    assert old in p; return p.replace(old,"configuration and measured")
def run(arm,rep):
    sp='fc' if arm=='H' else 'nb'; tag=f'{"fcA100" if sp=="fc" else "mA100"}{arm}_{"r" if sp=="fc" else "nb_r"}{rep}'
    f=os.path.join(s2.OUT,f'run_{tag}.json')
    if os.path.exists(f): return json.load(open(f))
    r=s2.step(f'{tag}_gen','sonnet',prompt(arm,rep),'hypotheses',lambda v: isinstance(v,list) and len(v)>=5)
    if not r['ok']: return None
    rec=dict(h=dict(measure='A',k=100,arm=arm),sp=sp,rep=rep,hyps=r['v'][:10],tok=r['tok'])
    json.dump(rec,open(f,'w')); return rec
# --- analysis: precision/SV plus conflict decomposition against Sonnet's own blind prior ---
def nb_ev(P):
    E=collections.defaultdict(list)
    for p,e,v in P:
        if p[e[0]]!=e[1]: E[e].append((p,np.array(v)))
    out={}
    for (j,b),L in E.items():
        for k in range(6):
            for c in s1.OPS:
                cell=(j,b,k,c); inn=[v for p,v in L if p[k]==c]; ot=[v for p,v in L if p[k]!=c]
                if cell in s1.H and inn and ot: out[cell]=float(np.mean(np.mean(inn,0)-np.mean(ot,0)))
    return out
def fc_ev(P):
    E=collections.defaultdict(list)
    for p,j,dd,v in P: E[(j,dd)].append((p,np.array(v)))
    out={}
    for (j,dd),L in E.items():
        for k in range(5):
            for c in ('low','mid','high'):
                cell=(P_[j],dd,P_[k],c); inn=[v for p,v in L if band(P_[k],p[k])==c]; ot=[v for p,v in L if band(P_[k],p[k])!=c]
                if cell in s9.H and inn and ot: out[cell]=float(np.mean(np.mean(inn,0)-np.mean(ot,0)))
    return out
def decompose(rec,ev,sp):
    claim=s2.SP[sp]['claim']; panel=s2.NBP if sp=='nb' else s9.FCP; n=conf=side=0
    for x in rec['hyps']:
        c=claim(x)
        if not c: continue
        cell,d=c[:4],c[4]; own=panel.get(cell,{}).get('sonnet'); n+=1
        if cell not in ev or own is None or own==50: continue
        if np.sign(ev[cell])!=(1 if own>50 else -1): conf+=1; side+= np.sign(ev[cell])==d
    return conf/max(n,1), (side/conf if conf else np.nan)
def score_pair(arm,rep):
    sp='fc' if arm=='H' else 'nb'; s2.GEN='sonnet'
    base=json.load(open(os.path.join(s2.OUT,f'run_fcA100_r{rep}.json' if sp=='fc' else f'run_mA100_nb_r{rep}.json')))
    new=run(arm,rep)
    if not new: return None
    ev=nb_ev(nb_probes(rep)) if sp=='nb' else fc_ev(fc_probes(rep))
    a,b=s2.score(base),s2.score(new); da,db=decompose(base,ev,sp),decompose(new,ev,sp)
    return dict(prec=(a['prec'],b['prec']),sv=(a['sv'],b['sv']),prior=(a['prior'],b['prior']),conf=(da[0],db[0]),side=(da[1],db[1]))
def analyze():
    rng=np.random.default_rng(0); out={}; ps={}
    def flip(d):  # one-sided paired sign-flip p for mean(d) > 0
        d=np.asarray(d,float); obs=d.mean(); return float(np.mean([(d*rng.choice([-1,1],len(d))).mean()>=obs for _ in range(10000)]))
    def ci(d): d=np.asarray(d,float); bs=[rng.choice(d,len(d)).mean() for _ in range(10000)]; return [float(np.percentile(bs,5)),float(np.percentile(bs,95))]
    for arm,reps in (('F',NB_REPS),('G',NB_REPS),('H',FC_REPS)):
        R=[x for x in (score_pair(arm,r) for r in reps) if x]
        g=lambda k,i: np.array([x[k][i] for x in R],float)
        dp=g('prec',1)-g('prec',0); dsv=g('sv',1)-g('sv',0)
        p=flip(dp) if arm!='H' else flip(-dp); ps[arm]=p
        out[arm]=dict(n=len(R),prec=[float(g('prec',0).mean()),float(g('prec',1).mean())],d_prec=float(dp.mean()),ci_prec=ci(dp),p=p,
                      sv=[float(g('sv',0).mean()),float(g('sv',1).mean())],ci_sv=ci(dsv),prior=[float(np.nanmean(g('prior',0))),float(np.nanmean(g('prior',1)))],
                      conflict_share=[float(g('conf',0).mean()),float(g('conf',1).mean())],side_with_evidence=[float(np.nanmean(g('side',0))),float(np.nanmean(g('side',1)))])
    order=sorted(ps,key=ps.get); m=len(order); prev=0
    for i,a in enumerate(order): prev=max(prev,min(1,(m-i)*ps[a])); out[a]['holm']=prev
    V={}
    for a in ('F','G'):
        o=out[a]; V[a]='SUPPORT' if o['d_prec']>=0.15 and o['holm']<.05 else ('REFUTE' if o['ci_prec'][1]<0.05 else 'INCONCLUSIVE')
    o=out['H']; V['H']='SUPPORT' if o['d_prec']<=-0.10 and o['holm']<.05 else ('REFUTE' if o['ci_prec'][0]>-0.05 else 'INCONCLUSIVE')
    for a,o in out.items():
        print(f"{a} (n={o['n']}): precision {o['prec'][0]:.2f} -> {o['prec'][1]:.2f} (d {o['d_prec']:+.2f}, 90% CI [{o['ci_prec'][0]:+.2f},{o['ci_prec'][1]:+.2f}], holm {o['holm']:.3f}); "
              f"SV {o['sv'][0]:.2f} -> {o['sv'][1]:.2f} (CI of d [{o['ci_sv'][0]:+.2f},{o['ci_sv'][1]:+.2f}]); prior {o['prior'][0]:.1f} -> {o['prior'][1]:.1f}; "
              f"conflict share {o['conflict_share'][0]:.2f} -> {o['conflict_share'][1]:.2f}; side with evidence {o['side_with_evidence'][0]:.2f} -> {o['side_with_evidence'][1]:.2f}")
    print('\npre-registered verdicts:',V)
    json.dump(dict(arms=out,verdicts=V),open(os.path.join(s1.HERE,'s11_results.json'),'w'),indent=1)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'show'
    s4.load()
    if m=='show':
        for a,r in (('F',117),('G',117),('H',200)):
            p=prompt(a,r); i=p.index('Before proposing'); print(f'--- {a}:',p[i:i+700].replace('\n',' | ')[:700],'...',p[-500:].replace('\n',' | '),'\n')
    if m=='run':
        J=[('F',r) for r in NB_REPS]+[('G',r) for r in NB_REPS]+[('H',r) for r in FC_REPS]
        with cf.ThreadPoolExecutor(4) as ex:
            for (a,r),rec in zip(J,ex.map(lambda j: run(*j),J)): print('s11',a,r,'ok' if rec else 'FAIL',flush=True)
    if m=='analyze': analyze()
