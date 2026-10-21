# Post-hoc decomposition of prior override (REPORT 7.2, paper table 'conflict'): for each measure-first hypothesis, does the model's own probe
# evidence agree with its own blind prior?  Needs the raw tables to recompute probe values: NB201_LOADER (NB201 table loader) and FC_TAB (HPOBench nn table).
# Usage: NB201_LOADER=... FC_TAB=... python3 post_conflict.py
# Decompose prior override into opportunity (evidence conflicts with own prior) and propensity (siding with the evidence).
import json,os,sys,random,collections
sys.path.insert(0,os.getcwd())
import numpy as np, s1, s2, s4, s9
from fc_space import band
s4.load()
P_=list(s9.P)
def nb_cells(P):
    E=collections.defaultdict(list)
    for p,e,v in P:
        if p[e[0]]!=e[1]: E[e].append((p,v))
    for (j,b),L in E.items():
        for k in range(6):
            for c in s1.OPS:
                cell=(j,b,k,c)
                if cell not in s1.H: continue
                inn=[v for p,v in L if p[k]==c]; out=[v for p,v in L if p[k]!=c]
                if inn and out: yield cell, float(np.mean(np.mean(inn,0)-np.mean(out,0)))
def fc_cells(P):
    E=collections.defaultdict(list)
    for p,e,v in P: E[e].append((p,v))
    for (j,dd),L in E.items():
        for k in range(5):
            for c in ('low','mid','high'):
                cell=(P_[j],dd,P_[k],c)
                if cell not in s9.H: continue
                inn=[v for p,v in L if band(P_[k],p[k])==c]; out=[v for p,v in L if band(P_[k],p[k])!=c]
                if inn and out: yield cell, float(np.mean(np.mean(inn,0)-np.mean(out,0)))
def nb_run(rep,suf):
    f=f'out2/run_mA100_nb_r{rep}{suf}.json'
    if not os.path.exists(f): return None
    r=json.load(open(f)); P=[(tuple(p),(e[0],e[1]),np.array(s4.measure(tuple(p),(e[0],e[1])))) for p,e in r['probes']]
    H=[]
    for x in r['hyps']:
        try: c=(int(x['edit_edge']),str(x['edit_op']),int(x['cond_edge']),str(x['cond_op'])); d=1 if str(x['direction']).lower().startswith('h') else -1
        except Exception: continue
        H.append((c,d))
    return P,H
def fc_run(rep,suf):
    f=f'out2/run_fcA100_r{rep}{suf}.json'
    if not os.path.exists(f): return None
    r=json.load(open(f)); rng=random.Random(rep); P=[(tuple(p),(j,dd),np.array(s9.measure(tuple(p),j,dd,rng))) for p,j,dd in r['probes']]
    H=[]
    for x in r['hyps']:
        cl=s9.to_claim(x)
        if cl: H.append((cl[:4],cl[4]))
    return P,H
for sp,run,cells,panel,truth,reps in (('NB201',nb_run,nb_cells,s2.NBP,s1.truth,range(116,140)),('HPOBench',fc_run,fc_cells,s9.FCP,s9.truth,s9.REPS)):
    for g,suf in (('sonnet',''),('opus','_opus')):
        conf=[];ev_right=[];pr_right=[];prop=[];prop_n=0
        for rep in reps:
            x=run(rep,suf)
            if not x or not x[0]: continue
            P,H=x; Hset={(c,d) for c,d in H}; proposed_cells={c for c,d in H}
            for cell,ev in cells(P):
                own=panel[cell].get(g)
                if own is None or ev==0: continue
                pd=1 if own>50 else -1 if own<50 else 0
                if pd==0: continue
                es=int(np.sign(ev)); c=es!=pd; conf.append(c)
                if c:
                    t=truth(cell)
                    if t!=0: ev_right.append(es==t); pr_right.append(pd==t)
                    if cell in proposed_cells:  # when the model proposed this cell, which side did it take?
                        prop.append((cell,es) in Hset); prop_n+=1
        print(f'{sp} {g}: evidence contradicts own prior on {np.mean(conf):.2f} of contrastable cells; in conflicts on valid cells, evidence right {np.mean(ev_right):.2f} vs prior right {np.mean(pr_right):.2f} (n={len(ev_right)}); proposed conflict cells {prop_n}, sided with evidence {np.mean(prop) if prop else float("nan"):.2f}')
print('--- proposed cells by agreement between evidence and own prior')
for sp,run,cells,panel,truth,reps in (('NB201',nb_run,nb_cells,s2.NBP,s1.truth,range(116,140)),('HPOBench',fc_run,fc_cells,s9.FCP,s9.truth,s9.REPS)):
    for g,suf in (('sonnet',''),('opus','_opus')):
        R=collections.defaultdict(lambda:[0,0,0])  # n, valid cell, correct
        for rep in reps:
            x=run(rep,suf)
            if not x or not x[0]: continue
            P,H=x; ev=dict(cells(P))
            for c,d in H:
                if c not in ev or panel.get(c,{}).get(g) is None: grp='no evidence'
                else:
                    pd=1 if panel[c][g]>50 else -1
                    grp='evidence agrees with prior' if np.sign(ev[c])==pd else 'evidence conflicts with prior'
                t=truth(c); R[grp][0]+=1; R[grp][1]+=t!=0; R[grp][2]+=t==d
        print(sp,g,{k:f'n {v[0]}, valid-cell share {v[1]/v[0]:.2f}, precision {v[2]/v[0]:.2f}' for k,v in R.items()})
