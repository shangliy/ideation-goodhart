# Post-hoc analysis of the Sonnet x NAS-Bench-201 failure (REPORT 7.2, paper table 'override'): share of proposals that contradict
# the generator's own blind prior rating, and their precision; plus the repeated self-condition patterns. Reads cached runs only.
# Usage: python3 post_override.py
import json,os,sys,collections
sys.path.insert(0,os.getcwd())
import numpy as np, s1, s2, s5, s9
def nb_h(f):
    for x in json.load(open(f))['hyps']:
        try: c=(int(x['edit_edge']),str(x['edit_op']),int(x['cond_edge']),str(x['cond_op'])); d=1 if str(x['direction']).lower().startswith('h') else -1
        except Exception: continue
        if c in s1.H: yield c,d,s1.truth(c)==d,s2.NBP[c]
def fc_h(f):
    for x in json.load(open(f))['hyps']:
        cl=s9.to_claim(x)
        if cl: yield cl[:4],cl[4],s9.truth(cl[:4])==cl[4],s9.FCP[cl[:4]]
for sp,fn,base,af,reps in (('NB201',nb_h,lambda r,suf:f'out2/run_nb_{s2.hid(s2.BASE)}_r{r}{suf}.json',lambda r,suf:f'out2/run_mA100_nb_r{r}{suf}.json',s5.REPS),
                           ('HPOBench',fc_h,lambda r,suf:f'out2/run_fc_{s2.hid(s2.BASE)}_r{r}{suf}.json',lambda r,suf:f'out2/run_fcA100_r{r}{suf}.json',s9.REPS)):
    for g,suf in (('sonnet',''),('opus','_opus')):
        for arm,ff in (('direct',base),('measure',af)):
            R=[]
            for r in reps:
                f=ff(r,suf)
                if not os.path.exists(f): continue
                for c,d,ok,pan in fn(f):
                    own=pan.get(g); 
                    if own is None: continue
                    own=own if d>0 else 100-own
                    R.append((ok,own))
            R=np.array(R,float); lo=R[:,1]<50
            print(f'{sp} {g} {arm}: share against own prior {lo.mean():.2f}; precision when against own prior {R[lo,0].mean() if lo.any() else float("nan"):.2f}, when consistent {R[~lo,0].mean():.2f}')

print()
for gen,suf in (('sonnet',''),('opus','_opus')):
    for arm in ('BASE','A'):
        C=collections.Counter(); W=collections.Counter()
        for rep in s5.REPS:
            f=f'out2/run_mA100_nb_r{rep}{suf}.json' if arm=='A' else f'out2/run_nb_{s2.hid(s2.BASE)}_r{rep}{suf}.json'
            if not os.path.exists(f): continue
            for x in json.load(open(f))['hyps']:
                try: c=(int(x['edit_edge']),str(x['edit_op']),int(x['cond_edge']),str(x['cond_op'])); d=1 if str(x['direction']).lower().startswith('h') else -1
                except Exception: continue
                if c not in s1.H or c[0]!=c[2]: continue
                key=(c[1],c[3],'higher' if d>0 else 'lower'); C[key]+=1; W[key]+= s1.truth(c)!=d
        print(gen,arm,'self-claim patterns (edit op, cond op, dir): count / wrong')
        for k,n in C.most_common(6): print('    ',k,n,'/',W[k])
