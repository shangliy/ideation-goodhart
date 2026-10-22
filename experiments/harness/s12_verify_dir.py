# Verify the direction / wording claims in the pasted theory note against our own cache (no API calls).
import sys,os,collections
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); os.chdir(HERE)
import numpy as np
import s1,s2,s5,s9
rng=np.random.default_rng(0)
def perm(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
    return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))

print('== 1. panel mean P(HIGHER)')
for name,P in (('NB',s2.NBP),('FC',s9.FCP)):
    print(name,{m:round(float(np.mean([v[m] for v in P.values() if m in v])),1) for m in s1.PANEL})

print('\n== 2. true directed claims and SV-eligible claims by direction')
for name,H,truth,P,prior in (('NB',s1.H,s1.truth,s2.NBP,s2.nb_prior),('FC',s9.H,s9.truth,s9.FCP,s9.fc_prior)):
    T=[(*c,truth(c)) for c in H if truth(c)!=0]
    low=np.mean([t[4]<0 for t in T])
    for gen in ('sonnet','opus'):
        s2.GEN=gen; sv=[t for t in T if prior(t)<50]
        print(f'{name} gen={gen}: true {len(T)} (LOWER {low:.2f}); SV-eligible {len(sv)} (LOWER {np.mean([t[4]<0 for t in sv]) if sv else 0:.2f})')

print('\n== 3. arm composition: HIGHER share, precision by direction, SV')
def runs(sp,gen,arm): return s5.runs(gen,arm) if sp=='NB' else s9.runs(gen,arm)
for sp in ('NB','FC'):
    truth=s1.truth if sp=='NB' else s9.truth
    for gen in ('sonnet','opus'):
        for arm in ('BASE','A'):
            R=runs(sp,gen,arm); C=[tuple(c) for r in R for c in r['claims']]
            hi=[c for c in C if c[4]>0]; lo=[c for c in C if c[4]<0]
            ph=np.mean([truth(c[:4])==c[4] for c in hi]) if hi else np.nan; pl=np.mean([truth(c[:4])==c[4] for c in lo]) if lo else np.nan
            print(f'{sp} {gen:6s} {arm:4s} n={len(R)} HIGHER share {len(hi)/len(C):.2f}  prec H {ph:.2f} L {pl:.2f}  SV/run {np.mean([r["sv"] for r in R]):.2f}  true-LOWER/run {sum(truth(c[:4])==c[4] for c in lo)/len(R):.2f}')

print('\n== 4. HPOBench up/down equivalence for k != j')
pairs=0; maxerr=0; tr_ok=0
for (j,dd,k,c) in s9.H:
    if dd!='up' or k==j: continue
    o=(j,'down',k,c)
    if o in s9.H:
        pairs+=1; a=np.array(s9.H[(j,dd,k,c)]['diffs']); b=np.array(s9.H[o]['diffs']); maxerr=max(maxerr,float(np.max(np.abs(a+b))))
        tr_ok+= s9.truth((j,dd,k,c))==-s9.truth(o)
print(f'pairs {pairs}, max |diff_up + diff_down| = {maxerr:.2e}, truth negated in {tr_ok}/{pairs}')
# label disagreement between the two wordings of the same proposition, by panel member
for m in s1.PANEL:
    dis=n=0
    for (j,dd,k,c) in s9.H:
        if dd!='up' or k==j or (j,'down',k,c) not in s9.H: continue
        pu=s9.FCP[(j,dd,k,c)].get(m); pd=s9.FCP[(j,'down',k,c)].get(m)
        if pu is None or pd is None: continue
        # proposition "up HIGHER" == "down LOWER": surprise labels under the complement rule
        n+=1; dis+= (pu<50)!=((100-pd)<50)
    print(f'  panel {m}: opposite surprise labels for the same proposition in {dis}/{n} pairs')

print('\n== 5. HPOBench M under wording-invariant scoring (A-100 minus BASE, one-sided perm p)')
def canon(c):  # map (j,down,k,c,d) with k!=j to the equivalent (j,up,k,c,-d) when that cell exists
    j,dd,k,cc,d=c
    if k!=j and dd=='down' and (j,'up',k,cc) in s9.H: return (j,'up',k,cc,-d)
    return c
def fc_prior_direct(c):  # prior from the panel's direct HIGHER rating of whichever wording is HIGHER
    j,dd,k,cc,d=c
    if d>0: return s9.fc_prior(c)
    if k!=j:
        o=(j,'down' if dd=='up' else 'up',k,cc)
        if o in s9.H: return s9.fc_prior((*o,1))
    return s9.fc_prior(c)
def fc_prior_avg(c):
    j,dd,k,cc,d=c
    if k!=j:
        o=(j,'down' if dd=='up' else 'up',k,cc)
        if o in s9.H: return (s9.fc_prior(c)+s9.fc_prior((*o,-d)))/2
    return s9.fc_prior(c)
for gen in ('sonnet','opus'):
    s2.GEN=gen; out={}
    for arm in ('BASE','A'):
        R=s9.runs(gen,arm); out[arm]={}
        for nm,pr,dd in (('original',s9.fc_prior,False),('dedup',s9.fc_prior,True),('dedup+avg',fc_prior_avg,True),('dedup+direct',fc_prior_direct,True)):
            v=[]
            for r in R:
                C=[tuple(c) for c in r['claims']]
                if dd:
                    seen={};
                    for c in C: seen.setdefault(canon(c),c)
                    C=list(seen.values())
                v.append(sum(1 for c in C if s9.truth(c[:4])==c[4] and pr(c)<50))
            out[arm][nm]=v
    pairs_run={arm:np.mean([len(r['claims'])-len({canon(tuple(c)) for c in r['claims']}) for r in s9.runs(gen,arm)]) for arm in ('BASE','A')}
    print(gen,'duplicate-wording pairs/run',{k:round(v,2) for k,v in pairs_run.items()})
    for nm in out['A']:
        a,b=out['A'][nm],out['BASE'][nm]; print(f'  {gen:6s} {nm:13s} BASE {np.mean(b):.2f} A {np.mean(a):.2f} diff {np.mean(a)-np.mean(b):+.2f} p {perm(a,b):.3f}')
