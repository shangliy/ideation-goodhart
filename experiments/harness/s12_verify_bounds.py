# Bounds on M when LOWER surprise is unmeasured, plus blocked-probe share and rep-116 sensitivity (no API calls).
import sys,os,json
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); os.chdir(HERE)
import numpy as np
import s1,s2,s5,s9
rng=np.random.default_rng(0)
def perm(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
    return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
print('== M (A-100 minus BASE) when true LOWER claims count as surprising: none (current) vs all (upper bound for BASE)')
for sp in ('NB','FC'):
    truth=s1.truth if sp=='NB' else s9.truth
    for gen in ('sonnet','opus'):
        R={a:(s5.runs(gen,a) if sp=='NB' else s9.runs(gen,a)) for a in ('BASE','A')}
        cur={a:[r['sv'] for r in R[a]] for a in R}
        allL={a:[r['sv']+sum(1 for c in r['claims'] if c[4]<0 and truth(tuple(c[:4]))==c[4]) for r in R[a]] for a in R}
        print(f'{sp} {gen:6s} current {np.mean(cur["A"])-np.mean(cur["BASE"]):+.2f} (p {perm(cur["A"],cur["BASE"]):.3f})   all-LOWER-surprising {np.mean(allL["A"])-np.mean(allL["BASE"]):+.2f} (BASE {np.mean(allL["BASE"]):.2f}, A {np.mean(allL["A"]):.2f})')

print('\n== blocked probes: share of proposed NB claims with >=1 probe pair differing only on edge k (same edit, one side k=c)')
for gen in ('sonnet','opus'):
    suf='' if gen=='sonnet' else '_opus'; tot=hit=0; prec={0:[],1:[],2:[]}
    for rep in s5.REPS:
        f=f'out2/run_mA100_nb_r{rep}{suf}.json'
        if not os.path.exists(f): continue
        r=json.load(open(f)); P=[(tuple(p),(e[0],e[1])) for p,e in r['probes']]
        s2.GEN=gen; sc=s2.score(r)
        for c in sc['claims']:
            j,b,k,cc,d=c; same=[p for p,e in P if e==(j,b)]
            npair=0
            for i in range(len(same)):
                for t in range(i+1,len(same)):
                    p,q=same[i],same[t]; diff=[x for x in range(6) if p[x]!=q[x]]
                    if diff==[k] and (p[k]==cc or q[k]==cc): npair+=1
            tot+=1; hit+= npair>0; prec[min(npair,2)].append(s1.truth((j,b,k,cc))==d)
    print(f'{gen}: {hit/tot:.3f} of {tot} claims blocked; precision by #pairs 0/1/>=2: '+' / '.join(f'{np.mean(v):.3f} (n={len(v)})' for v in prec.values()))

print('\n== Sonnet NB M excluding rep 116 (0 parsed probes)')
s2.GEN='sonnet'
for a in ('BASE','A'):
    R=s5.runs('sonnet',a); print(a,'all',round(np.mean([r['sv'] for r in R]),3),round(np.mean([r['prec'] for r in R]),3))
r116=json.load(open('out2/run_mA100_nb_r116.json')); print('rep116 probes',len(r116['probes']),'bad',r116.get('bad'))
R=[s2.score(json.load(open(f'out2/run_mA100_nb_r{rep}.json'))) for rep in s5.REPS if rep!=116]
print('A excl 116: SV',round(np.mean([r['sv'] for r in R]),3),'prec',round(np.mean([r['prec'] for r in R]),3))
