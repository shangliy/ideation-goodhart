# Stage 12: three-way panel prior (higher / lower / no interaction) and rescoring of every cached run, see PREREG_s12.md.
# Usage: python3 s12.py show | prior | analyze        (calls cached in out2/ as prior3_{nb,fc}_{model}_{i}.json; old priors untouched)
import os,sys,re,glob,json,collections,concurrent.futures as cf
import numpy as np
import s1,s2,s9
PANEL=s1.PANEL; SPACES=dict(nb=s1,fc=s9)
def ctext(sp,c):
    if sp=='nb': j,b,k,cc=c; return f"edit: set edge {j} to {b}; condition: parent edge {k} is {cc}"
    j,dd,k,cc=c; return f"edit: move {j} one step {dd}; condition: {k} is in the {cc} band"
def prompt(sp,i):
    B=SPACES[sp].CELLS[i:i+30]
    return (SPACES[sp].SETUP+"\n\nFor each numbered (edit, condition) pair, compare the edit's effect on validation accuracy when the condition holds with its "
            "effect when it does not. Give three probabilities (0-100, summing to 100): H = the effect is HIGHER under the condition, L = the effect is LOWER "
            "under the condition, N = neither (no difference that is consistent across the three datasets and of practically meaningful size).\n"
            +"\n".join(f"{n}. {ctext(sp,c)}" for n,c in enumerate(B))+'\nEnd with JSON only: {"p": {"0": [H, L, N], ...}} covering every pair.')
def ok(sp,i):
    n=len(SPACES[sp].CELLS[i:i+30])
    def chk(v):
        try: return isinstance(v,dict) and all(len(v[str(t)])==3 and sum(float(x) for x in v[str(t)])>0 for t in range(n))
        except Exception: return False
    return chk
def job(sp,model,i):
    r=s2.step(f'prior3_{sp}_{model}_{i}',model,prompt(sp,i),'p',ok(sp,i))
    if not r['ok']: r=s2.step(f'prior3_{sp}_{model}_{i}',model,prompt(sp,i),'p',ok(sp,i))  # one extra round (step already tries 3 times)
    return r
# --- three-way prior and leave-one-out scoring ---
P3={'nb':collections.defaultdict(dict),'fc':collections.defaultdict(dict)}
def load():
    for sp in P3:
        for f in glob.glob(os.path.join(s2.OUT,f'prior3_{sp}_*.json')):
            _,_,md,i=os.path.basename(f)[:-5].split('_'); v=json.load(open(f))['v']
            for n,c in enumerate(SPACES[sp].CELLS[int(i):int(i)+30]):
                x=np.array([float(t) for t in v[str(n)]]); P3[sp][c][md]=100*x/x.sum()
def p_dir(sp,cl,gen,mode='3'):
    """LOO panel probability of the stated direction. mode '3': H for HIGHER, L for LOWER; 'old': complement rule; 'mix': old H for HIGHER, new L for LOWER."""
    c,d=cl[:4],cl[4]; v=[]
    for m in PANEL:
        if m==gen: continue
        if mode=='old' or (mode=='mix' and d>0):
            old=(s2.NBP if sp=='nb' else s9.FCP)[c].get(m)
            if old is not None: v.append(old if d>0 else 100-old)
        elif m in P3[sp][c]: v.append(P3[sp][c][m][0 if d>0 else 1])
    return float(np.mean(v)) if v else np.nan
def twin(c):  # HPOBench: (j,up,k,c,d) with k!=j is the same proposition as (j,down,k,c,-d)
    j,dd,k,cc,d=c
    if k==j: return None
    o=(j,'down' if dd=='up' else 'up',k,cc)
    return (*o,-d) if o in s9.H else None
def canon(c):
    t=twin(c); return t if (t and c[1]=='down') else c
def prior(sp,cl,gen,mode='3'):
    p=p_dir(sp,cl,gen,mode)
    if sp=='fc' and mode!='old' and twin(cl): p=np.nanmean([p,p_dir(sp,twin(cl),gen,mode)])
    return p
def truth(sp,c): return SPACES[sp].truth(tuple(c[:4]))==c[4]
def claims(rec,merge=True):
    sp=rec['sp']; C=list(dict.fromkeys(c for c in map(s2.SP[sp]['claim'],rec['hyps']) if c))
    if sp=='fc' and merge: C=list(dict.fromkeys(canon(c) for c in C))
    return C
def score(rec,gen,thr=50,mode='3'):
    sp=rec['sp']; C=claims(rec,mode!='old'); val=[c for c in C if truth(sp,c)]
    P={c:prior(sp,c,gen,mode) for c in val}
    sv=[c for c in val if P[c]<thr]
    return dict(n=len(C),valid=len(val),prec=len(val)/max(len(C),1),sv=len(sv),svH=sum(c[4]>0 for c in sv),svL=sum(c[4]<0 for c in sv),
                sv1=sum(c[0]==c[2] for c in sv),sv2=sum(c[0]!=c[2] for c in sv),
                bits=float(sum(np.log2(100/max(P[c],1)) for c in val if not np.isnan(P[c]))),hi=sum(c[4]>0 for c in C)/max(len(C),1))
def gen_of(f):
    m=re.search(r'_r\d+(_[a-z]+)?\.json$',f); s=(m.group(1) or '')[1:]
    return s or 'sonnet'
def runs(pattern):
    out=[]
    for f in sorted(glob.glob(os.path.join(s2.OUT,pattern))):
        r=json.load(open(f))
        if r.get('sp') in ('nb','fc'): out.append((f,r))
    return out
ARMS=None
def arms():
    b,j=s2.hid(s2.BASE),s2.hid(json.load(open(os.path.join(s2.OUT,'train_J.json')))['final'])
    return {'nb':{'BASE':f'run_nb_{b}_r1[1-3][0-9]{{suf}}.json','A':'run_mA100_nb_r1[1-3][0-9]{suf}.json','J':f'run_nb_{j}_r1[1-3][0-9]{{suf}}.json'},
            'fc':{'BASE':f'run_fc_{b}_r2[0-2][0-9]{{suf}}.json','A':'run_fcA100_r2[0-2][0-9]{suf}.json','J':f'run_fc_{j}_r2[0-2][0-9]{{suf}}.json'}}
def arm_runs(sp,gen,arm):
    suf='' if gen=='sonnet' else '_'+gen; pat=arms()[sp][arm].format(suf=suf)
    R=[r for f,r in runs(pat) if gen_of(f)==gen]
    rep_ok=(lambda r: 116<=r['rep']<=139) if sp=='nb' else (lambda r: 200<=r['rep']<=223)
    return [r for r in R if rep_ok(r)]
def analyze():
    load(); rng=np.random.default_rng(0); res={}
    for sp in P3:
        cov=np.mean([len(P3[sp][c])==3 for c in SPACES[sp].CELLS]); print(f'{sp}: three-way prior coverage {cov:.3f} of {len(SPACES[sp].CELLS)} cells')
    # V1: share of true LOWER claims with LOO P(lower) < 50, averaged over the two generator LOO panels
    V1={}
    for sp in P3:
        T=[(*c,SPACES[sp].truth(c)) for c in SPACES[sp].CELLS if SPACES[sp].truth(c)<0]
        V1[sp]=float(np.mean([np.mean([prior(sp,t,g)<50 for t in T]) for g in ('sonnet','opus')]))
        old_hi=[m for c in SPACES[sp].CELLS for m in [(s2.NBP if sp=='nb' else s9.FCP)[c]] if m]
        r=np.corrcoef([[ (s2.NBP if sp=='nb' else s9.FCP)[c][m], P3[sp][c][m][0]] for c in SPACES[sp].CELLS for m in PANEL
                       if m in P3[sp][c] and m in (s2.NBP if sp=='nb' else s9.FCP)[c]],rowvar=False)[0,1]
        mean3={m:[round(float(np.mean([P3[sp][c][m][t] for c in SPACES[sp].CELLS if m in P3[sp][c]])),1) for t in range(3)] for m in PANEL}
        print(f'{sp}: true LOWER claims {len(T)}, share with LOO P(lower)<50 = {V1[sp]:.3f}; corr(old P(HIGHER), new H) = {r:.2f}; mean [H,L,N] by model {mean3}')
    tw=[abs(P3['fc'][c][m][0]-P3['fc'][(c[0],'down',c[2],c[3])][m][1]) for c in s9.CELLS if c[1]=='up' and c[0]!=c[2] and (c[0],'down',c[2],c[3]) in s9.H
        for m in PANEL if m in P3['fc'][c] and m in P3['fc'][(c[0],'down',c[2],c[3])]]
    print(f'fc twin wordings: median |H(up) - L(down)| = {np.median(tw):.1f} points over {len(tw)} model-pairs')
    v1='SUPPORT' if all(v>=.20 for v in V1.values()) else ('REFUTE' if all(v<.05 for v in V1.values()) else 'INCONCLUSIVE')
    # M3: A-100 minus BASE on SV3, Holm over 4 cells
    def perm(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
        return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
    def ci(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float); bs=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(5000)]
        return [float(np.percentile(bs,5)),float(np.percentile(bs,95))]
    M={}
    for sp in P3:
        for gen in ('sonnet','opus'):
            S={arm:[score(r,gen) for r in arm_runs(sp,gen,arm)] for arm in ('BASE','A','J')}
            a=[x['sv'] for x in S['A']]; b=[x['sv'] for x in S['BASE']]
            M[(sp,gen)]=dict(n=(len(S['BASE']),len(S['A'])),d=float(np.mean(a)-np.mean(b)),ci=ci(a,b),p=perm(a,b),
                             prec=(float(np.mean([x['prec'] for x in S['BASE']])),float(np.mean([x['prec'] for x in S['A']]))),
                             arms={arm:{k:float(np.mean([x[k] for x in L])) for k in ('sv','svH','svL','sv1','sv2','bits','prec','hi')} for arm,L in S.items() if L},
                             sens={f'{mode}@{thr}':float(np.mean([score(r,gen,thr,mode)['sv'] for r in arm_runs(sp,gen,'A')])-np.mean([score(r,gen,thr,mode)['sv'] for r in arm_runs(sp,gen,'BASE')]))
                                   for mode,thr in (('old',50),('mix',50),('3',33),('3',40),('3',60))},
                             bits_p=perm([x['bits'] for x in S['A']],[x['bits'] for x in S['BASE']]),
                             J_minus_BASE=float(np.mean([x['sv'] for x in S['J']])-np.mean(b)) if S['J'] else None)
    order=sorted(M,key=lambda k: M[k]['p']); prev=0
    for i,k in enumerate(order): prev=max(prev,min(1,(len(order)-i)*M[k]['p'])); M[k]['holm']=prev
    V={}
    for k,o in M.items():
        V[k]='SUPPORT' if o['holm']<.05 and o['prec'][1]>=0.8*o['prec'][0] else ('REFUTE' if o['ci'][1]<0.5 else 'INCONCLUSIVE')
        A_,B_=o['arms']['A'],o['arms']['BASE']
        print(f"\nM3 {k[0]} {k[1]}: SV3 BASE {B_['sv']:.2f} -> A {A_['sv']:.2f} (d {o['d']:+.2f}, 90% CI [{o['ci'][0]:+.2f},{o['ci'][1]:+.2f}], p {o['p']:.3f}, holm {o['holm']:.3f}) "
              f"prec {o['prec'][0]:.2f}->{o['prec'][1]:.2f} => {V[k]}")
        print(f"   SV3 by direction H/L: BASE {B_['svH']:.2f}/{B_['svL']:.2f}  A {A_['svH']:.2f}/{A_['svL']:.2f};  by order k=j / k!=j: BASE {B_['sv1']:.2f}/{B_['sv2']:.2f}  A {A_['sv1']:.2f}/{A_['sv2']:.2f}")
        print(f"   surprisal-weighted credit BASE {B_['bits']:.2f} A {A_['bits']:.2f} (p {o['bits_p']:.3f});  J - BASE on SV3 {o['J_minus_BASE']:+.2f}")
        print('   sensitivity (A - BASE):',{k2:round(v,2) for k2,v in o['sens'].items()})
    print('\npre-registered verdicts: V1',v1,'| M3',{f'{a} {b}':v for (a,b),v in V.items()})
    # descriptive: every cached arm on SV (old) and SV3
    G=collections.defaultdict(list)
    for f,r in runs('run_*.json'):
        key=re.sub(r'_r\d+(_[a-z]+)?\.json$',r'\1',os.path.basename(f)); g=gen_of(f); s2.GEN=g
        G[key].append((s2.score(r)['sv'],score(r,g)['sv'],score(r,g)['prec']))
    table={k:dict(n=len(v),sv_old=float(np.mean([x[0] for x in v])),sv3=float(np.mean([x[1] for x in v])),prec=float(np.mean([x[2] for x in v]))) for k,v in sorted(G.items())}
    json.dump(dict(V1=V1,V1_verdict=v1,M3={f'{a}_{b}':{**o,'verdict':V[(a,b)]} for (a,b),o in M.items()},all_arms=table),open(os.path.join(s1.HERE,'s12_results.json'),'w'),indent=1)
    print('\nall arms written to s12_results.json')
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'show'
    if m=='show': print(prompt('nb',0)[-1600:]); print('---'); print(prompt('fc',0)[-900:])
    if m=='prior':
        J=[(sp,md,i) for sp in ('fc','nb') for md in PANEL for i in range(0,len(SPACES[sp].CELLS),30)]
        with cf.ThreadPoolExecutor(4) as ex:
            for (sp,md,i),r in zip(J,ex.map(lambda j: job(*j),J)): print('prior3',sp,md,i,'ok' if r['ok'] else 'FAIL',flush=True)
    if m=='analyze': analyze()
