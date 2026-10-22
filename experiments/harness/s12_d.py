# Stage 12, post hoc: claim D (19 internal-reasoning harnesses vs direct generation) rescored with the three-way prior (SV3) and R.
import os,sys,json,glob
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); os.chdir(HERE)
import numpy as np, s1, s12
from s12_rmulti import fit, perm
def holm(ps):
    o=np.argsort(ps); adj=np.empty(len(ps)); run=0
    for i,k in enumerate(o): run=max(run,min(1,(len(ps)-i)*ps[k])); adj[k]=run
    return adj
def from_claims(C,gen,pi):
    C=list(dict.fromkeys(tuple(c) for c in C)); val=[c for c in C if s12.truth('nb',c)]
    return dict(sv3=sum(1 for c in val if s12.prior('nb',c,gen)<50),sv_old=sum(1 for c in val if s12.prior('nb',c,gen,'old')<50),
                prec=len(val)/max(len(C),1),R=sum(s12.truth('nb',c)-pi(c,s12.prior('nb',c,gen)) for c in C))
def rec_claims(r): return [c for c in map(s1.to_claim,r['hyps']) if c]
fam={}
pi={g:fit('nb',g,'3') for g in ('sonnet','opus')}
# Sonnet
F=json.load(open('out2/final.json')); S={}
for cfg in sorted({os.path.basename(f)[5:9] for f in glob.glob('out/run_c*.json')}):
    S[f's1 c{cfg}']=[from_claims(rec_claims(json.load(open(f))),'sonnet',pi['sonnet']) for f in sorted(glob.glob(f'out/run_c{cfg}_r*.json'))]
for nm in ('BASE','final_J','final_V'): S[f's2 {nm}']=[from_claims(r['claims'],'sonnet',pi['sonnet']) for r in F[f'nb|{nm}']['runs']]
S['s2 final_SV2']=[from_claims(r['claims'],'sonnet',pi['sonnet']) for r in json.load(open('out2/fresh_SV2.json'))['runs']]
s5={a:[from_claims(s12.claims(r),'sonnet',pi['sonnet']) for r in s12.arm_runs('nb','sonnet',a)] for a in ('BASE','J')}
S['s5 BASE'],S['s5 J']=s5['BASE'],s5['J']
S['s8 P100']=[from_claims(s12.claims(r),'sonnet',pi['sonnet']) for f,r in s12.runs('run_mP100_nb_r1[1-3][0-9].json') if s12.gen_of(f)=='sonnet']
cmp_s=[(k,'s1 c0000') for k in S if k.startswith('s1') and k!='s1 c0000']+[(k,'s2 BASE') for k in ('s2 final_J','s2 final_V','s2 final_SV2')]+[('s5 J','s5 BASE'),('s8 P100','s5 BASE')]
# Opus
O=json.load(open('out2/frontier_opus.json')); Q={f's3 {k}':[from_claims(r['claims'],'opus',pi['opus']) for r in v['runs']] for k,v in O.items()}
s5o={a:[from_claims(s12.claims(r),'opus',pi['opus']) for r in s12.arm_runs('nb','opus',a)] for a in ('BASE','J')}
Q['s5 BASE'],Q['s5 J']=s5o['BASE'],s5o['J']
Q['s8 P100']=[from_claims(s12.claims(r),'opus',pi['opus']) for f,r in s12.runs('run_mP100_nb_r1[1-3][0-9]_opus.json') if s12.gen_of(f)=='opus']
cmp_o=[(k,'s3 BASE') for k in Q if k.startswith('s3') and k!='s3 BASE']+[('s5 J','s5 BASE'),('s8 P100','s5 BASE')]
out={}
for g,H,CMP in (('sonnet',S,cmp_s),('opus',Q,cmp_o)):
    print(f'\n== {g}: {len(CMP)} harnesses')
    res={}
    for m in ('sv_old','sv3','R'):
        ps=[perm([x[m] for x in H[a]],[x[m] for x in H[b]]) for a,b in CMP]; adj=holm(np.array(ps)); res[m]=(ps,adj)
    rows=[]
    for i,(a,b) in enumerate(CMP):
        d={m:float(np.mean([x[m] for x in H[a]])-np.mean([x[m] for x in H[b]])) for m in ('sv_old','sv3','R','prec')}
        rows.append(dict(harness=a,base=b,n=len(H[a]),**{f'd_{m}':v for m,v in d.items()},**{f'holm_{m}':float(res[m][1][i]) for m in ('sv_old','sv3','R')},p_sv3=float(res['sv3'][0][i]),p_R=float(res['R'][0][i])))
        print(f"  {a:16s} n={len(H[a]):2d} d_prec {d['prec']:+.2f}  d_SVold {d['sv_old']:+.2f} (holm {res['sv_old'][1][i]:.2f})  d_SV3 {d['sv3']:+.2f} (p {res['sv3'][0][i]:.3f}, holm {res['sv3'][1][i]:.2f})  d_R {d['R']:+.2f} (p {res['R'][0][i]:.3f}, holm {res['R'][1][i]:.2f})")
    out[g]=rows
    print('  any Holm<.05 on SV3:',any(r['holm_sv3']<.05 for r in rows),' on R:',any(r['holm_R']<.05 for r in rows))
json.dump(out,open('s12_d.json','w'),indent=1)
