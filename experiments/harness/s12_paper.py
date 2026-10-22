# Stage 12, paper numbers: precision, original SV, SV3 and R (multi-covariate calibration) for every key arm (no API calls).
import os,sys,json
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); os.chdir(HERE)
import numpy as np, s12, s2
from s12_rmulti import fit, perm
def arm_list(sp,gen):
    suf='' if gen=='sonnet' else '_'+gen
    A={a:s12.arm_runs(sp,gen,a) for a in ('BASE','J','A')}
    if sp=='nb':
        for tag in (['P100']+(['A100S','A100N1','A100N2','A100L','A100D0'] if gen=='opus' else [])):
            A[tag]=[r for f,r in s12.runs(f'run_m{tag}_nb_r1[1-3][0-9]{suf}.json') if s12.gen_of(f)==gen]
    return A
out={}
for sp in ('nb','fc'):
    for gen in ('sonnet','opus'):
        pi=fit(sp,gen,'3'); s2.GEN=gen; A=arm_list(sp,gen); row={}
        Rb=[sum(s12.truth(sp,c)-pi(c,s12.prior(sp,c,gen)) for c in s12.claims(r)) for r in A['BASE']]
        for arm,R in A.items():
            if not R: continue
            Rv=[sum(s12.truth(sp,c)-pi(c,s12.prior(sp,c,gen)) for c in s12.claims(r)) for r in R]
            row[arm]=dict(n=len(R),prec=float(np.mean([s12.score(r,gen)['prec'] for r in R])),sv_old=float(np.mean([s12.score(r,gen,50,'old')['sv'] for r in R])),
                          sv3=float(np.mean([s12.score(r,gen)['sv'] for r in R])),R=float(np.mean(Rv)),
                          dR=float(np.mean(Rv)-np.mean(Rb)),p_up=perm(Rv,Rb),p_down=perm(Rb,Rv))
        out[f'{sp}_{gen}']=row
        print(sp,gen); [print(f'  {a:7s} n={v["n"]:2d} prec {v["prec"]:.2f} SVold {v["sv_old"]:.2f} SV3 {v["sv3"]:.2f} R {v["R"]:+.2f} dR {v["dR"]:+.2f} p_up {v["p_up"]:.3f} p_down {v["p_down"]:.3f}') for a,v in row.items()]
json.dump(out,open('s12_paper.json','w'),indent=1)
