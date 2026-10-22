# Stage 12, post hoc (descriptive): threshold sweep and threshold-free metrics under the three-way prior.
# R = sum over proposed claims of (truth - pi), pi = panel prior of the stated direction calibrated to truth (logistic on logit P, fit on every
# directed claim of the space, per leave-one-out panel). R > 0 means the run's claims are truer than the panel would have expected.
import sys,json
import numpy as np
import s12
s12.load(); rng=np.random.default_rng(0)
def perm(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
    return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
def calib(sp,gen):
    X,Y=[],[]
    for c in s12.SPACES[sp].CELLS:
        t=s12.SPACES[sp].truth(c)
        for d in (1,-1):
            p=s12.prior(sp,(*c,d),gen); X.append(np.log(max(p,.5)/max(100-p,.5))); Y.append(t==d)
    X,Y=np.array(X),np.array(Y,float); a,b=0.0,np.log(Y.mean()/(1-Y.mean()))
    for _ in range(500):  # Newton steps for 1-D logistic regression
        z=1/(1+np.exp(-(a*X+b))); g=np.array([np.sum((z-Y)*X),np.sum(z-Y)]); w=z*(1-z)
        Hm=np.array([[np.sum(w*X*X),np.sum(w*X)],[np.sum(w*X),np.sum(w)]])+1e-9*np.eye(2); s=np.linalg.solve(Hm,g); a,b=a-s[0],b-s[1]
        if np.abs(s).max()<1e-10: break
    return lambda p: 1/(1+np.exp(-(a*np.log(max(p,.5)/max(100-p,.5))+b)))
THR=(10,15,20,25,33,40,50)
out={}
for sp in ('nb','fc'):
    for gen in ('sonnet','opus'):
        pi=calib(sp,gen); row={}
        for arm in ('BASE','A','J')+(('P',) if sp=='nb' else ()):
            R=s12.arm_runs(sp,gen,arm) if arm!='P' else [r for f,r in s12.runs(f"run_mP100_nb_r1[1-3][0-9]{'' if gen=='sonnet' else '_'+gen}.json") if s12.gen_of(f)==gen]
            M=dict(thr={t:[s12.score(r,gen,t)['sv'] for r in R] for t in THR})
            M['R']=[sum(s12.truth(sp,c)-pi(s12.prior(sp,c,gen)) for c in s12.claims(r)) for r in R]
            M['bits']=[s12.score(r,gen)['bits'] for r in R]
            row[arm]=M
        out[(sp,gen)]=row
        B=row['BASE']; print(f'\n== {sp} {gen}')
        for arm in row:
            if arm=='BASE': continue
            A=row[arm]
            sweep='  '.join(f"<{t}: {np.mean(A['thr'][t])-np.mean(B['thr'][t]):+.2f} (p {perm(A['thr'][t],B['thr'][t]):.2f})" for t in THR)
            print(f"  {arm}-BASE  {sweep}")
            print(f"           R {np.mean(B['R']):+.2f} -> {np.mean(A['R']):+.2f} (d {np.mean(A['R'])-np.mean(B['R']):+.2f}, p {perm(A['R'],B['R']):.3f});  "
                  f"bits {np.mean(B['bits']):.2f} -> {np.mean(A['bits']):.2f} (p {perm(A['bits'],B['bits']):.3f})")
        print('  BASE SV3 by threshold:',{t:round(float(np.mean(B['thr'][t])),2) for t in THR})
json.dump({f'{a}_{b}':{arm:{'R':float(np.mean(M['R'])),'bits':float(np.mean(M['bits'])),**{f'sv<{t}':float(np.mean(M['thr'][t])) for t in THR}} for arm,M in row.items()}
           for (a,b),row in out.items()},open('s12_post.json','w'),indent=1)
