# R with multi-calibration: logistic on [logit P, order (k==j), direction, logit P x order], three-way prior and old prior side by side.
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); os.chdir(HERE)
import numpy as np, s12
s12.load(); rng=np.random.default_rng(0)
def perm(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); x=np.r_[a,b]; d=a.mean()-b.mean()
    return float(np.mean([(lambda p: p[:len(a)].mean()-p[len(a):].mean())(rng.permutation(x))>=d for _ in range(5000)]))
def feats(c,p): l=np.log(max(p,.5)/max(100-p,.5)); o=float(c[0]==c[2]); return np.array([l,o,float(c[4]>0),l*o,1.0])
def fit(sp,gen,mode):
    X,Y=[],[]
    for c in s12.SPACES[sp].CELLS:
        t=s12.SPACES[sp].truth(c)
        for d in (1,-1): cl=(*c,d); X.append(feats(cl,s12.prior(sp,cl,gen,mode))); Y.append(float(t==d))
    X,Y=np.array(X),np.array(Y); w=np.zeros(X.shape[1]); w[-1]=np.log(Y.mean()/(1-Y.mean()))
    for _ in range(200):
        z=1/(1+np.exp(-X@w)); g=X.T@(z-Y)+1e-3*w; Hm=(X*(z*(1-z))[:,None]).T@X+1e-3*np.eye(len(w)); s=np.linalg.solve(Hm,g); w-=s
        if np.abs(s).max()<1e-9: break
    return lambda cl,p: 1/(1+np.exp(-feats(cl,p)@w))
if __name__=='__main__':
    for mode in ('3','old'):
        print('== prior mode',mode)
        for sp in ('nb','fc'):
            for gen in ('sonnet','opus'):
                pi=fit(sp,gen,mode); R={}
                for arm in ('BASE','A','J'):
                    R[arm]=[sum(s12.truth(sp,c)-pi(c,s12.prior(sp,c,gen,mode)) for c in s12.claims(r,mode!='old')) for r in s12.arm_runs(sp,gen,arm)]
                print(f"  {sp} {gen:6s} R BASE {np.mean(R['BASE']):+.2f}  A {np.mean(R['A']):+.2f} (d {np.mean(R['A'])-np.mean(R['BASE']):+.2f}, p {perm(R['A'],R['BASE']):.3f})  "
                      f"J {np.mean(R['J']):+.2f} (d {np.mean(R['J'])-np.mean(R['BASE']):+.2f}, p_lower {perm(R['BASE'],R['J']):.3f})")
