# Second claim space: HPOBench tabular MLP benchmark (nn, 5 ordinal hyperparameters, 30,000 configs, 5 seeds each, max fidelity).
# Claim: "moving hyperparameter j one grid step UP (DOWN) has a HIGHER effect on validation accuracy when hyperparameter k is in band c than when it is not".
import os,json,itertools
import numpy as np
SRC=os.environ.get('FC_TAB',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data_dl','nn_tab.json'))
P=('alpha','batch_size','depth','learning_rate_init','width'); NAME=dict(alpha='L2 penalty (alpha)',batch_size='batch size',depth='depth (hidden layers)',
     learning_rate_init='initial learning rate',width='width (units per layer)')
BANDS=('low','mid','high'); TASKS=('9952','146822','53')  # phoneme, segment, vehicle (fixed before looking at claim validity; see PREREG_s9.md)
def band(p,i): return ('low','mid','high')[i] if p=='depth' else ('low' if i<=2 else 'mid' if i<=6 else 'high')
def load(split='val',tasks=TASKS):
    D=json.load(open(SRC)); L=D[tasks[0]]['levels']; keys=sorted(D[tasks[0]]['cells']); X=np.array([[int(x) for x in k.split('|')] for k in keys])
    Y={t:np.array([100*np.mean(D[t]['cells'][k][split]) for k in keys]) for t in tasks}; return L,keys,X,Y,D
EDITS=[(j,d) for j in range(5) for d in ('up','down')]
def space(split='val',tasks=TASKS):
    L,keys,X,Y,_=load(split,tasks); idx={tuple(r):i for i,r in enumerate(X.tolist())}; nl=[len(L[p]) for p in P]; H={}
    for j,d in EDITS:
        s=1 if d=='up' else -1; par=np.where((X[:,j]+s>=0)&(X[:,j]+s<nl[j]))[0]; ch=X[par].copy(); ch[:,j]+=s
        ci=np.array([idx[tuple(r)] for r in ch.tolist()]); Dl={t:Y[t][ci]-Y[t][par] for t in tasks}
        B=np.array([[band(P[k],i) for k,i in enumerate(r)] for r in X[par].tolist()])
        for k,c in itertools.product(range(5),BANDS):
            if P[k]=='depth' and c not in BANDS: continue
            m=B[:,k]==c
            if m.sum()<500 or (~m).sum()<500: continue
            H[(P[j],d,P[k],c)]=dict(diffs=[float(Dl[t][m].mean()-Dl[t][~m].mean()) for t in tasks],n=int(m.sum()))
    return H
if __name__=='__main__':
    import sys
    D=json.load(open(SRC))
    for t in D: print(t,'mean val acc',round(100*np.mean([np.mean(v['val']) for v in D[t]['cells'].values()]),1))
    H=space(); A=np.array([h['diffs'] for h in H.values()]); same=np.all(np.sign(A)==np.sign(A[:,:1]),1); mn=np.min(np.abs(A),1)*same
    print(len(H),'cells;',2*len(H),'claims; same-sign',same.sum())
    for tau in (0.25,0.5,1.0,1.5,2.0): print('tau',tau,'valid cells',int((mn>=tau).sum()),'random precision',round((mn>=tau).sum()/(2*len(H)),3))
