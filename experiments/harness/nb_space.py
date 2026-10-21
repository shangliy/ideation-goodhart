# NB201 hypothesis space for the harness study: "setting edge j to op b has a higher/lower effect when parent edge k == op c".
# space() reads the shipped nb_space.json (derived from NAS-Bench-201). Rebuilding it (python3 nb_space.py) needs a tabulated NB201 loader on NB201_LOADER, see README.
import os,sys,json,itertools
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); JSON=os.path.join(HERE,'nb_space.json')
OPS=['none','skip_connect','nor_conv_1x1','nor_conv_3x3','avg_pool_3x3']; EDGES=['1<-0','2<-0','2<-1','3<-0','3<-1','3<-2']; DS=('C10','C100','IN16')
def arrays():
    sys.path.insert(0,os.environ['NB201_LOADER']); from bench.nb201 import table
    C=sorted(table('C10')); X=np.array([[OPS.index(o) for o in c] for c in C]); idx={tuple(r):i for i,r in enumerate(X.tolist())}
    Y={ds:np.array([table(ds)[c] for c in C]) for ds in DS}; return C,X,idx,Y
def space(rebuild=False):
    if os.path.exists(JSON) and not rebuild:
        return {(int(k.split('|')[0]),k.split('|')[1],int(k.split('|')[2]),k.split('|')[3]):v for k,v in json.load(open(JSON)).items()}
    C,X,idx,Y=arrays(); H={}
    for j,b in itertools.product(range(6),range(5)):
        par=np.where(X[:,j]!=b)[0]; ch=X[par].copy(); ch[:,j]=b; ci=np.array([idx[tuple(r)] for r in ch.tolist()])
        D={ds:Y[ds][ci]-Y[ds][par] for ds in DS}
        for k,c in itertools.product(range(6),range(5)):
            if k==j and c==b: continue
            m=X[par,k]==c
            if m.sum()<100 or (~m).sum()<100: continue
            diffs=[float(D[ds][m].mean()-D[ds][~m].mean()) for ds in DS]
            H[(j,OPS[b],k,OPS[c])]=dict(diffs=diffs,n=int(m.sum()))
    return H
if __name__=='__main__':
    H=space(rebuild=True); A=np.array([h['diffs'] for h in H.values()])
    same=np.all(np.sign(A)==np.sign(A[:,:1]),axis=1); mn=np.min(np.abs(A),axis=1)*same
    print(len(H),'cells;',2*len(H),'directional claims')
    for q in (50,75,90,95,97.5): print('pct',q,'of min|diff| (same-sign)',round(float(np.percentile(mn,q)),3))
    for tau in (0.25,0.5,1.0,2.0): print('tau',tau,'valid cells',int((mn>=tau).sum()))
    json.dump({'|'.join(map(str,k)):h for k,h in H.items()},open('nb_space.json','w'))
