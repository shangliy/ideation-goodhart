# Stage 1: structural knob screen for ideation harnesses on NAS-Bench-201 (see PREREG_s1.md).
# Usage: python3 s1.py show | prior | run [max_runs] | judge     (every call cached in out/, safe to re-run)
import sys,os,json,re,random,subprocess,tempfile,itertools,concurrent.futures as cf
import numpy as np
from nb_space import space,OPS,EDGES
from search import search
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'out'); os.makedirs(OUT,exist_ok=True)
GEN='sonnet'; JUDGE='opus'; PANEL=('haiku','sonnet','opus'); REPS=8; TAU=2.0
H=space()
L9=[(0,0,0,0),(0,1,1,1),(0,2,2,2),(1,0,1,2),(1,1,2,0),(1,2,0,1),(2,0,2,1),(2,1,0,2),(2,2,1,0)]  # (R,G,C,S)
def truth(cell):
    d=np.array(H[cell]['diffs']); return int(np.sign(d[0])) if np.all(np.sign(d)==np.sign(d[0])) and np.min(np.abs(d))>=TAU else 0
SETUP="\n".join([
 "Benchmark: NAS-Bench-201 cell search space. A cell is a DAG with nodes 0 (input) to 3 (output) and six edges; each edge carries one operation.",
 "Edges: "+"; ".join(f"edge {j} = node {e.split('<-')[1]} -> node {e.split('<-')[0]}" for j,e in enumerate(EDGES))+".",
 "Operations: "+", ".join(OPS)+" ('none' removes the edge).",
 "Datasets: CIFAR-10, CIFAR-100, ImageNet16-120; fitness is validation accuracy of the full network built by stacking the cell.",
 "An edit sets one edge j to operation b. A condition is 'parent edge k currently has operation c' (k may equal j, in which case it says what op b replaces).",
 "A hypothesis states that the edit's effect on validation accuracy is HIGHER (or LOWER) when the parent satisfies the condition than when it does not, "
 "consistently across the three datasets and by a practically meaningful margin. It is about the interaction, not the edit's average effect."])
def fmt(n): return ('End with JSON only: {"hypotheses": [{"edit_edge": <0-5>, "edit_op": "<op>", "cond_edge": <0-5>, "cond_op": "<op>", '
                    f'"direction": "higher" or "lower", "why": "<one sentence>"}}, ...]}} with exactly {n} distinct hypotheses.')
def call(model,prompt):
    with tempfile.TemporaryDirectory() as d:
        r=subprocess.run(['claude','-p','--model',model,'--tools','','--strict-mcp-config','--output-format','json'],input=prompt,capture_output=True,text=True,timeout=900,cwd=d)
    try:
        j=json.loads(r.stdout); u=j.get('usage',{}) or {}
        return j.get('result',''),sum(int(u.get(k,0) or 0) for k in ('input_tokens','output_tokens','cache_creation_input_tokens','cache_read_input_tokens'))
    except Exception: return 'ERROR '+r.stdout[-300:]+r.stderr[-300:],0
def lastjson(txt,key):
    for m in reversed(list(re.finditer(r'\{\s*"'+key+r'"\s*:',txt))):
        s=txt[m.start():]; depth=0
        for j,ch in enumerate(s):
            depth+= ch=='{'; depth-= ch=='}'
            if depth==0:
                try: return json.loads(s[:j+1])[key]
                except Exception: break
    return None
def step(key,model,prompt,key_json,check=lambda v: v is not None):
    f=os.path.join(OUT,key+'.json')
    if os.path.exists(f): return json.load(open(f))
    for _ in range(3):
        t,tok=call(model,prompt); v=lastjson(t,key_json)
        if check(v): rec=dict(ok=True,v=v,raw=t[:12000],tok=tok); json.dump(rec,open(f,'w')); return rec
    return dict(ok=False,v=None,raw=t[:500],tok=tok)
def hyps_ok(n): return lambda v: isinstance(v,list) and len(v)>=max(5,n//2)
def show(h): return json.dumps({'hypotheses':h})
NEG=("Before proposing, list 8 assumptions that practitioners usually hold about how operations on these edges interact. Negate each and ask what conditional "
     "effect would follow if it were false; build your hypotheses from these negated assumptions.")
QRY={1:"Write 3 literature search queries about neural network architecture knowledge (operations, skip connections, cell topology) that would help predict such interactions.",
     2:("Write 3 literature search queries, each in a DIFFERENT field outside machine learning (for example network flow, electrical circuits, ecology, hydrology, "
        "epidemiology, economics), about mechanisms whose structure resembles how parallel paths and operations in a small DAG interact.")}
def run(cfg,rep):
    R,G,C,S=cfg; tag=f"c{''.join(map(str,cfg))}_r{rep}"; n=10 if S==0 else 30; tok=0; trace={}
    base=SETUP+f"\n(Independent attempt #{rep+1}.)\n\n"
    def go(name,role_prompt,key_json='hypotheses',check=None):
        nonlocal tok
        r=step(f'{tag}_{name}',GEN,role_prompt,key_json,check or hyps_ok(n)); tok+=r.get('tok',0); trace[name]=r.get('ok'); return r
    lit=''
    if R:
        q=go('query',base+QRY[R]+'\nEnd with JSON only: {"queries": ["...", "...", "..."]}','queries',lambda v: isinstance(v,list) and len(v)>=1)
        if not q['ok']: return None
        res=[]
        for s in q['v'][:3]:
            res+= [f"- {x['title']}: {x['abstract'][:400]}" for x in search(str(s),4)]
        lit="Literature found by your searches (titles and abstracts):\n"+"\n".join(res)+"\n\n"
    neg=NEG+"\n" if C==2 else ''
    if G==0:
        r=go('gen',base+(lit if R else '')+neg+f"Propose {n} conditional hypotheses about this benchmark that you believe are true.\n"+fmt(n))
    else:
        gap=go('gap',base+(lit if R==1 else '')+("You are the gap finder. Describe what conventional architecture knowledge predicts about how operations on different edges interact, "
               "and identify 4 kinds of conditional effects that this conventional knowledge does NOT settle. Be concrete.")+
               '\nEnd with JSON only: {"gaps": ["...", ...]}','gaps',lambda v: isinstance(v,list) and len(v)>=2)
        if not gap['ok']: return None
        r=go('gen',base+(lit if R==2 else '')+"You are the innovator. The gap finder identified these open questions:\n"+"\n".join(f"- {g}" for g in gap['v'])+
             "\n\n"+neg+f"Propose {n} conditional hypotheses that address these gaps and that you believe could be true.\n"+fmt(n))
    if not r['ok']: return None
    h=r['v']
    if G==2:
        r=go('critic',base+"You are the critic. Review these hypotheses proposed by a colleague:\n"+show(h)+
             f"\n\nFor each, judge whether it is plausible and non-trivial; drop weak ones and replace them with better ones. Output the final {n}.\n"+fmt(n))
        if not r['ok']: return None
        h=r['v']
    if C==1:
        r=go('flaw',base+"Your hypotheses were:\n"+show(h)+"\n\nNow challenge them. For each, state the strongest flaw (a reason it could be false, a confound, or why the "
             f"condition should not matter), then keep, revise or replace it. Output your final {n}.\n"+fmt(n))
        if not r['ok']: return None
        h=r['v']
    if S:
        crit=("the 10 most likely to be true" if S==1 else
              "the 10 that other strong language models would be UNLIKELY to propose, but that could still be true (avoid the obvious, consensus ones)")
        r=go('select',base+"Candidate hypotheses:\n"+show(h)+f"\n\nSelect {crit}. Copy them exactly.\n"+fmt(10),check=hyps_ok(10))
        if not r['ok']: return None
        h=r['v']
    return dict(cfg=cfg,rep=rep,hyps=h[:10],tok=tok,trace=trace)
def to_claim(x):
    try:
        j,b,k,c=int(x['edit_edge']),str(x['edit_op']).strip(),int(x['cond_edge']),str(x['cond_op']).strip(); d=1 if str(x['direction']).lower().startswith('h') else -1
    except Exception: return None
    return (j,b,k,c,d) if (j,b,k,c) in H else None
CELLS=sorted(H); random.Random(0).shuffle(CELLS)
def ctext(c): j,b,k,cc=c; return f"setting edge {j} to {b} has a HIGHER effect when parent edge {k} is {cc} than when it is not"
def prior_job(model,i):
    B=CELLS[i:i+30]
    p=(SETUP+"\n\nFor each numbered claim, give your probability (0-100) that it holds (consistent across the three datasets, practically meaningful margin). "
       "If you think the effect is instead LOWER under the condition, give a low number.\n"+"\n".join(f"{n}. {ctext(c)}" for n,c in enumerate(B))+
       '\nEnd with JSON only: {"p": {"0": <0-100>, ...}} covering every claim.')
    r=step(f'prior_{model}_{i}',model,p,'p',lambda v: isinstance(v,dict) and all(str(n) in v for n in range(len(B))))
    return r
def judge_job(run_rec):
    tag=f"c{''.join(map(str,run_rec['cfg']))}_r{run_rec['rep']}"
    p=(SETUP+"\n\nA research assistant proposed these hypotheses:\n"+show(run_rec['hyps'])+
       "\n\nRate this set as a reviewer would, without running anything: novelty (1-10: would an expert find these surprising and non-obvious?) and "
       "quality (1-10: soundness, specificity, plausibility). End with JSON only: {\"score\": {\"novelty\": <1-10>, \"quality\": <1-10>}}")
    return step(f'judge_{tag}',JUDGE,p,'score',lambda v: isinstance(v,dict) and 'novelty' in v)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'show'
    if m=='show':
        print(SETUP+"\n\n"+NEG+"\nPropose 30 conditional hypotheses about this benchmark that you believe are true.\n"+fmt(30)); print('\n',len(H),'cells; valid',sum(truth(c)!=0 for c in H))
    if m=='prior':
        J=[(md,i) for md in PANEL for i in range(0,len(CELLS),30)]
        with cf.ThreadPoolExecutor(4) as ex:
            for (md,i),r in zip(J,ex.map(lambda j: prior_job(*j),J)): print('prior',md,i,'ok' if r['ok'] else 'FAIL',flush=True)
    if m=='run':
        J=[(cfg,r) for r in range(REPS) for cfg in L9][:int(sys.argv[2]) if len(sys.argv)>2 else None]
        with cf.ThreadPoolExecutor(4) as ex:
            for (cfg,r),rec in zip(J,ex.map(lambda j: run(*j),J)):
                if rec: json.dump(rec,open(os.path.join(OUT,f"run_c{''.join(map(str,cfg))}_r{r}.json"),'w'))
                print('run',cfg,r,'ok' if rec else 'FAIL',flush=True)
    if m=='judge':
        R=[json.load(open(os.path.join(OUT,f))) for f in sorted(os.listdir(OUT)) if f.startswith('run_')]
        with cf.ThreadPoolExecutor(4) as ex:
            for rr,j in zip(R,ex.map(judge_job,R)): print('judge',rr['cfg'],rr['rep'],'ok' if j['ok'] else 'FAIL',flush=True)
