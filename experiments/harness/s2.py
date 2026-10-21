# Stage 2: optimize an ideation harness under three fitness targets (J judge / V validity / SV surprising-valid) on NB201,
# then test on fresh NB201 runs (and, in the original study, a private CTR space that is not released). See PREREG_s2.md.
# Usage: python3 s2.py show | train [J V SV SV2] | final | fresh ARM | frontier   (S2_GEN=opus switches the generator; see PREREG_s3.md)
#              (every call cached in out2/, safe to re-run)
import sys,os,json,re,glob,hashlib,threading,collections,concurrent.futures as cf
import numpy as np
import s1
from search import search
OUT=os.path.join(s1.HERE,'out2'); os.makedirs(OUT,exist_ok=True)
GEN=os.environ.get('S2_GEN','sonnet'); SUF='' if GEN=='sonnet' else '_'+GEN; JUDGE,PROP='opus','opus'; ROUNDS,NCAND,M,CAP=3,3,6,40000; FINAL_REPS=range(100,108)
SEM=threading.BoundedSemaphore(6)

# --- the two spaces: claim parser, truth, leave-one-out prior (generator excluded) ---
NBP=collections.defaultdict(dict)
for f in glob.glob(s1.OUT+'/prior_*.json'):
    _,md,i=os.path.basename(f)[:-5].split('_')
    for n,c in enumerate(s1.CELLS[int(i):int(i)+30]): NBP[c][md]=float(json.load(open(f))['v'][str(n)])
def nb_prior(cl): v=[p if cl[4]>0 else 100-p for m,p in NBP[cl[:4]].items() if m!=GEN]; return float(np.mean(v)) if v else np.nan
# The original study also registered a private CTR space here (same interface: setup, fmt, claim, valid, prior) for the transfer test.
# It depends on an unreleased benchmark and is not included; final() evaluates every space present in SP.
SP={'nb':dict(setup=s1.SETUP,fmt=s1.fmt,claim=s1.to_claim,valid=lambda c: s1.truth(c[:4])==c[4],prior=nb_prior)}

def step(key,model,prompt,key_json,check=lambda v: v is not None):
    f=os.path.join(OUT,key+'.json')
    if os.path.exists(f): return json.load(open(f))
    for _ in range(3):
        with SEM: t,tok=s1.call(model,prompt)
        v=s1.lastjson(t,key_json)
        if check(v): rec=dict(ok=True,v=v,raw=t[:12000],tok=tok); json.dump(rec,open(f,'w')); return rec
    return dict(ok=False,v=None,raw=t[:500],tok=tok)

# --- harness and its runner (role texts are domain-general; SETUP and output format stay outside the harness) ---
BASE=dict(retrieval='none',roles='single',challenge='none',pool=10,select='likely_true',select_rule='',strategy='')
ENUM=dict(retrieval=('none','in_domain','cross_domain'),roles=('single','gap_innovator','gap_innovator_critic'),challenge=('none','flaw','negate'),
          pool=(10,20,30),select=('likely_true','prior_avoid','custom'))
BAN=re.compile(r'\b(conv(olution\w*|\d\w*)?|pool(ing|s)?|skip\w*|cifar\w*|imagenet\w*|edges?|nodes?|cells?|nas|dag|emb\w*|fm|mlp|dropout|autoint|crossnet\w*|bilinear|afm|senet|'
               r'focal|bpr|ctr|auc|weight decay|learning rate|movielens|kuairand|frappe|amazon|nb201|nas-bench\w*|click\w*)\b',re.I)
def clean(h):
    h={k:h.get(k,BASE[k]) for k in BASE}
    try: h['pool']=int(h['pool'])
    except Exception: return None,'pool not int'
    for k,vals in ENUM.items():
        if h[k] not in vals: return None,f'{k}={h[k]!r} not allowed'
    for k,lim in (('strategy',150),('select_rule',50)):
        h[k]=str(h[k] or '').strip()
        if len(h[k].split())>lim: return None,f'{k} over {lim} words'
        if BAN.search(h[k]): return None,f'{k} uses space-specific term {BAN.search(h[k]).group(0)!r}'
    if h['pool']==10: h['select'],h['select_rule']='likely_true',''
    if h['select']!='custom': h['select_rule']=''
    if h['select']=='custom' and not h['select_rule']: return None,'custom select without select_rule'
    return h,None
def hid(h): return hashlib.sha1(json.dumps(h,sort_keys=True).encode()).hexdigest()[:10]
NEG=("Before proposing, list 8 assumptions that practitioners usually hold about how these edits interact with the parent configuration. Negate each and ask what "
     "conditional effect would follow if it were false; build your hypotheses from these negated assumptions.")
QRY={'in_domain':"Write 3 literature search queries about established knowledge in this benchmark's own field that would help predict such conditional effects.",
     'cross_domain':("Write 3 literature search queries, each in a DIFFERENT field outside machine learning (for example network flow, electrical circuits, ecology, "
                     "hydrology, epidemiology, economics), about mechanisms whose structure resembles these conditional interactions between components.")}
SEL={'likely_true':"the 10 most likely to be true",
     'prior_avoid':"the 10 that other strong language models would be UNLIKELY to propose, but that could still be true (avoid the obvious, consensus ones)"}
def show(h): return json.dumps({'hypotheses':h})
def run(h,sp,rep):
    S=SP[sp]; tag=f'{sp}_{hid(h)}_r{rep}{SUF}'; f=os.path.join(OUT,f'run_{tag}.json')
    if os.path.exists(f): return json.load(open(f))
    n=h['pool']; tok=0; ok=lambda k: (lambda v: isinstance(v,list) and len(v)>=max(5,k//2))
    base=S['setup']+f"\n(Independent attempt #{rep+1}.)\n\n"
    strat=("Approach to follow:\n"+h['strategy']+"\n\n") if h['strategy'] else ''
    def go(name,prompt,key='hypotheses',check=None):
        nonlocal tok
        r=step(f'{tag}_{name}',GEN,prompt,key,check or ok(n)); tok+=r.get('tok',0); return r
    lit=''
    if h['retrieval']!='none':
        q=go('query',base+QRY[h['retrieval']]+'\nEnd with JSON only: {"queries": ["...", "...", "..."]}','queries',lambda v: isinstance(v,list) and len(v)>=1)
        if not q['ok']: return None
        res=[f"- {x['title']}: {x['abstract'][:400]}" for s in q['v'][:3] for x in search(str(s),4)]
        lit="Literature found by your searches (titles and abstracts):\n"+"\n".join(res)+"\n\n"
    neg=NEG+"\n\n" if h['challenge']=='negate' else ''
    if h['roles']=='single':
        r=go('gen',base+lit+neg+strat+f"Propose {n} conditional hypotheses about this benchmark that you believe are true.\n"+S['fmt'](n))
    else:
        g=go('gap',base+(lit if h['retrieval']=='in_domain' else '')+("You are the gap finder. Describe what conventional knowledge in this field predicts about these "
             "conditional effects, and identify 4 kinds of conditional effects that this conventional knowledge does NOT settle. Be concrete.")+
             '\nEnd with JSON only: {"gaps": ["...", ...]}','gaps',lambda v: isinstance(v,list) and len(v)>=2)
        if not g['ok']: return None
        r=go('gen',base+(lit if h['retrieval']=='cross_domain' else '')+"You are the innovator. The gap finder identified these open questions:\n"+
             "\n".join(f"- {x}" for x in g['v'])+"\n\n"+neg+strat+f"Propose {n} conditional hypotheses that address these gaps and that you believe could be true.\n"+S['fmt'](n))
    if not r['ok']: return None
    hy=r['v']
    if h['roles']=='gap_innovator_critic':
        r=go('critic',base+"You are the critic. Review these hypotheses proposed by a colleague:\n"+show(hy)+
             f"\n\nFor each, judge whether it is plausible and non-trivial; drop weak ones and replace them with better ones. Output the final {n}.\n"+S['fmt'](n))
        if not r['ok']: return None
        hy=r['v']
    if h['challenge']=='flaw':
        r=go('flaw',base+"Your hypotheses were:\n"+show(hy)+"\n\nNow challenge them. For each, state the strongest flaw (a reason it could be false, a confound, or why the "
             f"condition should not matter), then keep, revise or replace it. Output your final {n}.\n"+S['fmt'](n))
        if not r['ok']: return None
        hy=r['v']
    if n>10:
        crit=SEL.get(h['select']) or ("the 10 that best satisfy this criterion: "+h['select_rule'])
        r=go('select',base+"Candidate hypotheses:\n"+show(hy)+f"\n\nSelect {crit}. Copy them exactly.\n"+S['fmt'](10),check=ok(10))
        if not r['ok']: return None
        hy=r['v']
    rec=dict(h=h,sp=sp,rep=rep,hyps=hy[:10],tok=tok); json.dump(rec,open(f,'w')); return rec
def judge(rec):
    S=SP[rec['sp']]
    p=(S['setup']+"\n\nA research assistant proposed these hypotheses:\n"+show(rec['hyps'])+
       "\n\nRate this set as a reviewer would, without running anything: novelty (1-10: would an expert find these surprising and non-obvious?) and "
       "quality (1-10: soundness, specificity, plausibility). End with JSON only: {\"score\": {\"novelty\": <1-10>, \"quality\": <1-10>}}")
    r=step(f"judge_{rec['sp']}_{hid(rec['h'])}_r{rec['rep']}{SUF}",JUDGE,p,'score',lambda v: isinstance(v,dict) and 'novelty' in v and 'quality' in v)
    return r['v'] if r['ok'] else None
def score(rec,with_judge=False):
    S=SP[rec['sp']]; on=list(dict.fromkeys(c for c in map(S['claim'],rec['hyps']) if c))
    val=[c for c in on if S['valid'](c)]; sv=[c for c in val if S['prior'](c)<50]
    d=dict(on=len(on),valid=len(val),sv=len(sv),prec=len(val)/max(len(on),1),prior=float(np.nanmean([S['prior'](c) for c in on])) if on else np.nan,
           tok=rec['tok'],claims=[list(c) for c in on])
    if with_judge:
        j=judge(rec); d['judge']=float(j['novelty'])+float(j['quality']) if j else np.nan; d['nov']=j and j['novelty']; d['qual']=j and j['quality']
    return d

# --- optimizer: identical for all arms except the objective ---
OBJ={'J':"an expert reviewer's rating of the set of 10 hypotheses (novelty 1-10 plus quality 1-10, so 2-20), given without running any test",
     'V':"the number of the 10 hypotheses that turn out TRUE when tested against measured results (0-10)",
     'SV':("the number of the 10 hypotheses that turn out TRUE when tested against measured results AND that other strong language models rate as less than "
           "50% likely to hold (true surprises; 0-10)")}
OBJ['SV2']=OBJ['SV']  # stage 3: SV arm rerun with the fixed invariant and a 12-run baseline
FIT={'J':'judge','V':'valid','SV':'sv','SV2':'sv'}
FIELDS=("Harness fields:\n"
 "- retrieval: none | in_domain | cross_domain. The generator writes 3 literature queries; titles and abstracts of 12 results are shown to it. in_domain searches the "
 "benchmark's own field; cross_domain searches 3 fields outside machine learning.\n"
 "- roles: single | gap_innovator | gap_innovator_critic. A gap finder first lists what conventional knowledge does not settle; an innovator proposes; a critic drops "
 "weak hypotheses and replaces them.\n"
 "- challenge: none | flaw (after proposing, find the strongest flaw in each and keep, revise or replace it) | negate (list and negate practitioners' default "
 "assumptions before proposing).\n"
 "- pool: 10 | 20 | 30. How many to propose before selecting 10 (10 means no selection step).\n"
 "- select: likely_true | prior_avoid (those other models would be unlikely to propose but could be true) | custom. Used only when pool > 10.\n"
 "- select_rule: at most 50 words, the selection criterion when select = custom.\n"
 "- strategy: at most 150 words of instructions inserted into the generator's (or innovator's) prompt.\n"
 "Approximate cost per run: single generator about 5k tokens; each extra role, challenge, selection or retrieval step adds 5-10k. Mean cost must stay at or below "
 f"{CAP//1000}k tokens per run or the harness is disqualified.")
def evaluate(h,arm,reps,sp='nb'):
    with cf.ThreadPoolExecutor(6) as ex: recs=list(ex.map(lambda r: run(h,sp,r),reps))
    recs=[r for r in recs if r]
    with cf.ThreadPoolExecutor(6) as ex: S=list(ex.map(lambda r: score(r,with_judge=(arm=='J')),recs))
    return S
def summ(S,arm): return dict(fit=float(np.nanmean([s[FIT[arm]] for s in S])) if S else -1,tok=float(np.mean([s['tok'] for s in S])) if S else 0,n=len(S))
def proposer(arm,rnd,hist,inc,sample):
    p=("You are optimizing a HARNESS: a domain-general procedure by which a language model (the generator) proposes 10 testable conditional hypotheses about an "
       "empirical benchmark whose data it cannot see. A hypothesis says that the effect of a one-field edit is HIGHER or LOWER when the parent configuration "
       "satisfies a condition. The same harness will later be deployed on other benchmarks, so its text fields must be domain-general: they must not mention "
       "benchmark-specific names, components, datasets or metrics (such harnesses are rejected).\n\n"+FIELDS+
       f"\n\nScore of one run: {OBJ[arm]}. Each harness is run {M} times; its score is the mean.\n\n"
       "History (every harness tried so far, with mean score, mean tokens and status):\n"+
       "\n".join(json.dumps(x) for x in hist)+
       "\n\nCurrent best harness:\n"+json.dumps(inc)+"\nOne of its outputs (unlabelled):\n"+show(sample)+
       f"\n\nPropose {NCAND} new candidate harnesses that you expect to score higher than the current best. Make them differ from each other and from those already "
       "tried. Give each a one-sentence rationale.\nEnd with JSON only: {\"candidates\": [{\"retrieval\": ..., \"roles\": ..., \"challenge\": ..., \"pool\": ..., "
       "\"select\": ..., \"select_rule\": \"...\", \"strategy\": \"...\", \"rationale\": \"...\"}, ...]}")
    return p if rnd is None else step(f'prop_{arm}_r{rnd}',PROP,p,'candidates',lambda v: isinstance(v,list) and len(v)>=1)
def train(arm):
    f=os.path.join(OUT,f'train_{arm}.json'); inc=dict(BASE); S=evaluate(inc,arm,range(2*M if arm=='SV2' else M)); best=summ(S,arm); sample=run(inc,'nb',0)['hyps']
    hist=[dict(round=0,harness=inc,score=round(best['fit'],2),tokens=int(best['tok']),status='start')]; log=[dict(hist[0],n=best['n'])]
    print(arm,'base',best,flush=True)
    for rnd in range(ROUNDS):
        pr=proposer(arm,rnd,hist,inc,sample); cands=[]
        for c in (pr['v'] or [])[:NCAND]:
            h,err=clean(c if isinstance(c,dict) else {})
            if err: log.append(dict(round=rnd+1,harness=c,status='rejected: '+err)); hist.append(dict(round=rnd+1,harness=c,status='rejected: '+err)); continue
            cands.append((h,(c or {}).get('rationale','')))
        res=[(h,why,summ(evaluate(h,arm,range(M)),arm)) for h,why in cands]
        ok=[x for x in res if x[2]['tok']<=CAP and x[2]['n']>=M//2]
        top=max(ok,key=lambda x: x[2]['fit']) if ok else None
        for h,why,s in res:
            st='accepted' if (top and h is top[0] and s['fit']>best['fit']) else ('over token cap' if s['tok']>CAP else 'not better')
            hist.append(dict(round=rnd+1,harness=h,score=round(s['fit'],2),tokens=int(s['tok']),status=st)); log.append(dict(hist[-1],n=s['n'],rationale=why))
            print(arm,rnd+1,st,s,flush=True)
        if top and top[2]['fit']>best['fit']:
            inc,best=top[0],top[2]; sample=run(inc,'nb',0)['hyps']
        json.dump(dict(arm=arm,final=inc,final_score=best,log=log),open(f,'w'),indent=1)
    return inc
def final():
    H={'BASE':BASE}
    for arm in ('J','V','SV'): H['final_'+arm]=json.load(open(os.path.join(OUT,f'train_{arm}.json')))['final']
    out={}
    for sp in SP:
        for name,h in H.items():
            with cf.ThreadPoolExecutor(6) as ex: recs=[r for r in ex.map(lambda r: run(h,sp,r),FINAL_REPS) if r]
            with cf.ThreadPoolExecutor(6) as ex: out[f'{sp}|{name}']=dict(harness=h,runs=list(ex.map(lambda r: score(r,True),recs)))
            print('final',sp,name,len(recs),'runs',flush=True)
    json.dump(out,open(os.path.join(OUT,'final.json'),'w'),indent=1,default=float)
def fresh(arm):  # stage 3 E1: fresh NB201 runs of one arm's final harness, judged
    h=json.load(open(os.path.join(OUT,f'train_{arm}.json')))['final']
    with cf.ThreadPoolExecutor(6) as ex: recs=[r for r in ex.map(lambda r: run(h,'nb',r),FINAL_REPS) if r]
    with cf.ThreadPoolExecutor(6) as ex: S=list(ex.map(lambda r: score(r,True),recs))
    json.dump(dict(harness=h,runs=S),open(os.path.join(OUT,f'fresh_{arm}.json'),'w'),indent=1,default=float); print('fresh',arm,len(S),flush=True)
def frontier_set():  # stage 3 E2: harnesses fixed before any Opus run (PREREG_s3.md)
    T=lambda a: json.load(open(os.path.join(OUT,f'train_{a}.json')))
    L=T('SV')['log']; pick=lambda f: next(x['harness'] for x in L if isinstance(x.get('harness'),dict) and x.get('score') is not None and f(x['harness']))
    return {'BASE':BASE,'final_J':T('J')['final'],'final_V':T('V')['final'],
            'SV_prior_avoid':pick(lambda h: h['roles']=='single' and h['pool']==20 and h['select']=='prior_avoid' and h['retrieval']=='none'),
            'SV_cross_domain':pick(lambda h: h['retrieval']=='cross_domain')}
def frontier():
    out={}
    for name,h in frontier_set().items():
        with cf.ThreadPoolExecutor(6) as ex: recs=[r for r in ex.map(lambda r: run(h,'nb',r),FINAL_REPS) if r]
        out[name]=dict(harness=h,runs=[score(r) for r in recs]); print('frontier',GEN,name,len(recs),flush=True)
    json.dump(out,open(os.path.join(OUT,f'frontier_{GEN}.json'),'w'),indent=1,default=float)
if __name__=='__main__':
    m=sys.argv[1] if len(sys.argv)>1 else 'show'
    if m=='show':
        print(proposer('SV',None,[dict(round=0,harness=BASE,score=0.96,tokens=5000,status='start')],BASE,[{'example':'...'}]))
        bad=dict(BASE,strategy='prefer skip connections on edge 3'); print('\ninvariant check:',clean(bad)[1]); print('NB valid cells',sum(s1.truth(c)!=0 for c in s1.H))
    if m=='train':
        arms=sys.argv[2:] or list(FIT); evaluate(dict(BASE),'J',range(M))  # shared start, run once before the arms fork
        with cf.ThreadPoolExecutor(len(arms)) as ex: list(ex.map(train,arms))
    if m=='final': final()
    if m=='fresh': fresh(sys.argv[2])
    if m=='frontier': frontier()
