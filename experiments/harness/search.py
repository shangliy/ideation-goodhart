# Retrieval tool for ideation harnesses: literature search with a benchmark blocklist, cached per query and logged for audit.
# Usage: python3 search.py "query" [--limit 8]
import sys,os,re,json,hashlib,subprocess,argparse,time
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'search_cache'); LOG=os.path.join(HERE,'search_log.jsonl')
LIT=os.environ.get('LITSEARCH',os.path.expanduser('~/.claude/skills/paper/scripts/litsearch.py'))  # arXiv/OpenAlex/S2 retriever; only needed for uncached queries
BLOCK=re.compile(r'nas[- ]?bench|nats[- ]?bench|\bnb[- ]?201\b|hw[- ]?nas[- ]?bench|tabular (nas )?benchmark',re.I)  # papers that analyse the training-space table
def search(q,limit=8):
    os.makedirs(CACHE,exist_ok=True); f=os.path.join(CACHE,hashlib.sha1(f'{q}|{limit}'.encode()).hexdigest()+'.json')
    if os.path.exists(f): R=json.load(open(f))
    else:
        ws=os.path.join(CACHE,'ws'); os.makedirs(ws,exist_ok=True)
        out=subprocess.run(['python3',LIT,'search',q,'--limit',str(limit*2),'--abstracts','--workspace',ws],capture_output=True,text=True,timeout=180).stdout
        R=[]
        for m in re.finditer(r'^\s{2}\[(\S+)\] (.*?)\n\s{6}(.*?)\n\s{6}(\S+)\n\s{6}(.*?)$',out,re.M):
            R.append(dict(key=m[1],meta=m[2],title=m[3],url=m[4],abstract=m[5]))
        json.dump(R,open(f,'w'))
    keep=[r for r in R if not BLOCK.search(r['title']+' '+r['abstract'])][:limit]
    with open(LOG,'a') as g: g.write(json.dumps(dict(t=time.time(),q=q,n=len(R),blocked=len(R)-len([r for r in R if not BLOCK.search(r['title']+' '+r['abstract'])]),kept=[r['title'] for r in keep]))+'\n')
    return keep
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('q'); a.add_argument('--limit',type=int,default=8); x=a.parse_args()
    for r in search(x.q,x.limit): print(f"- {r['title']} ({r['meta']})\n  {r['abstract'][:400]}\n")
