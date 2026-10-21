"""Stand-in for an LLM client in tests: reads the prompt on stdin and answers with random but well-formed JSON."""
import json, random, re, sys
p = sys.stdin.read()
rng = random.Random(len(p))
params = {}
for m in re.finditer(r'^- (\w+): (.+)$', p, re.M):
    vals = re.sub(r' \(condition levels:.*\)$', '', m.group(2)).split(', ')
    levels = re.findall(r'(\w+) = ', m.group(2).split('(condition levels:')[1]) if 'condition levels' in m.group(2) else vals
    params[m.group(1)] = (vals, levels)
names = list(params)
if '"experiments"' in p:
    k = int(re.search(r'with exactly (\d+) experiments', p).group(1))
    ex = []
    for _ in range(k):
        j = rng.choice(names)
        ex.append({'config': {n: rng.choice(v[0]) for n, v in params.items()}, 'set_param': j, 'set_value': rng.choice(params[j][0])})
    print('thinking...\n' + json.dumps({'experiments': ex}))
elif '"ratings"' in p:
    n = len(re.findall(r'^\d+\. Setting', p, re.M))
    print(json.dumps({'ratings': [{'id': i + 1, 'p': rng.randrange(101)} for i in range(n)]}))
else:
    n = int(re.search(r'with exactly (\d+) distinct', p).group(1))
    hs = []
    for _ in range(n):
        j, k = rng.sample(names, 2)
        hs.append({'set_param': j, 'set_value': rng.choice(params[j][0]), 'cond_param': k, 'cond_level': rng.choice(params[k][1]),
                   'direction': rng.choice(['more', 'less']), 'why': 'test'})
    print(json.dumps({'hypotheses': hs}))
