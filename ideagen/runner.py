"""Run experiments with the user's own command, and verify hypotheses with fresh paired runs.

The command gets one configuration as JSON on stdin, e.g. {"lr": 0.001, "model": "gru"}, and prints the metric:
either a bare number as its last output line or a JSON object with the metric's name as a key.
"""
import json, math, random, re, shlex, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from .space import Param, Space


def load_space(path):
    """A JSON spec: {"metric": "val_acc", "minimize": false, "describe": "...", "params": {"lr": [0.0001, 0.001], "model": ["mf", "gru"]}}"""
    d = json.load(open(path))
    params = [Param(k, v) for k, v in d['params'].items()]
    if any(len(p.values) < 2 for p in params):
        raise SystemExit('every parameter needs at least two values')
    return Space(params, d['metric'], bool(d.get('minimize')), d.get('describe', ''))


class Command:
    def __init__(self, sp, cmd, jobs=1, timeout=None, log=sys.stderr):
        self.sp, self.cmd, self.jobs, self.timeout, self.log, self.runs = sp, shlex.split(cmd), jobs, timeout, log, 0

    def score(self, cfg):
        conf = {p.name: p.values[i] for p, i in zip(self.sp.params, cfg)}
        r = subprocess.run(self.cmd, input=json.dumps(conf), capture_output=True, text=True, timeout=self.timeout)
        self.runs += 1
        if r.returncode != 0:
            print(f'[ideagen] run failed for {conf}: {r.stderr[-300:]}', file=self.log)
            return None
        out = r.stdout.strip()
        try:
            v = json.loads(out.splitlines()[-1])
            v = v[self.sp.metric] if isinstance(v, dict) else v
        except (json.JSONDecodeError, IndexError, KeyError):
            nums = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', out)
            v = nums[-1] if nums else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            print(f'[ideagen] no metric in output for {conf}: {out[-200:]}', file=self.log)
            return None
        return -v if self.sp.minimize else v

    def pairs(self, pairs):
        """Score each (parent, child) pair; return child - parent per pair (None when either run failed)."""
        flat = [c for p in pairs for c in p]
        with ThreadPoolExecutor(self.jobs) as ex:
            s = list(ex.map(self.score, flat))
        return [None if s[2 * i] is None or s[2 * i + 1] is None else s[2 * i + 1] - s[2 * i] for i in range(len(pairs))]

    def run_many(self, probes, rng):
        pairs = [(cfg, cfg[:j] + [b] + cfg[j + 1:]) for cfg, j, b in probes]
        return [[d] for d in self.pairs(pairs)]


def verify(sp, cmdr, hyps, n=8, z=2.0, seed=0, log=sys.stderr):
    """Pre-registered check per hypothesis: n random parents inside the condition and n outside, each edited; the
    interaction is the mean change inside minus outside. supported: stated sign with |I| >= z*se; contradicted: the
    opposite sign at the same bar; otherwise inconclusive. Costs 4n runs per hypothesis."""
    rng = random.Random(seed)
    print(f'[ideagen] verifying {len(hyps)} hypotheses with {4 * n * len(hyps)} runs; criterion: |interaction| >= {z} x se', file=log)
    for h in hyps:
        j, b, k, c = h['claim']
        inside = sp.band_set(k, c)
        pairs, where = [], []
        for side in (True, False):
            for _ in range(n):
                cfg = [rng.randrange(len(p.values)) for p in sp.params]
                cfg[j] = rng.choice([i for i in range(len(sp.params[j].values)) if i != b])
                cfg[k] = rng.choice([i for i in range(len(sp.params[k].values)) if (i in inside) == side])
                pairs.append((cfg, cfg[:j] + [b] + cfg[j + 1:])); where.append(side)
        d = cmdr.pairs(pairs)
        a = [x for x, w in zip(d, where) if w and x is not None]
        o = [x for x, w in zip(d, where) if not w and x is not None]
        if len(a) < 2 or len(o) < 2:
            h['verdict'] = 'inconclusive (runs failed)'; continue
        var = lambda v: sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)
        I = sum(a) / len(a) - sum(o) / len(o)
        se = math.sqrt(var(a) / len(a) + var(o) / len(o))
        h['interaction'] = round(I * (-1 if sp.minimize else 1), 6)
        h['verdict'] = 'supported' if h['direction'] * I >= z * se else 'contradicted' if -h['direction'] * I >= z * se else 'inconclusive'
