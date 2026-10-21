"""Hypothesis space over a grid of experiment settings.

A hypothesis is a conditional interaction claim:
  "setting <edit param> to <value> raises the metric MORE (or LESS) when <cond param> is in <level>
   than when it is not."
Its effect size is the interaction contrast
  I = [m(edit, cond) - m(not edit, cond)] - [m(edit, not cond) - m(not edit, not cond)]
on a score where higher is better (the metric is negated when it should be minimized).
"""
import csv, itertools, math, random
import numpy as np


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def fmtv(v):
    return f'{v:g}' if isinstance(v, float) else str(v)


class Param:
    """One experiment parameter with ordered levels. `values` are what a run receives; `labels` are what the model reads."""

    def __init__(self, name, values, labels=None, numeric=None):
        self.name = name
        self.values = list(values)
        self.labels = labels or [fmtv(v) for v in self.values]
        self.numeric = all(isinstance(v, (int, float)) for v in self.values) if numeric is None else numeric

    def bands(self):
        """Condition levels: every value when there are at most 4, otherwise low / mid / high thirds of the ordered values."""
        L = len(self.values)
        if L <= 4 or not self.numeric:
            return [(lab, frozenset([i])) for i, lab in enumerate(self.labels)]
        a = L // 3
        return [('low', frozenset(range(a))), ('mid', frozenset(range(a, L - a))), ('high', frozenset(range(L - a, L)))]

    def band_text(self):
        bs = self.bands()
        if all(len(s) == 1 for _, s in bs):
            return ''
        return ' (condition levels: ' + '; '.join(f"{n} = {', '.join(self.labels[i] for i in sorted(s))}" for n, s in bs) + ')'

    def index(self, v):
        """Map a value or label the model wrote to a level index (exact label first, then nearest number, log scale when positive)."""
        s = str(v).strip()
        if s in self.labels:
            return self.labels.index(s)
        x = _num(s)
        if x is None or not self.numeric:
            low = [l.lower() for l in self.labels]
            return low.index(s.lower()) if s.lower() in low else None
        vals = [float(u) for u in self.values]
        if x > 0 and all(u > 0 for u in vals):
            return int(np.argmin([abs(math.log(u) - math.log(x)) for u in vals]))
        return int(np.argmin([abs(u - x) for u in vals]))


class Space:
    def __init__(self, params, metric, minimize=False, describe='', groups=('all',)):
        self.params, self.metric, self.minimize, self.describe, self.groups = params, metric, minimize, describe, list(groups)
        self.P = {p.name: i for i, p in enumerate(params)}

    def claims(self):
        """All (edit param, edit level, cond param, cond band name) cells; each gives two directional claims."""
        for j, pj in enumerate(self.params):
            for b in range(len(pj.values)):
                for k, pk in enumerate(self.params):
                    if k == j:
                        continue
                    for c, _ in pk.bands():
                        yield (j, b, k, c)

    def band_set(self, k, c):
        return dict(self.params[k].bands())[c]

    def sentence(self, cl, direction):
        j, b, k, c = cl
        pj, pk = self.params[j], self.params[k]
        cond = f'{pk.name} = {c}' if len(self.band_set(k, c)) == 1 else f'{pk.name} is {c}'
        verb = 'lowers' if self.minimize else 'raises'
        return f'Setting {pj.name} = {pj.labels[b]} {verb} {self.metric} {"more" if direction > 0 else "less"} when {cond} than otherwise'

    def config_text(self, idx):
        return ', '.join(f'{p.name}={p.labels[i]}' for p, i in zip(self.params, idx))


def load_table(path, metric, params=None, group=None, minimize=False, describe='', ignore=(), max_levels=12, bins=5):
    """Read a CSV of finished runs (one row per run). Numeric parameters with more than `max_levels` distinct values are binned into quantiles."""
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if _num(r.get(metric)) is not None]
    if not rows:
        raise SystemExit(f'no rows with a numeric "{metric}" column in {path}')
    cols = list(rows[0].keys())
    for c in [metric] + ([group] if group else []) + list(params or []):
        if c not in cols:
            raise SystemExit(f'column "{c}" not in {path}; columns are {cols}')
    params = params or [c for c in cols if c not in (metric, group) and c not in ignore]
    P, X = [], []
    for name in params:
        raw = [r[name] for r in rows]
        uniq = sorted(set(raw), key=lambda v: (_num(v) is None, _num(v) if _num(v) is not None else 0, v))
        numeric = all(_num(v) is not None for v in uniq)
        if len(uniq) < 2:
            continue
        if numeric and len(uniq) > max_levels:
            x = np.array([_num(v) for v in raw])
            edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
            lev = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)
            meds = [float(np.median(x[lev == i])) for i in range(len(edges) - 1)]
            labels = [f'{edges[i]:.3g}..{edges[i + 1]:.3g}' for i in range(len(edges) - 1)]
            P.append(Param(name, meds, labels, numeric=True)); X.append(lev.tolist())
        elif len(uniq) > max_levels:
            raise SystemExit(f'parameter "{name}" has {len(uniq)} categories (max {max_levels}); drop it with --params or merge categories')
        else:
            vals = [_num(v) if numeric else v for v in uniq]
            vals = [int(v) if numeric and float(v).is_integer() and all(float(u).is_integer() for u in vals) else v for v in vals]
            P.append(Param(name, vals, [fmtv(v) for v in vals], numeric=numeric)); X.append([uniq.index(v) for v in raw])
    y = np.array([_num(r[metric]) for r in rows]) * (-1 if minimize else 1)
    g = [r[group] for r in rows] if group else ['all'] * len(rows)
    sp = Space(P, metric, minimize, describe, groups=sorted(set(g)))
    return sp, np.array(X, dtype=int).T, y, np.array(g)


class Table:
    """Finished runs split in two halves: `explore` answers the model's probes, `holdout` decides which claims are true."""

    def __init__(self, space, X, y, g, seed=0, min_n=10, z=2.0, tau=0.0):
        self.sp, self.min_n, self.z, self.tau = space, min_n, z, tau
        idx = np.random.default_rng(seed).permutation(len(y))
        half = len(y) // 2
        self.explore = (X[idx[:half]], y[idx[:half]], g[idx[:half]])
        self.holdout = (X[idx[half:]], y[idx[half:]], g[idx[half:]])
        self._lookup = {}
        Xe, ye, ge = self.explore
        for row, v, gg in zip(map(tuple, Xe.tolist()), ye.tolist(), ge.tolist()):
            self._lookup.setdefault((gg, row), []).append(v)

    def interaction(self, cl, split='holdout'):
        """Per-group interaction contrast and its standard error; None for a group whose four cells are too thin."""
        X, y, g = getattr(self, split)
        j, b, k, c = cl
        e, cond = X[:, j] == b, np.isin(X[:, k], list(self.sp.band_set(k, c)))
        out = []
        for gg in self.sp.groups:
            m = g == gg
            cells = [y[m & e & cond], y[m & ~e & cond], y[m & e & ~cond], y[m & ~e & ~cond]]
            if min(len(s) for s in cells) < self.min_n:
                out.append(None); continue
            I = cells[0].mean() - cells[1].mean() - cells[2].mean() + cells[3].mean()
            se = math.sqrt(sum(s.var(ddof=1) / len(s) for s in cells))
            out.append((float(I), se))
        return out

    def verdict(self, cl, direction):
        """'supported' when every group shows the stated sign with |I| >= max(tau, z*se); 'contradicted' for the opposite sign; else 'not supported'."""
        st = self.interaction(cl)
        if any(s is None for s in st):
            return 'untestable'
        if all(direction * I >= max(self.tau, self.z * se) for I, se in st):
            return 'supported'
        if all(-direction * I >= max(self.tau, self.z * se) for I, se in st):
            return 'contradicted'
        return 'not supported'

    def base_rate(self):
        """Share of testable directional claims that are supported: the precision of guessing at random."""
        n = v = 0
        for cl in self.sp.claims():
            for d in (1, -1):
                r = self.verdict(cl, d)
                if r != 'untestable':
                    n += 1; v += r == 'supported'
        return v / max(n, 1), v, n

    def measure(self, parent, j, b, rng):
        """One noisy experiment per group: a single run of the parent and of the edited child, drawn from the explore half."""
        child = list(parent); child[j] = b
        out = []
        for gg in self.sp.groups:
            a, c = self._lookup.get((gg, tuple(parent))), self._lookup.get((gg, tuple(child)))
            out.append(None if not a or not c else rng.choice(c) - rng.choice(a))
        return out
