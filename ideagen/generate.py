"""Two ways to generate hypotheses, as compared in the ideation-goodhart study.

direct   one call: propose N conditional hypotheses you believe are true.
measure  the model first designs K single-edit experiments, sees their (noisy) results, then proposes N hypotheses.
         In the study this raised true-and-surprising hypotheses in 3 of 4 model x space cells; internal-reasoning
         harnesses (retrieval, roles, self-challenge, strategy text) did not, and optimizing for an LLM judge cut precision.
"""
import random
from .llm import last_json


def setup(sp):
    multi = len(sp.groups) > 1
    lines = [sp.describe.strip() or 'A grid of experiment settings.',
             f"Metric: {sp.metric} ({'lower' if sp.minimize else 'higher'} is better)"
             + (f', measured separately on {len(sp.groups)} groups: {", ".join(sp.groups)}.' if multi else '.'),
             'Parameters and their values:',
             *[f'- {p.name}: {", ".join(p.labels)}{p.band_text()}' for p in sp.params],
             'A hypothesis states that setting parameter j to value b improves the metric MORE (direction "more") or LESS (direction "less") '
             'when parameter k is at condition level c than when it is not. It is about the interaction, not about the average effect of the edit'
             + (', and it must hold on every group.' if multi else '.')]
    return '\n'.join(lines)


def fmt(sp, n):
    names = ' | '.join(p.name for p in sp.params)
    return ('End with JSON only: {"hypotheses": [{"set_param": "<' + names + '>", "set_value": "<one of its values>", '
            '"cond_param": "<' + names + '>", "cond_level": "<a condition level of cond_param>", "direction": "more" or "less", '
            f'"why": "<one sentence>"}}, ...]}} with exactly {n} distinct hypotheses.')


def parse_hypotheses(sp, items):
    out, seen = [], set()
    for x in items or []:
        try:
            j, k = sp.P[str(x['set_param']).strip()], sp.P[str(x['cond_param']).strip()]
            b = sp.params[j].index(x['set_value'])
            d = {'more': 1, 'less': -1}[str(x['direction']).strip().lower()]
        except (KeyError, TypeError, AttributeError):
            continue
        if b is None or k == j:
            continue
        c = band_name(sp.params[k], x['cond_level'])
        if c is None or (j, b, k, c, d) in seen:
            continue
        seen.add((j, b, k, c, d))
        out.append(dict(claim=(j, b, k, c), direction=d, text=sp.sentence((j, b, k, c), d), why=str(x.get('why', ''))))
    return out


def band_name(p, v):
    s = str(v).strip().lower()
    for name, _ in p.bands():
        if name.lower() == s:
            return name
    i = p.index(v)  # a value inside a band, e.g. "0.001" for lr when lr has low/mid/high levels
    return next((name for name, members in p.bands() if i in members), None) if i is not None else None


def direct(sp, llm, n=10, attempt=0):
    prompt = setup(sp) + tag(attempt) + \
        f'\n\nPropose {n} conditional hypotheses about this experiment grid that you believe are true.\n' + fmt(sp, n)
    hyps = parse_hypotheses(sp, last_json(llm.ask(prompt, 'direct'), 'hypotheses'))
    return dict(arm='direct', hypotheses=hyps, probes=[])


def tag(attempt):
    return f'\n(Independent attempt #{attempt + 1}.)' if attempt else ''


def design_prompt(sp, k, n, attempt=0):
    ex = ', '.join(f'"{p.name}": <value>' for p in sp.params)
    groups = f' on each of the {len(sp.groups)} groups' if len(sp.groups) > 1 else ''
    return (setup(sp) + tag(attempt) + f'\n\nYou may run {k} experiments before proposing hypotheses. An experiment applies one edit (set one parameter to a new value) '
            f'to one configuration you choose, runs both configurations once (a single noisy run each) and returns the change in {sp.metric}{groups}. '
            f'Design the experiments that will best reveal conditional effects; you will see the results and then propose {n} hypotheses.\n'
            'End with JSON only: {"experiments": [{"config": {' + ex + '}, "set_param": "<parameter>", "set_value": "<new value>"}, ...]} '
            f'with exactly {k} experiments.')


def parse_probes(sp, items):
    out = []
    for x in items or []:
        try:
            cfg = [p.index(x['config'][p.name]) for p in sp.params]
            j = sp.P[str(x['set_param']).strip()]
            b = sp.params[j].index(x['set_value'])
        except (KeyError, TypeError, AttributeError):
            continue
        if None in cfg or b is None or b == cfg[j]:
            continue
        out.append((cfg, j, b))
    return out


def results_text(sp, probes):
    sign = -1 if sp.minimize else 1
    lines = []
    for i, (cfg, j, b, diffs) in enumerate(probes):
        if all(d is None for d in diffs):
            res = 'no result (configuration not available)'
        elif len(sp.groups) == 1:
            res = f'{sign * diffs[0]:+.4g}'
        else:
            res = ', '.join(f'{g} ' + ('n/a' if d is None else f'{sign * d:+.4g}') for g, d in zip(sp.groups, diffs))
        lines.append(f'{i + 1}. config ({sp.config_text(cfg)}); set {sp.params[j].name} = {sp.params[j].labels[b]}; change in {sp.metric}: {res}')
    return '\n'.join(lines)


def measure(sp, llm, run_many, k=100, n=10, seed=0, attempt=0):
    """`run_many(probes, rng)` executes the (config, param, value) experiments and returns each one's per-group score change (None if unavailable)."""
    probes = parse_probes(sp, last_json(llm.ask(design_prompt(sp, k, n, attempt), 'design'), 'experiments'))[:k]
    if not probes:
        raise RuntimeError('the model designed no usable experiments; see the cached answer in the cache directory')
    done = [(cfg, j, b, d) for (cfg, j, b), d in zip(probes, run_many(probes, random.Random(seed)))]
    prompt = (setup(sp) + tag(attempt) + f'\n\nBefore proposing, you ran {len(done)} experiments. Each applied one edit to one configuration, ran both once '
              f'(single noisy runs) and measured the change in {sp.metric}; these are real measurements and can serve as evidence:\n'
              + results_text(sp, done) + f'\n\nPropose {n} conditional hypotheses about this experiment grid that you believe are true.\n' + fmt(sp, n))
    hyps = parse_hypotheses(sp, last_json(llm.ask(prompt, 'propose'), 'hypotheses'))
    return dict(arm='measure', hypotheses=hyps, probes=[dict(config=sp.config_text(c), edit=f'{sp.params[j].name}={sp.params[j].labels[b]}', change=d)
                                                         for c, j, b, d in done])


def prior(sp, llms, hyps):
    """Mean probability (0-100) that each hypothesis is true, rated blind by a panel. Below 50 = the panel did not expect it."""
    if not llms or not hyps:
        return
    listing = '\n'.join(f'{i + 1}. {h["text"]}' for i, h in enumerate(hyps))
    prompt = (setup(sp) + '\n\nWithout running anything, rate each hypothesis below by the probability (0-100) that it is true on this grid.\n'
              + listing + '\nEnd with JSON only: {"ratings": [{"id": 1, "p": <0-100>}, ...]} covering every hypothesis.')
    scores = [[] for _ in hyps]
    for m in llms:
        for r in last_json(m.ask(prompt, f'prior:{m.model}'), 'ratings') or []:
            try:
                i = int(r['id']) - 1
                if 0 <= i < len(hyps):
                    scores[i].append(float(r['p']))
            except (KeyError, TypeError, ValueError):
                pass
    for h, s in zip(hyps, scores):
        h['prior'] = round(sum(s) / len(s), 1) if s else None
