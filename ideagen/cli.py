"""ideagen: generate testable research hypotheses about an experiment grid, the way the ideation-goodhart study found works best.

  ideagen demo                                   synthetic grid, direct vs measure, scored on held-out runs
  ideagen table runs.csv --metric val_acc        your finished runs: probes answered from one half, claims checked on the other
  ideagen run --space space.json --run-cmd CMD   your training command: probes are real runs; --verify-n re-tests each claim
"""
import argparse, json, os, sys
from . import demo as demo_mod
from .generate import direct, measure, prior
from .llm import LLM, DryRun
from .runner import Command, load_space, verify
from .space import Table, load_table


def common(ap):
    ap.add_argument('--mode', choices=['measure', 'direct', 'both'], default='measure', help='measure = design and read experiments first (default)')
    ap.add_argument('--probes', type=int, default=100, help='experiments the model designs before proposing (default 100)')
    ap.add_argument('--n', type=int, default=10, help='hypotheses to propose (default 10)')
    ap.add_argument('--model', default='sonnet', help='generator model passed to the LLM command (default sonnet)')
    ap.add_argument('--prior', default='', help='comma-separated models that rate each hypothesis blind; below 50 = unexpected')
    ap.add_argument('--llm-cmd', help='LLM command template, prompt on stdin, {model} substituted (default: claude -p ...; env IDEAGEN_LLM_CMD)')
    ap.add_argument('--describe', default='', help='what the experiments are (the model sees this; more context gives better hypotheses)')
    ap.add_argument('--out', default='ideagen_out', help='output directory')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', help='print the first prompt and stop, no LLM calls')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ideagen', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('demo', help='run on a synthetic grid (3 LLM calls with --mode both)'); common(d)
    d.set_defaults(mode='both')
    t = sub.add_parser('table', help='a CSV of finished runs, one row per run'); common(t)
    t.add_argument('csv'); t.add_argument('--metric', required=True)
    t.add_argument('--params', help='comma-separated parameter columns (default: every column except metric, group and --ignore)')
    t.add_argument('--group', help='column whose values (e.g. dataset) a hypothesis must hold on separately')
    t.add_argument('--ignore', default='seed', help='comma-separated columns that are not parameters (default: seed)')
    t.add_argument('--minimize', action='store_true')
    t.add_argument('--min-n', type=int, default=10, help='minimum runs per cell for a claim to be testable')
    t.add_argument('--tau', type=float, default=0.0, help='smallest interaction (metric units) that counts as real')
    r = sub.add_parser('run', help='propose with real experiments from your own command'); common(r)
    r.add_argument('--space', required=True, help='JSON: {"metric": ..., "minimize": false, "describe": ..., "params": {name: [values]}}')
    r.add_argument('--run-cmd', required=True, help='command that reads one config as JSON on stdin and prints the metric')
    r.add_argument('--jobs', type=int, default=1, help='parallel runs')
    r.add_argument('--verify-n', type=int, default=0, help='after proposing, re-test each hypothesis with 4*N fresh runs (0 = skip)')
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    llm = LLM(a.model, a.llm_cmd, os.path.join(a.out, 'cache'), dry_run=a.dry_run)
    panel = [LLM(m.strip(), a.llm_cmd, os.path.join(a.out, 'cache'), dry_run=a.dry_run) for m in a.prior.split(',') if m.strip()]

    table = None
    if a.cmd in ('demo', 'table'):
        if a.cmd == 'demo':
            path = demo_mod.write(os.path.join(a.out, 'demo.csv'))
            print(f'[ideagen] wrote synthetic grid {path}', file=sys.stderr)
            sp, X, y, g = load_table(path, 'recall_at_20', list(demo_mod.GRID), 'task', describe=a.describe or demo_mod.DESCRIBE)
            table = Table(sp, X, y, g, seed=a.seed)
        else:
            ign = {c.strip() for c in a.ignore.split(',') if c.strip()}
            params = [c.strip() for c in a.params.split(',')] if a.params else None
            sp, X, y, g = load_table(a.csv, a.metric, params, a.group, a.minimize, a.describe, ignore=ign)
            table = Table(sp, X, y, g, seed=a.seed, min_n=a.min_n, tau=a.tau)
        run_many = lambda probes, rng: [table.measure(c, j, b, rng) for c, j, b in probes]
    else:
        sp = load_space(a.space)
        sp.describe = a.describe or sp.describe
        cmdr = Command(sp, a.run_cmd, a.jobs)
        run_many = cmdr.run_many

    arms = ['direct', 'measure'] if a.mode == 'both' else [a.mode]
    results = []
    try:
        for arm in arms:
            res = direct(sp, llm, a.n) if arm == 'direct' else measure(sp, llm, run_many, a.probes, a.n, a.seed)
            prior(sp, panel, res['hypotheses'])
            if table:
                for h in res['hypotheses']:
                    h['verdict'] = table.verdict(h['claim'], h['direction'])
            elif a.verify_n:
                verify(sp, cmdr, res['hypotheses'], a.verify_n, seed=a.seed)
            results.append(res)
    except DryRun:
        print('[ideagen] dry run: stopped before the first LLM call', file=sys.stderr)
        return 0

    base = table.base_rate() if table else None
    report = render(sp, results, base, panel)
    open(os.path.join(a.out, 'hypotheses.md'), 'w').write(report)
    json.dump(dict(metric=sp.metric, groups=sp.groups, base_rate=base, results=results), open(os.path.join(a.out, 'run.json'), 'w'), indent=1, default=str)
    print(report)
    print(f'[ideagen] {llm.calls + sum(m.calls for m in panel)} LLM calls; wrote {a.out}/hypotheses.md and run.json', file=sys.stderr)
    return 0


def summary(hs, panel):
    tested = [h for h in hs if h.get('verdict', '').split(' ')[0] in ('supported', 'contradicted', 'not', 'inconclusive')]
    sup = [h for h in hs if h.get('verdict') == 'supported']
    s = f'{len(hs)} hypotheses'
    if tested:
        s += f', {len(sup)} supported of {len(tested)} testable (precision {len(sup) / len(tested):.2f})'
    if panel:
        s += f', {sum(1 for h in sup if h.get("prior") is not None and h["prior"] < 50)} supported and unexpected (panel prior < 50)'
    return s


def render(sp, results, base, panel):
    out = ['# ideagen hypotheses', '']
    if base:
        out += [f'Held-out check: a hypothesis is supported when every group shows the stated sign with |interaction| >= 2 standard errors. '
                f'Guessing at random would be right {base[0]:.2f} of the time ({base[1]} of {base[2]} testable directional claims).', '']
    for res in results:
        hs = res['hypotheses']
        out += [f'## {res["arm"]}' + (f' ({len(res["probes"])} experiments first)' if res['probes'] else ''), '', summary(hs, panel), '',
                '| # | hypothesis | why (model) |' + (' prior |' if panel else '') + (' verdict |' if any('verdict' in h for h in hs) else ''),
                '|---|---|---|' + ('---|' if panel else '') + ('---|' if any('verdict' in h for h in hs) else '')]
        for i, h in enumerate(hs):
            out.append(f'| {i + 1} | {h["text"]} | {h["why"].replace("|", "/")} |' + (f' {h.get("prior")} |' if panel else '')
                       + (f' {h.get("verdict", "")} |' if any('verdict' in x for x in hs) else ''))
        out.append('')
    return '\n'.join(out)


if __name__ == '__main__':
    sys.exit(main())
