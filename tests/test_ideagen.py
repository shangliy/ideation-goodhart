"""End-to-end tests with a fake LLM; no API calls.  python3 -m unittest discover tests"""
import json, os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from ideagen.cli import main
from ideagen.llm import last_json
from ideagen import demo
from ideagen.space import Table, load_table

FAKE = f'{sys.executable} {os.path.join(HERE, "fake_llm.py")}'


class T(unittest.TestCase):
    def test_last_json(self):
        self.assertEqual(last_json('x {"a": [1, "}"]} y {"a": [2]}', 'a'), [2])
        self.assertIsNone(last_json('no json', 'a'))

    def test_demo_truth_finds_planted_effects(self):
        with tempfile.TemporaryDirectory() as d:
            sp, X, y, g = load_table(demo.write(os.path.join(d, 'g.csv')), 'recall_at_20', list(demo.GRID), 'task')
            t = Table(sp, X, y, g)
            P = sp.P
            mf256 = (P['emb_dim'], sp.params[P['emb_dim']].labels.index('256'), P['model'], 'mf')
            self.assertEqual(t.verdict(mf256, -1), 'supported')
            self.assertEqual(t.verdict(mf256, 1), 'contradicted')
            rate, v, n = t.base_rate()
            self.assertTrue(0.02 < rate < 0.5, rate)

    def test_demo_cli(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(main(['demo', '--llm-cmd', FAKE, '--out', d, '--prior', 'a,b', '--probes', '30', '--runs', '2']), 0)
            r = json.load(open(os.path.join(d, 'run.json')))
            self.assertEqual([x['arm'] for x in r['results']], ['direct', 'measure'] * 2)
            self.assertTrue(0 < len(r['results'][1]['probes']) <= 30)  # the fake sometimes repeats the current value, which is dropped
            self.assertTrue(all('verdict' in h and 'prior' in h for x in r['results'] for h in x['hypotheses']))

    def test_run_cmd_with_verify(self):
        with tempfile.TemporaryDirectory() as d:
            ex = os.path.join(ROOT, 'examples')
            self.assertEqual(main(['run', '--space', os.path.join(ex, 'space.json'), '--run-cmd', f'{sys.executable} {os.path.join(ex, "toy_train.py")}',
                                   '--llm-cmd', FAKE, '--out', d, '--probes', '6', '--n', '3', '--verify-n', '3', '--jobs', '4']), 0)
            r = json.load(open(os.path.join(d, 'run.json')))
            self.assertTrue(all(h['verdict'] in ('supported', 'contradicted', 'inconclusive') for h in r['results'][0]['hypotheses']))

    def test_dry_run_makes_no_calls(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(main(['demo', '--llm-cmd', 'false', '--out', d, '--dry-run']), 0)
            self.assertFalse(os.path.exists(os.path.join(d, 'run.json')))


if __name__ == '__main__':
    unittest.main()
