# ideation-goodhart

**What should you optimize when you optimize an LLM research-ideation harness?**
Pre-registered pilot experiments on a hypothesis space with measured ground truth (NAS-Bench-201).

![precision vs surprising-valid](figures/frontier.png)

## Summary

We ask an LLM to propose 10 testable conditional hypotheses about a benchmark whose data it cannot see:

> "Setting edge *j* to op *b* helps more (or less) when the parent cell has op *c* on edge *k*."

Every such claim, 1,740 in all, can be checked against the fully tabulated NAS-Bench-201 results; 198 of them hold. We also need to know which claims are surprising. Three models (Haiku, Sonnet, Opus) rate every claim, which gives a shared prior. A claim is **surprising-valid (SV)** if it holds and the models other than the generator rate it below 50%.

We then optimize the ideation **harness** with the same Harness²-style optimizer under three fitness functions. The harness covers retrieval, roles, self-challenge, selection, and free-text strategy. The three fitness functions are:

- **J**: an LLM judge's novelty + quality score;
- **V**: the number of hypotheses that hold;
- **SV**: the number of surprising-valid hypotheses.

## Findings (pilot scale; all criteria pre-registered)

| | Result | Verdict |
|---|---|---|
| **A. Judge fitness Goodharts** | The judge-optimized harness raises the judge score (7.5 → 8.2, p = .018) while precision collapses from .78 to .29 and SV falls from 1.62 to 0.62 per run. On a private transfer space, the judge score rises from 6.6 to 8.8 while precision falls from .40 to .03. With Opus as generator, precision again collapses (.79 → .25), but SV does not fall. The optimizer learned to *ban* the claim types that are most often true, because the judge finds them obvious. | P1 supported |
| **Validity fitness plays safe** | The validity-optimized harness restricts itself to mechanical claim types. The mean prior rises by 6 points, the set of distinct claims narrows, and neither precision nor SV improves. | P2 supported |
| **SV fitness does not help** | No candidate beat direct generation. After a bug fix and a rerun, the one accepted candidate regressed on fresh runs from 1.67 to 1.25 (winner's curse). | P3 refuted |
| **D. The frontier does not move** | 18 harnesses across two generators were tested on fresh runs; none beats plain direct generation on SV after Holm correction (largest gain +0.25 per run). On Sonnet, lowering the prior costs precision (ρ = .87 across 13 harnesses); on Opus, the frontier shape is inconclusive (ρ = .40, 5 harnesses). | D1 supported, D2 inconclusive |

**Takeaway.**
- An LLM judge is an exploitable target for ideation optimization.
- Harness engineering trades precision against surprise, but in this setting it does not produce more claims that are both true and surprising.
- Direct generation is 3–7× cheaper and sits on or near the frontier.

The full report is in Chinese: [REPORT.md](REPORT.md). It covers background, related work, every stage, verdicts, limitations, and cost.

## Repository layout

```
REPORT.md                       full experiment report (Chinese)
figures/                        frontier.png and the script that draws it
experiments/harness/
  nb_space.py, nb_space.json    NB201 claim space (870 cells x 3 datasets, derived from NAS-Bench-201)
  search.py                     literature retrieval with a blocklist for papers that analyse the benchmark table
  s1.py, s1_analyze.py          stage 1: L9 screen of structural knobs (retrieval, roles, challenge, selection)
  s2.py, s2_analyze.py          stage 2: one optimizer, three fitness functions; stage 3: fixed SV arm, Opus generator
  s3_analyze.py                 stage 3: frontier test (claim D)
  PREREG_s*.md, RESULT_s*.md    pre-registrations (written before any call) and results, in Chinese
  out/, out2/                   every LLM call, cached by content (prompts' outputs, priors, judge scores, optimizer logs)
```

## Reproduce

```bash
pip install -r requirements.txt
cd experiments/harness
python3 s1_analyze.py && python3 s2_analyze.py && python3 s3_analyze.py   # no API calls, reads the caches
python3 ../../figures/make_frontier.py
```

Producing new runs needs the `claude` CLI. Every role is one `claude -p --tools ""` call. Uncached literature queries also need a retriever (`LITSEARCH`, a script that prints arXiv/OpenAlex/Semantic Scholar hits). Rebuilding `nb_space.json` needs a NAS-Bench-201 table loader (`NB201_LOADER`).

## Scope and caveats

- **Single training space.** NB201 is familiar to the models: direct generation already reaches .78 precision. The private CTR transfer space used in the original study is not released, so only aggregate numbers are reported.
- **Small optimization and evaluation budgets.** Each arm ran 3 rounds × 3 candidates × 6 runs, and each harness got 8 fresh runs. Two independent batches of direct generation differ by 0.37 SV per run.
- **Claude-only model panel.** The judge and proposer are both Opus, and the prior panel contains only Claude models.
- **A regex bug**, now fixed, rejected 3 candidates in stage 2. See REPORT.md §5.4.
- **Claim D was formed after stage 2.** Its test (stage 3) was pre-registered.

## Related work

Si et al. 2601.14525 (execution-grounded ideation rewards; safe-idea collapse), HindSight 2603.15164 (judge scores vs later impact), Gao et al. 2210.10760 (reward-model overoptimization), IdeaScientist 2610.04074, Harness² 2610.14224, ADAS 2408.08435, AFlow 2410.10762, NAS-Bench-201 (Dong & Yang, ICLR 2020).

## License

MIT for the code. `nb_space.json` contains summary statistics derived from NAS-Bench-201; see that benchmark's own terms.
