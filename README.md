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
| **A. Judge fitness Goodharts** | The judge-optimized harness raises the judge score (Sonnet 7.5 → 8.2, p = .018; Opus 8.2 → 8.9, p < .001) while precision collapses (stage 5, 24 fresh runs per arm: Sonnet .79 → .43, Opus .87 → .28; on a private transfer space .40 → .03). The optimizer learned to *ban* the claim types that are most often true, because the judge finds them obvious. The stage-2 drop in surprising-valid claims (1.62 → 0.62) did **not** replicate at larger n: the judge arm moves along the frontier to low precision and low prior. | P1, A-prec, A-judge supported; A-SV not replicated |
| **Validity fitness plays safe** | The validity-optimized harness restricts itself to mechanical claim types. The mean prior rises by 6 points, the set of distinct claims narrows, and neither precision nor SV improves. | P2 supported |
| **SV fitness does not help** | No candidate beat direct generation. After a bug fix and a rerun, the one accepted candidate regressed on fresh runs from 1.67 to 1.25 (winner's curse). | P3 refuted |
| **M. Measurement pushes the frontier, for one model** | Before proposing, the model designs 100 experiments (one edit on one parent cell; measured on the test split, truth uses val) and reads the results. With Opus, SV doubles from 1.08 to 2.12 per run (24 fresh runs per arm, 90% CI [+0.54, +1.58], Holm p = .003) at .75 precision; the arm scored 2.12 in all three independent batches. With Sonnet, SV does not move (+0.04, CI upper bound +0.38) and precision collapses (.79 → .45), matching an earlier pilot in which Sonnet followed noise-level measurements 94% of the time under a reconsider prompt vs 41% for Opus. | Stage 5 M: Opus supported, Sonnet refuted |
| **D. The frontier does not move** | 18 harnesses across two generators were tested on fresh runs; none beats plain direct generation on SV after Holm correction (largest gain +0.25 per run). On Sonnet, lowering the prior costs precision (ρ = .87 across 13 harnesses); on Opus, the frontier shape is inconclusive (ρ = .40, 5 harnesses). | D1 supported, D2 inconclusive |

**Takeaway.**
- An LLM judge is an exploitable target for ideation optimization: it rewards moving away from the prior and pays for it in precision.
- Harness engineering that only changes the model's internal reasoning (retrieval, roles, challenge, selection, strategy text) trades precision against surprise along a frontier; it does not push the frontier.
- Letting the model design and read a small number of real experiments can double the true-and-surprising findings, but only for a model that discounts single observations; the same measurements wreck precision for another model.

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
  s4.py                         stage 4: measurement knob (random / actively designed probes before proposing)
  s5.py                         stage 5: 24-run confirmation of A and M for both generators
  s6.py                         stage 6: M under noisy and low-fidelity measurements (Opus)
  PREREG_s*.md, RESULT_s*.md    pre-registrations (written before any call) and results, in Chinese
  out/, out2/                   every LLM call, cached by content (prompts' outputs, priors, judge scores, optimizer logs)
```

## Reproduce

```bash
pip install -r requirements.txt
cd experiments/harness
python3 s1_analyze.py && python3 s2_analyze.py && python3 s3_analyze.py && python3 s4.py analyze && python3 s5.py analyze && python3 s6.py analyze   # no API calls, reads the caches
python3 ../../figures/make_frontier.py
```

Producing new runs needs the `claude` CLI. Every role is one `claude -p --tools ""` call. Uncached literature queries also need a retriever (`LITSEARCH`, a script that prints arXiv/OpenAlex/Semantic Scholar hits). Rebuilding `nb_space.json` needs a NAS-Bench-201 table loader (`NB201_LOADER`).

## Scope and caveats

- **Single training space.** NB201 is familiar to the models: direct generation already reaches .78 precision. The private CTR transfer space used in the original study is not released, so only aggregate numbers are reported.
- **Small optimization and evaluation budgets.** Each arm ran 3 rounds × 3 candidates × 6 runs. Most harnesses got 8 fresh runs, and only the three key arms got 24 in stage 5. Two batches of 8 runs of the same harness can differ by up to 1 SV per run.
- **Measurement noise.** Main measurements are seed-averaged table lookups. In stage 6, Opus was given the same designed probes with synthetic seed-level noise (N1, assumed σ), doubled noise (N2), or 12-epoch proxy values (L, ~70% sign agreement). The SV gain survived in all three (+0.75 to +1.13 over direct generation); only precision degraded (.75 → .65). Because SV is insensitive to measurement quality, the gain may come partly from the act of designing experiments rather than from their values; the shuffled-results control has not been run.
- **Claude-only model panel.** The judge and proposer are both Opus, and the prior panel contains only Claude models.
- **A regex bug**, now fixed, rejected 3 candidates in stage 2. See REPORT.md §5.4.
- **Claim D was formed after stage 2.** Its test (stage 3) was pre-registered.

## Related work

Si et al. 2601.14525 (execution-grounded ideation rewards; safe-idea collapse), HindSight 2603.15164 (judge scores vs later impact), Gao et al. 2210.10760 (reward-model overoptimization), IdeaScientist 2610.04074, Harness² 2610.14224, ADAS 2408.08435, AFlow 2410.10762, NAS-Bench-201 (Dong & Yang, ICLR 2020).

## License

MIT for the code. `nb_space.json` contains summary statistics derived from NAS-Bench-201; see that benchmark's own terms.
