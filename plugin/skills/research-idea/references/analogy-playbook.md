# Analogy playbook: from a gap to a mechanism from elsewhere

The innovator's job is not to think harder inside the subfield. It is to
strip the gap to its structure, find where that structure has already been
solved, and carry the mechanism across. IdeaScientist measured this: retrieving
from other domains raised in-domain novelty from 36% to 67%, same-domain
retrieval did not.

## Step 1: abstract the gap

Rewrite the gap with no domain nouns. Templates:

- "A scorer only sees candidates chosen by an upstream selector, so its errors
  are confounded with the selector's." (selection / missing-data structure)
- "A reward depends on a low-dimensional function of a high-dimensional
  sample, so most sampled noise is irrelevant." (variance / sufficiency)
- "Confidences are calibrated on one distribution and used on another."
  (shift / transfer of calibration)
- "An optimizer sees only an aggregate score per trial and cannot attribute
  failure." (credit assignment / diagnosis)
- "Independent factors are multiplied although they are correlated."
  (factorization / overconfidence)
- "A policy decides when to stop gathering evidence using its own report of
  coverage." (self-assessment / stopping)
- "A budget is fixed per item although difficulty varies per item."
  (adaptive allocation)
- "A pool contains near-duplicates that crowd out distinct evidence."
  (redundancy / diversity)

If you cannot write the structure in one sentence without domain words, the
gap is not yet understood; go back to the paper.

## Step 2: where has this structure been solved?

Search by structure, not by topic. Useful homes for common structures:

| structure | fields with mature mechanisms | mechanisms worth reading |
|---|---|---|
| selection confounds the scorer | causal inference, survey statistics, counterfactual learning to rank | inverse propensity weighting, doubly robust estimators, Heckman correction, logQ correction for sampled softmax |
| high-variance gradient, low-dimensional sufficient statistic | statistics, RL theory | Rao-Blackwellization, control variates, conditional expectation, antithetic / orthogonal sampling, zeroth-order methods |
| calibration under shift | weather forecasting, medical risk, conformal prediction | temperature per group, label-shift adaptation, conformal risk control, selective prediction with risk-coverage curves |
| credit assignment across a pipeline | compilers, systems performance engineering, program synthesis | profiling with attribution, counterfactual replacement of one stage, Shapley over stages |
| overconfident factorized scores | speech recognition, machine translation | length normalization, product-of-experts calibration, lattice rescoring, teacher-forced vs free-running mismatch |
| self-reported stopping | sequential analysis, clinical trials, active learning | pre-registered stopping rules, SPRT, value of information, budget-aware stopping with external validation |
| per-item adaptive budgets | adaptive testing (IRT), anytime algorithms, speculative decoding | item response theory, early exit with confidence thresholds, bandit allocation |
| redundancy in a pool | ecology (diversity indices), search result diversification, portfolio theory | effective number of species (Hill numbers), xQuAD, determinantal processes, risk-adjusted selection |
| learning from a proxy objective | economics (Goodhart), RLHF | reward modelling with uncertainty, constrained optimization, proxy-outcome validation |
| transfer of a trained decision model | aviation and medicine automation | doctrine for automation level, unassisted practice, human-in-the-loop thresholds |

Add rows as you find them; this table is a seed, not a boundary.

## Step 3: map and stress the mapping

For the chosen mechanism, write:
- what plays the role of each object in the source (the unit, the noise, the
  signal, the budget);
- the assumption the source relies on and whether it holds here (for example
  IPW needs known propensities; does the target have them?);
- the first thing that would go wrong (dimension too small, labels too sparse,
  the correction cancels the signal you want).

If two mechanisms compete, keep both as separate candidates. If none maps
cleanly, say so and write the in-domain patch instead; an honest patch beats a
forced analogy.

## Step 4: label the novelty type

- **new topic**: the setting itself has not been studied (rare; check the
  obvious next-step problem).
- **recombination of distant concepts**: the mechanism is from a different
  field and the mapping is non-trivial (the typical good idea).
- **reinforcing a weak connection**: the link exists in a footnote or a
  workshop paper and nobody tested it properly.
- **in-domain patch**: a missing baseline or ablation; valuable as an
  experiment, not as a paper by itself.

## Prompts that help when stuck

- Who else has to make this decision with this information? What do they do?
- What would the problem look like if the consumer of the output were an
  agent, a regulator, a marketplace, a classroom?
- What is the cheapest quantity that, if measured, would change what everyone
  in this subfield does next?
- Which claim in the paper rests on an identity or counterfactual that could
  be checked on cached data in a day?
