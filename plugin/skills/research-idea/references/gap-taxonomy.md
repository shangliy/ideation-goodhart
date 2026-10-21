# Gap taxonomy and reviewer checklist

## Deficiency kinds (tag every gap with one)

| kind | what it means | typical evidence |
|---|---|---|
| missing mechanism | nothing in the method addresses a sub-problem it must solve | the method's own failure cases; a setting where its assumption is false |
| algorithmic limitation | the mechanism exists but scales, converges or generalizes badly | complexity, latency tables, OOD drops |
| wrong inductive bias | the model assumes structure the data lacks (or ignores structure it has) | an ablation that removes the structure and loses nothing; independence assumptions over correlated variables |
| credit-assignment flaw | the signal used for learning or selection does not reach the component that caused the outcome | aggregate-only rewards, per-trial scores, no attribution |
| training-objective mismatch | the loss optimizes a proxy that diverges from the evaluated quantity | gains on the proxy with flat or negative downstream metrics |
| representation limitation | the representation cannot express what the task needs | collisions, saturation, loss of fine-grained information |
| evaluation gap | the claim is not what the experiment measures | strawman baseline, confounded comparison, test-set tuning, n too small, same-source teacher and competitor, metric that hides cost |
| scalability limit | works at the paper's size, no evidence beyond | single corpus size, no growth curve |
| missing setting | an important deployment condition is untested | distribution shift, adversarial inputs, new consumer of the output (an agent instead of a human) |

Evaluation gaps are the most common and the most useful: a missing baseline
that could erase a claim is a gap *and* an experiment, and the experiment is
usually cheap.

## Reviewer checklist (run per paper before writing gap cards)

1. What does the abstract claim, and which table is the evidence? Write both.
2. Which baseline is missing that a sceptic would ask for first? Often it is
   the cheapest one: a one-shot rerank, a deduplication filter, a frozen
   embedding plus kNN, a prior-informed random search.
3. Which comparison is confounded? Same-source teacher and competitor, numbers
   copied from other papers with different embeddings, hyperparameters tuned on
   the test set, baselines run inside the proposed system's own harness.
4. Where is the n, the seed count, the CI? Compute the standard error yourself
   when the paper does not; a 9-point gain on 100 questions has SE near 5.
5. Which sentence in the abstract is stronger than the table supports?
6. Which design choice is asserted but never ablated?
7. What is reported in the appendix but not in the conclusion? Hidden negative
   results are gaps with evidence already attached.
8. Which assumption would break first in a different setting the user cares
   about (their data, their consumer, their scale)?
9. What did the authors list as future work, and why has it not been done? If
   the answer is "it is hard because of Z", Z is the gap; if the answer is
   "nobody got to it yet", it is a patch and probably in someone's queue.

## A gap is not

- "Only four datasets." Unless a fifth would plausibly reverse the conclusion.
- "No theory." Unless a specific theoretical question would change practice.
- "Could be combined with X." That is a candidate, and a weak one unless the
  combination resolves a named deficiency.
- The paper's limitations section copied out. Those are known; the gap is what
  they imply and what evidence already points at it.

## Working from summaries

Newsletter summaries and abstracts drop caveats, misstate counts and omit
negative results. When the seed is a summary, obtain the full text before
writing a card. If that is impossible, label every card `[from summary]` and
lower its confidence; the review stage should not promote such a card past
"revise".
