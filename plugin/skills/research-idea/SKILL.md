---
name: research-idea
description: Grounded research ideation. Turns seed material (papers, a newsletter issue, a topic, the user's own results) into a short ranked list of research ideas, each with a gap card, a cross-domain mechanism, a prior-art check, a minimal decisive experiment and kill criteria, persisted to an idea ledger. Use this whenever the user wants to FIND or CHOOSE what to work on next: "give me research ideas", "what is the gap in these papers", "is there a paper in this", "what can I do with this newsletter", "brainstorm directions", "找研究想法", "选题", "从这几篇论文里找 gap", "有什么可以突破的", "下一步做什么", or when a paper review ends with "what could be done better". Do NOT use when the user already has one chosen idea and wants the paper written (that is a paper-writing skill's job; this skill hands off to it).
---

# Research idea

Ideation is a task of its own, upstream of writing a paper. This skill runs it as
three roles with fixed artifacts, borrowed from IdeaScientist (arXiv 2610.04074):
a **gap finder**, an **innovator**, and a **report writer**, with a **reviewer**
between them. The artifacts are the point: they let the user audit each step,
and they let a refuted idea stay refuted instead of being reinvented next month.

## Principles (why the workflow looks like this)

- **Fewer, better.** Pierre Marion's essay (pierremarion23.github.io, 2026-10-01)
  argues the bottleneck has moved from solving to *choosing*: judgment, taste and
  verification. Ten shallow ideas waste the user's scarce judgment. Deliver three
  that survived a real attempt to kill them, plus the kill list with reasons.
- **Explore formulations, not just solutions.** For each gap, ask what the
  problem would look like reframed (different unit of analysis, different
  consumer of the output, different observable). Solvable *and* useful is the
  target: an idea that leads nowhere downstream is not worth the user's time.
- **Novelty comes from distant structure.** IdeaScientist's ablation shows
  cross-domain retrieval raises in-domain novelty by about 30 points. Same-field
  reading gives patches; mechanisms from a field with the same abstract problem
  give ideas. Every gap gets at least one candidate whose mechanism comes from
  outside the seed papers' subfield.
- **Prior art before ranking.** Idea ledgers keep recording top picks that died
  because the novelty check came after the ranking. A search that finds the idea
  already published is a success, not a failure. Do it before scoring.
- **Cheap decisive tests win.** The ideas that survive in practice are diagnostic
  findings with a clean counterfactual or identity behind them, testable on
  cached data. Prefer an idea with a one-day experiment that could refute it over
  a grand method with a two-month build.
  When the test is free and read-only, run it as a pilot (step 5b) instead of
  only describing it.
- **Evidence beats deliberation; never rank by an LLM's novelty score.** In the
  ideation-goodhart study (this plugin's repository), optimizing an ideation
  harness for an LLM judge raised the judge's score while precision fell from
  .79–.90 to .28–.43, in two benchmark spaces and two models. Extra internal
  reasoning (retrieval, roles, self-challenge, strategy text) did not yield more
  true-and-unexpected hypotheses than direct generation. Reading about 100 cheap,
  even noisy, experiments before proposing did, in 3 of 4 model × space cells. So the
  review scores here are for the user to audit, not a target to push up, and
  when the user has an experiment grid or a training script, generate
  hypotheses from measurements with `ideagen` (see the `ideagen` skill).
- **Build with the user, do not hand them a verdict.** Mark speculation as
  speculation, never invent numbers, and leave the final pick to the user with
  the evidence laid out. Analogies are hypotheses until data supports them.

## Workspace

```
ideas/                          # in the project root (or where the user says)
  ideas.md                      # ledger: one block per idea, newest first
  ideas_verdicts.md             # every idea ever proposed and what happened to it
  <YYYY-MM-DD>-<slug>/
    00-intake.md                # seed, constraints, what the ledger already rules out
    01-gaps.md                  # gap cards (gap finder)
    02-candidates.md            # candidate cards (innovator)
    03-prior-art.md             # search log and verdict per candidate
    04-review.md                # reviewer verdicts, scores, ranking, kill list
    05-proposals.md             # full proposal for each surviving idea
```

Read `ideas_verdicts.md` first if it exists. Never re-propose a refuted idea
without saying what changed. If the user already keeps a ledger elsewhere (for
example `kg/ideas.md`), use that location instead of creating a second one.

Schemas for every artifact are in `references/schemas.md`. Read it before
writing the first card; the fields are what make the cards auditable.

## Workflow

### 0. Intake

Write `00-intake.md`:
- **Seed.** Which papers, newsletter issue, topic or result. Get full text
  (arXiv HTML via curl, local files); abstracts and newsletter summaries are
  not enough to find real gaps, and they may be wrong.
- **Constraints.** Compute, data, API budget, time, team. Ask only if nothing in
  the conversation, memory or project files answers it; otherwise state the
  assumption and continue.
- **Fit.** The user's existing results, methods and comparative advantage. An
  idea that reuses their cached data or proven pipeline is worth more than an
  equally novel one that does not.
- **Ruled out.** Entries from the ledger that touch the same seed.

### 1. Gap finding (reviewer mindset)

For each seed paper, read it as a hostile but fair reviewer and write at most
two gap cards. When the seed is a topic or the user's own result rather than a
paper, section `01-gaps.md` by axis instead (for a result: its mechanism, its
measurement, its consequences, its boundary conditions), with the same limit
of two cards per axis. The limit forces a choice; the observations that did not make
the cut go into a short "Observations not promoted" list at the end of the
paper's section (one line each, with pointer), so they are not lost and the
reviewer can promote one later. A gap is stated in contrast form: *prior methods do X through Y,
but Y leaves Z unresolved in setting S*. Each card names the deficiency kind
(see `references/gap-taxonomy.md`), the exact section, table or number that
shows it, who is blocked by it, and the evidence that would show it closed.

What counts as a gap and what does not:
- A missing baseline that could erase the paper's claim: yes.
- A confound (same-source teacher, test-set tuning, strawman baseline): yes.
- An unexamined setting the method's assumption breaks in: yes.
- The paper's own future-work paragraph restated: no, unless you add why it is
  hard and what the paper's evidence says about it.
- "More datasets / more seeds": no, unless the missing seeds change a conclusion.

With more than three papers, fan out subagents with the gap-card schema in
the brief (two to four papers each is fine; the point is full-text reading in
a fresh context, not a fixed ratio), then merge. Keep every card's evidence
pointer; a gap without a pointer gets dropped at review.

### 2. Innovation (cross-domain mechanism transfer)

For each gap, abstract it first: what is the *structure* of the problem
(an estimator with high variance, a selection step hiding ground truth, a
calibration that breaks under shift, a credit-assignment failure). Then ask
where else that structure appears and what mechanism solved it there. Search
beyond the subfield (statistics, control, economics, systems, cognitive
science, other ML areas). `references/analogy-playbook.md` has prompts and a
seed list of recurring structures and their known mechanisms.

With many gaps (more than about six), do not write candidates per gap; group
the gaps by abstract structure and write candidates per structure, listing
the gaps each one carries and, in one line each, the gaps carried by none and
why. Looking up the closest in-domain work while writing a card is normal;
log those scoping searches in `03-prior-art.md` under a "scoping" heading so
the later per-candidate check can build on them instead of repeating them.

Write candidate cards: the intuition in two sentences, the gap it attacks,
the source mechanism and why it works at the source, how it maps here, the key
assumption that must hold, the closest in-domain work and the difference, the
single biggest risk, and the novelty type (new topic, recombination of distant
concepts, or reinforcing a weak connection). Aim for two or three candidates per
gap; at least one must be cross-domain. Also include, where it exists, the
boring in-domain patch, so the user can see the contrast.

### 3. Prior-art check (before any ranking)

For every candidate, run two to four searches with specific queries (method
name plus setting, the mechanism plus the target domain, the claim in plain
words). Add one **system-level query** per candidate: the broad goal the
mechanism serves, phrased the way a systems paper would title it (for
example "recursive self-improvement research agent", not "acceptance rule
under noise"), restricted to the last two months. Mechanism-only queries miss
the paper that reached the same goal without the mechanism (a real miss:
arXiv 2609.26457 was not found by noise/acceptance-rule queries). Web search is the default (in Claude Code the WebSearch tool is
deferred; load it with ToolSearch first). A literature retriever over arXiv,
OpenAlex or Semantic Scholar, if one is installed, is a useful second source
for citation-ranked hits, but such tools are usually title-level search and
return noise for mechanism-level queries; when that happens, say so in the log
and rely on web search. Record each
query and the closest hits in `03-prior-art.md`, then label the candidate:
- **exists**: someone did this; cite it and drop or narrow the candidate.
- **partial**: the mechanism exists elsewhere or the setting was studied with
  a different mechanism; state precisely what remains.
- **open**: nothing close found; say what you searched so the user can judge.

Respect time: a seed paper from this month will not have follow-ups yet, so
"open" means little for ideas that are the obvious next step. Apply the
non-obviousness test from step 4.

### 4. Review and ranking

Act as the reviewer role: for each candidate give a verdict (survive, revise
with the single weakest dimension named, or reject with a reason), the
strongest objection in one sentence, and evidence bullets with pointers.

Run the **mechanism non-obviousness test**: could a competent researcher derive
this candidate from the seed paper plus its closest prior work alone? If yes,
it is a patch, and it belongs in the proposal of a bigger idea or in the kill
list, not in the top three.

Score survivors 1 to 5 on novelty, feasibility under the stated constraints,
expected impact if the hypothesis holds, and fit with the user's advantage;
the total is the plain sum, and a tie goes to the cheaper decisive test. Tag
feasibility concretely (for example `[small-GPU]`, `[needs cluster]`,
`[needs private data]`, `[API ~$N]`). Rank and keep up to three: if the user
asked for a number, that number wins; if the third would not survive review,
present two and say why. A candidate with verdict "revise" may be kept only
when the named weakness is fixable inside its proposal, and the verdict stays
visible in the table. Write the kill list with one line each. Write all of it
to `04-review.md`.

### 5. Proposals

For each surviving idea write a proposal in `05-proposals.md` using the schema
in `references/schemas.md`: thesis under 30 words, core problem with a concrete
failing example, key novelty, problem definition, method at the level of flow
and modules, comparison to the closest methods, the **minimal decisive
experiment** (data, baseline, metric, cost, and decision criteria fixed in
advance: what result would support, refute, or leave it inconclusive),
**alternative explanations** with the control that separates each,
falsifiable predictions, limitations, and **kill criteria**
(the observation after which the user should stop). No measured results, ever;
expected effects are labelled as expectations.

Append each proposal's one-block summary to `ideas.md` (the **Workspace**
field is a path relative to the project root) and add a row to
`ideas_verdicts.md` with status `proposed`. When the user later reports an
outcome, update the row; the ledger is only useful if it records deaths.

### 5b. Pilot when it is free

Run this between prior art and review when the cheapest test is already
runnable, so the reviewer can score with numbers instead of guesses; a pilot
can also kill a candidate outright, and the kill list then cites it. If a
pilot runs after the review (because the proposal made it concrete), re-score
the affected candidates and leave the change visible in `04-review.md`.
The decisive experiment is sometimes already runnable: the data is cached, a
script exists, nothing needs training, and a read-only run takes under an
hour. When that holds and the user has not said otherwise, run it, in a
scratch directory, touching nothing in the project. Report it as a **pilot**
with its exact scope (sample, metric, n, CI) and label it so in the proposal
and the ledger; a pilot is evidence about whether to proceed, not a result for
a paper. Marion's point applies: cheap verification is the whole advantage, so
an idea that can be checked today should not wait for a proposal document.
Do not run anything that costs API budget or GPU-hours without saying so first.

### 6. Hand-off

End with: a ranked table (idea, novelty label, feasibility tag, one-line
decisive test), the top proposal in full, and the kill list. If a paper-writing
skill is installed, offer to continue into it with the chosen idea as input. If the user
wants to run the decisive experiment first, that is usually the better order.

## Reply shape

Keep the chat reply short; the files hold the detail. Lead with the ranked
table, then the top proposal, then the kill list. For the top proposal the
reply carries only thesis, core problem, key novelty, minimal decisive
experiment, falsifiable predictions and kill criteria; the remaining fields
stay in `05-proposals.md` with a pointer. Name the workspace path once.
Mark anything not yet verified (an "open" novelty label, an analogy without
data) as such in the reply, not only in the files.

## References

- `references/schemas.md`: exact fields for intake, gap, candidate, prior-art,
  review and proposal artifacts.
- `references/gap-taxonomy.md`: deficiency kinds, the reviewer checklist, and
  what separates a gap from a complaint.
- `references/analogy-playbook.md`: how to abstract a gap into a structure and
  where to look for mechanisms; seed list of structures and known transfers.
- `references/example.md`: one worked chain from a real paper (FLASH, arXiv
  2610.07402) to a ranked proposal, showing the expected depth of each card.
  Its user profile is an illustration; take the real user's constraints and
  prior work from the conversation, never from the example. If a seed paper
  happens to be the example's paper, re-derive the gap from the text and say
  how it differs from the example's card.
