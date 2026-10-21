# Artifact schemas

Every artifact is Markdown. Fields are headings or bold labels so a reader can
scan them; keep the order. Fields marked (req) must be present; an artifact
missing one is incomplete and should not feed the next stage. Pointers are
`arXiv id §section / Table n / Eq. n` or `file:line`; a claim about a paper
without a pointer is a guess and must be labelled `[speculation]`.

## 00-intake.md

```
# Intake: <slug>            <date>
**Seed** (req): list of papers (id, title, venue/date) or topic or result
**Full text obtained**: which ones, where stored
**Constraints** (req): compute / data / API budget / time / team, each with its source (user said, memory, project file, assumed)
**User's advantage**: cached data, pipelines, prior results, access
**Ruled out by ledger**: entries from ideas_verdicts.md that touch this seed, with their verdict
**Open questions for the user**: only ones that change the work materially
```

## 01-gaps.md: gap card (gap finder)

One `##` section per seed paper (or per axis when the seed is a topic), with a
3-5 sentence mechanism-level summary of what the paper does, then at most two
gap cards.

```
### G<n>: <short name>
**Gap** (req): one sentence in contrast form: prior methods do X through Y, but Y leaves Z unresolved in setting S.
**Deficiency kind** (req): one of the kinds in gap-taxonomy.md
**Evidence** (req): pointer(s) and the number or statement that shows it
**Near-miss work**: the closest methods and how each comes close; what they share that misses the gap
**Who is blocked / what it unlocks** (req): the downstream question that cannot be answered while the gap stands
**Evidence route** (req): the observable comparison that would show a future method closed the gap, without proposing the method
**Confidence**: verified / likely / speculation
```

After the cards for a paper, add `**Observations not promoted:**` with one
line per weakness that did not become a card (pointer included). The reviewer
may promote one of these later; nothing observed should be lost.

## 02-candidates.md: candidate card (innovator)

```
### C<n>: <short name>   (attacks G<m>)
**Intuition** (req): one or two sentences, plain language
**Abstract structure of the gap** (req): the problem stripped of domain words (e.g. "a selection step hides ground truth from the scorer")
**Source mechanism** (req): the field and mechanism borrowed, why it works there, with a citation or pointer
**Mapping** (req): how the mechanism translates to this setting, in words; no pseudocode or hyperparameters
**Key assumption** (req): what must hold for the transfer to work
**Closest in-domain work** (req): cite it and name the specific difference
**Novelty type** (req): new topic / recombination of distant concepts / reinforcing a weak connection / in-domain patch
**Main risk** (req): the single most likely reason it fails
**Cheapest test**: the smallest experiment that could refute it, with rough cost
```

## 03-prior-art.md

Start with a `### Scoping` block for searches run while writing candidate
cards (same table columns), then one block per candidate.

```
### C<n>
| query | source | closest hit (id, title, year) | why it is or is not the same |
...
**Label** (req): exists / partial / open
**What remains** (req when partial): the precise unclaimed part
**Searched on** (req): date, so a later reader knows the cutoff
```

## 04-review.md

```
### C<n> verdict
**Verdict** (req): survive / revise: <weakest dimension> / reject: <reason>
**Strongest objection** (req): one sentence
**Evidence**: bullets with pointers; each either supports a source mechanism or challenges one
**Non-obviousness** (req): derivable from seed + closest prior work alone? yes / no, with one line why
**Pilot evidence** (if a pilot ran): the numbers that bear on this candidate, with n and CI, marked `[pilot]`
**Scores** (survivors only): novelty, feasibility, impact, fit, each 1-5 with a phrase of justification
**Feasibility tag** (req): e.g. [small-GPU] [needs cluster] [needs private data] [API ~$N] [needs human labels]

## Ranking
| rank | candidate | verdict | novelty label | feasibility | decisive test (one line) | total (sum of 4 scores) |

## Kill list
- C<n>: <one line reason>   (cite `[pilot]` numbers when the kill came from a pilot)
```

## 05-proposals.md: proposal (report writer)

Adapted from IdeaScientist Table 5; results-masked: no measured numbers.

```
## P<n>: <title>
**Thesis** (req): one sentence, under 30 words, no metric numbers
**Core problem** (req): what fails, why it matters, one concrete failing example with a pointer
**Key novelty** (req): a name and an intuition-level explanation
**Problem definition** (req): setting, inputs, outputs
**Builds on**: gap G<m>, candidate C<n>, source mechanism
**Method** (req): flow, modules and their roles, the novel element, key design choices
**Comparison to closest methods** (req): each with the specific difference
**Minimal decisive experiment** (req): data, baselines, metric, estimated cost and time, and decision criteria fixed before the run: the result that would support, the result that would refute, and the result that would be inconclusive with what to do next if it is
**Alternative explanations** (req): competing accounts, confounders and measurement limits that could produce the supporting result without the thesis being true, each with the comparison or control that tells it apart
**Full evaluation plan**: benchmarks, baselines, metrics, ablations, as a plan
**Falsifiable predictions** (req): 2-4 claims, each paired with the observation that falsifies it
**Limitations** (req): three to five
**Kill criteria** (req): the observation after which to stop
**Pilot** (if run): what was run, on what data and sample, the observed numbers with CI, and what they imply for the kill criteria; marked `[pilot, not a result]`
**Prior-art status** (req): label from 03 and the closest work
**Feasibility tag** (req)
```

## ideas.md ledger block

```
## <date> <title>   [status: proposed]
**Builds on:** ...  **Gap:** ...  **Idea:** ...  **Minimal experiment:** ...
**Why new:** ...  **Risk:** ...  **Workspace:** ideas/<date>-<slug>/   (relative to the project root)
```

## ideas_verdicts.md row

```
| id | idea (one line) | date proposed | verdict (proposed / pilot-positive / pilot-refuted / running / kept / weak / refuted / prior art) | evidence or reason |
```

Update the verdict row when the user reports an outcome. A ledger that only
records births teaches nothing.

When a row turns `refuted` because a specific mechanism failed, add one line
to its evidence: **direction without the mechanism**: does the broader
system-level claim (for example "the loop improves the agent at all") still
stand, and is it already published? A refuted mechanism is not a dead
direction; record which of the two died.
