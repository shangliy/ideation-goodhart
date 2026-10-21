# Worked example: from one paper to a ranked proposal

The user profile below is part of the example, not a default. Take the real
user's constraints, data and prior work from the conversation and project
files; do not assume they share this profile or this research agenda.

Seed: FLASH, "Rethinking Semantic ID Construction for Generative
Recommendation" (arXiv 2610.07402). User context: recommender-systems
researcher, one 8 GB GPU, Amazon review and MovieLens data, SASRec and ItemKNN
code, an LLM reranker API at about $0.1 per 1k calls, prior work on
calibration and on candidate-set construction bias.

## Gap card (abridged)

### G1: factorized semantic-ID scores are overconfident
**Gap:** generative recommenders with parallel decoding score an item by
summing per-position log-probabilities as if code positions were independent,
but SimHash codes are sign bits of one vector and are strongly correlated, so
the item posterior is miscalibrated in setting "SID retrieval feeding a
downstream reranker or a confidence-gated decision".
**Deficiency kind:** wrong inductive bias (independence over correlated
factors).
**Evidence:** 2610.07402 Eq. 3 (conditional independence across m positions),
Eq. 1 (codes from random hyperplanes on the same embedding), Eq. 9 (sum of
cached log-probs at inference). No calibration metric anywhere in the paper.
**Near-miss work:** RPG (unordered PQ codes, also parallel) shares the
assumption; TIGER avoids it by autoregressive decoding at a latency cost.
**Who is blocked:** anyone using SID scores as confidences: gating an LLM
reranker, selective recommendation, fusion with CF scores.
**Evidence route:** ECE and NLL of the item posterior versus a calibrated
dense scorer at matched ranking quality; effect of m on overconfidence.
**Confidence:** verified for the assumption; the magnitude of the
miscalibration is speculation until measured.

## Candidate cards (abridged)

### C1: length-normalized and product-of-experts calibration from ASR/MT
**Intuition:** treat the m position heads as experts whose log-scores are
summed; borrow score normalization and PoE temperature fitting from speech
and translation, where summing correlated token scores is known to need
correction.
**Abstract structure:** independent factors multiplied although correlated.
**Source mechanism:** length normalization and lattice rescoring in ASR;
PoE calibration. Works there because per-token scores are correlated through
the acoustic / source context, exactly as code positions are through the
embedding.
**Mapping:** fit one temperature per position plus a global item-level
temperature on held-out interactions; alternatively learn a low-rank
correction over positions.
**Key assumption:** the miscalibration is systematic (same direction for
most items), not item-specific noise.
**Closest in-domain work:** FLASH's learnable per-position tau_j (Eq. 9),
which is a ranking temperature, never evaluated for calibration.
**Novelty type:** recombination of distant concepts.
**Main risk:** calibration may not change top-k ranking at all, making the
paper a "so what" unless a downstream consumer (gating, fusion) benefits.
**Cheapest test:** train FLASH-style head on Amazon Beauty (12k items), report
ECE/NLL for m in {4, 16, 64}, one day on 8 GB.

### C2: SID candidate pools are semantically clustered, so an LLM reranker
gains less from them than from CF pools
(attacks a second gap on candidate-set composition; cross-domain source:
diversity indices from ecology; omitted here for length)

### C3: in-domain patch: a frozen-embedding kNN baseline for Table 1
(novelty type: patch; valuable as an experiment inside P1, not alone)

## Prior art (abridged)

| query | closest hit | same? |
|---|---|---|
| "semantic ID calibration generative recommendation" | none on calibration; TIGER/LETTER/RPG on ranking only | no |
| "multi-token prediction calibration independent heads" | MTP papers for LLMs discuss acceptance rates, not posterior calibration | partial: mechanism studied for LLM decoding, not for item posteriors |
| "product of experts temperature scaling correlated factors" | PoE calibration in classification | partial: mechanism exists, setting new |

**Label:** partial. What remains: the recommendation setting, the link to
code length m, and the downstream consumer (gating / fusion).

## Review (abridged)

**Verdict C1:** survive. **Strongest objection:** ranking-invariant
calibration changes nothing unless a downstream decision uses the
confidence; the proposal must include that consumer. **Non-obviousness:**
not derivable from FLASH plus RPG alone; neither discusses confidences.
**Scores:** novelty 3, feasibility 5, impact 3, fit 5 (matches the user's
calibration work and cached pipelines). **Tag:** [small-GPU].

## Proposal (abridged)

**Thesis:** parallel-decoded semantic-ID recommenders produce overconfident
item posteriors, and a per-position plus item-level recalibration restores
confidences usable for selective recommendation and CF fusion.
**Minimal decisive experiment:** Amazon Beauty; FLASH-style head with
m in {4, 16, 64}; baselines SASRec softmax and a dense kNN scorer; metrics
NDCG@10 (must be unchanged), ECE, NLL, and risk-coverage AURC when gating a
reranker call at 20% coverage. Supports if ECE rises with m and
recalibration recovers AURC; refutes if ECE is flat in m or gating gains
nothing. Cost: two GPU-days, about $0.2 of reranker calls.
**Kill criteria:** ECE at m=64 within 0.02 of m=4, or recalibrated gating
no better than SASRec-confidence gating.
