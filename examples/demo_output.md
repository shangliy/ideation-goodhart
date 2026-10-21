<!-- Output of `ideagen demo --model sonnet --prior haiku` (one run, 2026-10-21). One run on a synthetic grid: an illustration, not a benchmark. A single Haiku rater scores almost everything below 50, so use two or more prior models in practice. -->

# ideagen hypotheses

Held-out check: a hypothesis is supported when every group shows the stated sign with |interaction| >= 2 standard errors. Guessing at random would be right 0.08 of the time (41 of 512 testable directional claims).

## direct

10 hypotheses, 2 supported of 10 testable (precision 0.20), 2 supported and unexpected (panel prior < 50)

| # | hypothesis | why (model) | prior | verdict |
|---|---|---|---|---|
| 1 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Large embeddings overfit more, so regularization pays off more at emb_dim 256. | 22.0 | not supported |
| 2 | Setting lr = 0.01 raises recall_at_20 less when model = transformer than otherwise | Transformers are unstable at high learning rates, so a high lr hurts them more. | 28.0 | contradicted |
| 3 | Setting lr = 0.003 raises recall_at_20 more when model = mf than otherwise | Matrix factorization tolerates and benefits from larger learning rates. | 22.0 | not supported |
| 4 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | Large batches pair well with a high learning rate, which compensates for fewer updates. | 25.0 | supported |
| 5 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | A large batch plus a tiny lr leaves very few effective updates and slows convergence. | 33.0 | supported |
| 6 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = low than otherwise | Small embeddings are capacity limited, so heavy dropout hurts more. | 30.0 | contradicted |
| 7 | Setting emb_dim = 128 raises recall_at_20 less when lr = low than otherwise | Larger embeddings need more training to converge, which a tiny lr does not give. | 27.0 | not supported |
| 8 | Setting emb_dim = 128 raises recall_at_20 more when model = transformer than otherwise | Transformers use extra embedding capacity better than simpler models. | 24.0 | not supported |
| 9 | Setting dropout = 0.1 raises recall_at_20 more when lr = high than otherwise | Light regularization stabilizes the noisy training caused by a high learning rate. | 22.0 | not supported |
| 10 | Setting dropout = 0.5 raises recall_at_20 less when model = mf than otherwise | Matrix factorization has little overfitting from depth, so heavy dropout mostly removes useful signal. | 27.0 | not supported |

## measure (100 experiments first)

10 hypotheses, 7 supported of 10 testable (precision 0.70), 7 supported and unexpected (panel prior < 50)

| # | hypothesis | why (model) | prior | verdict |
|---|---|---|---|---|
| 1 | Setting emb_dim = 256 raises recall_at_20 less when model = mf than otherwise | Larger embeddings help gru and transformer by several points but give about zero or negative gains for mf. | 30.0 | supported |
| 2 | Setting emb_dim = 16 raises recall_at_20 more when model = mf than otherwise | Shrinking embeddings costs gru and transformer a lot, but mf barely loses (or gains) since it did not benefit from size. | 20.0 | not supported |
| 3 | Setting model = gru raises recall_at_20 less when lr = high than otherwise | Switching mf to gru at lr=0.01 was clearly negative on every task, unlike at lower learning rates. | 25.0 | supported |
| 4 | Setting lr = 0.01 raises recall_at_20 less when model = gru than otherwise | gru dropped sharply at lr=0.01 (-5.6, -7.6) while mf and transformer were mixed or mild. | 25.0 | supported |
| 5 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | The switch to transformer gave its largest gains at lr=0.01 (+5.5, +1.6, +8.1), as transformer tolerates high lr better than gru. | 25.0 | supported |
| 6 | Setting dropout = 0.3 raises recall_at_20 less when batch = 4096 than otherwise | Dropout 0.3 was negative on all tasks at batch 4096, worse than at batch 1024. | 30.0 | not supported |
| 7 | Setting lr = 0.01 raises recall_at_20 more when batch = 4096 than otherwise | Large batches stabilize high lr, so the lr=0.01 penalty is only about -1 at batch 4096 versus -5 to -8 at smaller batches. | 40.0 | supported |
| 8 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | At lr=0.01 the larger batch gave positive gains (+1.5, +4.1), unlike the slightly negative effects at other learning rates. | 40.0 | supported |
| 9 | Setting emb_dim = 256 raises recall_at_20 less when dropout = 0.5 than otherwise | Heavy dropout on a large embedding over-regularizes, so the emb_dim gain shrinks at dropout 0.5. | 30.0 | supported |
| 10 | Setting model = transformer raises recall_at_20 more when dropout = 0.5 than otherwise | Transformer's advantage over gru was larger at dropout 0.5 (+2.4, +3.7, +3.2) than at dropout 0. | 30.0 | not supported |
