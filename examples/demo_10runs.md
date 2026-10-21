<!-- Output of `ideagen demo --model sonnet --runs 10` (2026-10-21; 30 Claude CLI calls). Synthetic grid; truth = held-out half of the runs. -->

# ideagen hypotheses

Held-out check: a hypothesis is supported when every group shows the stated sign with |interaction| >= 2 standard errors. Guessing at random would be right 0.08 of the time (41 of 512 testable directional claims).

## Summary over runs

| arm | runs | supported per run [90% CI] | precision [90% CI] |
|---|---|---|---|
| direct | 10 | 2.90 [2.60, 3.20] | 0.30 [0.28, 0.33] |
| measure | 10 | 6.40 [5.70, 7.10] | 0.65 [0.58, 0.72] |

## direct

10 hypotheses, 3 supported of 10 testable (precision 0.30)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting emb_dim = 256 raises recall_at_20 more when dropout = 0.5 than otherwise | Heavy dropout offsets the overfitting risk of a larger embedding, so added capacity pays off more. | contradicted |
| 2 | Setting emb_dim = 256 raises recall_at_20 less when lr = high than otherwise | Large embeddings combined with a high learning rate tend to be unstable. | not supported |
| 3 | Setting lr = 0.001 raises recall_at_20 more when model = transformer than otherwise | Transformers are especially sensitive to learning rate and benefit from a well-tuned mid value. | not supported |
| 4 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Dropout regularizes large embeddings but hurts small, underfit ones. | not supported |
| 5 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | Large batches with a low learning rate give too few effective updates and undertrain the model. | supported |
| 6 | Setting lr = 0.003 raises recall_at_20 more when batch = 4096 than otherwise | Large batches tolerate and need a larger learning rate. | supported |
| 7 | Setting dropout = 0.5 raises recall_at_20 less when model = mf than otherwise | Matrix factorization has little capacity to overfit, so heavy dropout mostly hurts. | not supported |
| 8 | Setting model = transformer raises recall_at_20 more when emb_dim = high than otherwise | Attention models need width to realize their advantage over simpler models. | supported |
| 9 | Setting batch = 256 raises recall_at_20 more when lr = high than otherwise | Smaller batches add gradient noise that stabilizes training at high learning rates. | contradicted |
| 10 | Setting dropout = 0.1 raises recall_at_20 more when model = gru than otherwise | Recurrent models benefit from light regularization. | not supported |

## measure (100 experiments first)

10 hypotheses, 7 supported of 10 testable (precision 0.70)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | Switching gru to transformer gave +5.5/+1.6/+8.1 at lr=0.01, much larger than at mid or low lr. | supported |
| 2 | Setting model = gru raises recall_at_20 less when lr = high than otherwise | Switching mf to gru was negative on all tasks at lr=0.01 but positive at lr=0.001. | supported |
| 3 | Setting emb_dim = 256 raises recall_at_20 more when model = transformer than otherwise | Going to 256 helped transformer consistently on all tasks, more than for mf. | supported |
| 4 | Setting emb_dim = 256 raises recall_at_20 less when model = mf than otherwise | Going to 256 gave roughly zero or negative changes for mf rows while other models gained. | supported |
| 5 | Setting emb_dim = 256 raises recall_at_20 less when dropout = 0.5 than otherwise | At dropout 0.5 the large-embedding gain shrank or turned negative, compared with dropout 0. | supported |
| 6 | Setting emb_dim = 256 raises recall_at_20 more when dropout = 0 than otherwise | Without dropout, going to 256 gave strong gains (about +4 on transformer rows). | not supported |
| 7 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | Batch 4096 gained at lr=0.01 across gru, mf and transformer versus losses at lr=0.001. | supported |
| 8 | Setting lr = 0.01 raises recall_at_20 more when batch = 4096 than otherwise | Raising lr to 0.01 hurt much less (about -1) at batch 4096 than at batch 1024 (-5.6/-2.5 for gru). | supported |
| 9 | Setting model = transformer raises recall_at_20 more when dropout = 0.5 than otherwise | Switching to transformer gained more at dropout 0.5 than at dropout 0 on all tasks. | not supported |
| 10 | Setting model = transformer raises recall_at_20 less when emb_dim = low than otherwise | Switching to transformer gave a negative or smaller gain at emb_dim=16 than at 64 or 256. | not supported |

## direct, run 2

10 hypotheses, 3 supported of 10 testable (precision 0.30)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting emb_dim = 256 raises recall_at_20 more when model = transformer than otherwise | Transformers have more capacity to exploit large embeddings than the other models. | supported |
| 2 | Setting dropout = 0.5 raises recall_at_20 more when emb_dim = high than otherwise | Large embeddings overfit more, so strong regularization pays off more. | contradicted |
| 3 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Moderate dropout helps more when the embeddings are large and prone to overfitting. | not supported |
| 4 | Setting lr = 0.01 raises recall_at_20 less when model = transformer than otherwise | Transformers are unstable at high learning rates, so the edit helps less. | contradicted |
| 5 | Setting lr = 0.001 raises recall_at_20 more when model = transformer than otherwise | A learning rate of 0.001 is near the transformer's optimum, so it gains more. | not supported |
| 6 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | Large batches with a tiny learning rate give too few effective updates. | supported |
| 7 | Setting batch = 256 raises recall_at_20 more when lr = low than otherwise | Small batches give more update steps, compensating for a small learning rate. | supported |
| 8 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = low than otherwise | A small model is capacity-limited, so heavy dropout hurts it more. | contradicted |
| 9 | Setting emb_dim = 128 raises recall_at_20 less when lr = high than otherwise | A high learning rate with larger embeddings is less stable, reducing the gain. | not supported |
| 10 | Setting model = mf raises recall_at_20 more when emb_dim = high than otherwise | Matrix factorization depends mainly on embedding size, so it benefits most from large embeddings. | contradicted |

## measure, run 2 (93 experiments first)

10 hypotheses, 6 supported of 10 testable (precision 0.60)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | Switching gru to transformer gained +4.2/+1.6/+5.3 at lr=0.01, versus +2.1/n/a/-1.2 at lr=0.001. | supported |
| 2 | Setting model = mf raises recall_at_20 more when lr = high than otherwise | Switching to mf was about neutral or positive at lr=0.01 but clearly negative at lr=0.001. | supported |
| 3 | Setting model = mf raises recall_at_20 less when emb_dim = high than otherwise | Switching to mf at emb_dim=256 cost -7.9 on task_c versus about -1.8 at emb_dim=64. | supported |
| 4 | Setting emb_dim = 256 raises recall_at_20 less when lr = high than otherwise | Raising emb_dim to 256 gave smaller gains at lr=0.01 (+0.66/+2.3) than at lower lr (+1.5/+4.9 or more). | not supported |
| 5 | Setting lr = 0.01 raises recall_at_20 more when batch = 4096 than otherwise | lr=0.01 hurt much less with batch=4096 (-0.96/+0.06/-0.48) than with batch=1024 (-3.7/n/a/-2.5). | supported |
| 6 | Setting lr = 0.01 raises recall_at_20 less when emb_dim = high than otherwise | lr=0.01 hurt more with a large embedding (-5.3 on task_c) than at emb_dim=64 (-2.5). | not supported |
| 7 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | Going to batch=4096 gave +3.1/+0.2/+2.4 at lr=0.01 but negative changes at lr=0.001. | supported |
| 8 | Setting emb_dim = 16 raises recall_at_20 less when model = transformer than otherwise | Shrinking to emb_dim=16 hurt the transformer most (-5.7/-1.6/-2.7) compared with gru or mf. | not supported |
| 9 | Setting emb_dim = 256 raises recall_at_20 less when model = mf than otherwise | A larger embedding gave mf slight negative changes in every group, while transformer and gru gained. | supported |
| 10 | Setting batch = 4096 raises recall_at_20 more when emb_dim = low than otherwise | Batch=4096 cost less at emb_dim=16 (-0.37/-2.1/-1.5) than at emb_dim=64 (-1.6/n/a/-2.3), so it is a weak, partly speculative pick. | not supported |

## direct, run 3

9 hypotheses, 4 supported of 9 testable (precision 0.44)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = low than otherwise | Heavy dropout on a tiny embedding removes scarce capacity, so it hurts more than at other sizes. | contradicted |
| 2 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Large embeddings overfit and gain more from moderate regularization. | not supported |
| 3 | Setting lr = 0.01 raises recall_at_20 less when model = transformer than otherwise | Transformers are unstable at high learning rates, so raising lr to 0.01 helps less. | contradicted |
| 4 | Setting model = transformer raises recall_at_20 more when emb_dim = high than otherwise | Transformers exploit larger embedding dimensions better than the other models. | supported |
| 5 | Setting model = mf raises recall_at_20 less when emb_dim = high than otherwise | Matrix factorization saturates with dimension, so it gains less relative to others at 256. | supported |
| 6 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | Few large-batch updates at a tiny learning rate leave the model under-trained. | supported |
| 7 | Setting batch = 256 raises recall_at_20 more when lr = high than otherwise | Small batches add gradient noise that stabilizes a high learning rate. | contradicted |
| 8 | Setting dropout = 0 raises recall_at_20 more when emb_dim = low than otherwise | A low-capacity model underfits and needs no regularization. | not supported |
| 9 | Setting lr = 0.001 raises recall_at_20 more when model = gru than otherwise | GRU models train best at moderate learning rates, so this setting helps more there. | supported |

## measure, run 3 (97 experiments first)

10 hypotheses, 4 supported of 10 testable (precision 0.40)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | Switching gru→transformer gains the most at lr=0.01 (+4.6, +5.5, +6.6 etc.) versus small or mixed gains at lr=0.0001. | supported |
| 2 | Setting model = gru raises recall_at_20 less when lr = high than otherwise | mf/transformer→gru is strongly negative at lr=0.01 (rows 18, 97) but positive at lr=0.0001 (row 17). | supported |
| 3 | Setting lr = 0.01 raises recall_at_20 more when emb_dim = high than otherwise | Raising lr to 0.01 hurts at emb 16/64 but is roughly neutral or positive at emb 256 (rows 26, 29). | not supported |
| 4 | Setting emb_dim = 16 raises recall_at_20 more when lr = high than otherwise | Dropping to emb 16 costs a lot at lr≤0.001 but less at lr=0.01 (row 57 versus rows 54–56, 58). | not supported |
| 5 | Setting model = transformer raises recall_at_20 more when emb_dim = high than otherwise | The transformer gain is largest at emb 256 for mf (+4.8, +2.6, +7.9) and gru at lr=0.01, and weaker at emb 16. | supported |
| 6 | Setting model = transformer raises recall_at_20 less when emb_dim = low than otherwise | At emb 16 the transformer switch gives small or negative gains (rows 10, 2) compared with larger embeddings. | not supported |
| 7 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | Larger batch helps at lr=0.01 (rows 84, 86, 87) but is flat or negative at lr 0.0001 and 0.001. | supported |
| 8 | Setting emb_dim = 256 raises recall_at_20 more when lr = high than otherwise | At lr=0.01 the jump to emb 256 is positive on gru and transformer (rows 43, 49), unlike at lr=0.001 for mf and transformer. | not supported |
| 9 | Setting batch = 4096 raises recall_at_20 less when emb_dim = high than otherwise | Batch 4096 at emb 256 hurts task_c (-3.3), more than at emb 16 or 64. | not supported |
| 10 | Setting lr = 0.01 raises recall_at_20 more when emb_dim = low than otherwise | The lr=0.01 penalty is milder at emb 16 (-4.7, -1.0, -0.8) than at emb 64 (-5.6, -5.1). | not supported |

## direct, run 4

10 hypotheses, 3 supported of 10 testable (precision 0.30)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | A large embedding overfits more, so moderate dropout helps more there. | not supported |
| 2 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = low than otherwise | A small embedding has little spare capacity, so heavy dropout hurts it more. | contradicted |
| 3 | Setting lr = 0.01 raises recall_at_20 less when model = transformer than otherwise | Transformers are unstable at high learning rates, so lr 0.01 is more damaging for them. | contradicted |
| 4 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | Large batches at a tiny lr mean few effective updates, leaving the model undertrained. | supported |
| 5 | Setting batch = 256 raises recall_at_20 more when lr = low than otherwise | Small batches give more update steps, which compensates for a low learning rate. | supported |
| 6 | Setting emb_dim = 256 raises recall_at_20 more when dropout = 0.5 than otherwise | A large embedding offsets the capacity lost to heavy dropout, so it helps more there. | contradicted |
| 7 | Setting emb_dim = 256 raises recall_at_20 less when lr = high than otherwise | Large embeddings with a high learning rate are prone to instability, which reduces the gain from width. | not supported |
| 8 | Setting dropout = 0.1 raises recall_at_20 more when model = transformer than otherwise | Transformers benefit from light regularization more than the other models. | not supported |
| 9 | Setting lr = 0.003 raises recall_at_20 more when batch = 4096 than otherwise | Large batches tolerate and need a higher learning rate to make enough progress. | supported |
| 10 | Setting dropout = 0.3 raises recall_at_20 more when lr = high than otherwise | Regularization stabilizes training at a high learning rate, so dropout helps more there. | not supported |

## measure, run 4 (100 experiments first)

10 hypotheses, 8 supported of 10 testable (precision 0.80)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | Switching gru to transformer gains +5.5/+1.6/+8.4 at lr=0.01 but about 0 to +2 at lower lr. | supported |
| 2 | Setting emb_dim = 256 raises recall_at_20 more when model = gru than otherwise | emb_dim 256 helps gru on most runs (especially task_c, about +5), while mf is flat or negative and transformer is near zero. | supported |
| 3 | Setting emb_dim = 256 raises recall_at_20 less when model = mf than otherwise | mf mostly loses or gains nothing from a larger embedding, unlike gru. | supported |
| 4 | Setting dropout = 0.5 raises recall_at_20 more when emb_dim = low than otherwise | At emb_dim=16 dropout 0.5 gives large gains (up to +6) in all models, versus about 0 at larger emb_dim. | supported |
| 5 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = high than otherwise | At emb_dim=256 high dropout adds little or hurts, because the model is already regularized by its size. | supported |
| 6 | Setting lr = 0.01 raises recall_at_20 more when model = transformer than otherwise | lr 0.01 helps transformer in most runs but usually hurts gru. | supported |
| 7 | Setting model = transformer raises recall_at_20 more when emb_dim = high than otherwise | Transformer's gain over mf is largest at emb_dim=256 (+4.8/+2.6/+7.9) versus about +2 at smaller sizes. | supported |
| 8 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | A large batch pairs better with high lr (gru +3.1/+1.4/+4.5), but hurts at low lr. | supported |
| 9 | Setting model = transformer raises recall_at_20 less when lr = low than otherwise | Transformer's advantage shrinks at lr=0.0001 compared with its large gains at lr=0.01 for gru. | not supported |
| 10 | Setting emb_dim = 256 raises recall_at_20 more when lr = high than otherwise | Larger embeddings pair with higher lr: transformer and gru gain +3 to +5 at lr=0.01 versus small gains at lr=0.0001. | not supported |

## direct, run 5

10 hypotheses, 3 supported of 10 testable (precision 0.30)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting emb_dim = 256 raises recall_at_20 more when model = transformer than otherwise | Transformers can exploit the extra embedding capacity better than mf or gru. | supported |
| 2 | Setting lr = 0.01 raises recall_at_20 less when model = transformer than otherwise | Transformers are unstable at high learning rates. | contradicted |
| 3 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Large embeddings overfit more, so regularization helps more. | not supported |
| 4 | Setting dropout = 0.3 raises recall_at_20 less when emb_dim = low than otherwise | A small model is capacity-limited and gains little from dropout. | not supported |
| 5 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | Large batches with a low lr give too few effective updates and undertrain. | supported |
| 6 | Setting lr = 0.003 raises recall_at_20 more when batch = 4096 than otherwise | Large batches tolerate and need a higher lr. | supported |
| 7 | Setting dropout = 0.5 raises recall_at_20 less when model = mf than otherwise | mf has little need for heavy regularization, so strong dropout hurts it more. | not supported |
| 8 | Setting lr = 0.001 raises recall_at_20 more when emb_dim = low than otherwise | Small models need a higher lr to converge in the same number of steps. | not supported |
| 9 | Setting dropout = 0.1 raises recall_at_20 more when model = gru than otherwise | Recurrent models overfit on sequences, so light dropout helps more. | not supported |
| 10 | Setting emb_dim = 128 raises recall_at_20 less when model = mf than otherwise | mf saturates at smaller dimensions, so extra width adds less. | not supported |

## measure, run 5 (90 experiments first)

10 hypotheses, 6 supported of 10 testable (precision 0.60)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting model = gru raises recall_at_20 less when lr = high than otherwise | Switching to gru loses a lot at lr=0.01 (about -4 to -8 from transformer, -1 to -4 from mf) but is near neutral or positive at lr=0.0001. | supported |
| 2 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | Transformer gains most at lr=0.01 (gru->transformer +6.7/+1.6/+8.4), consistent with it tolerating high lr better. | supported |
| 3 | Setting model = transformer raises recall_at_20 more when emb_dim = high than otherwise | For mf, switching to transformer gives +4.6/+2.6/+7.9 at emb_dim=256 versus small changes at emb_dim=16. | supported |
| 4 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = high than otherwise | Dropout 0.5 helps strongly at emb_dim=16 (up to +6 to +7) but is near zero or negative at emb_dim=256. | supported |
| 5 | Setting dropout = 0.5 raises recall_at_20 more when emb_dim = low than otherwise | Small embeddings with mf and transformer gain +4 to +7 from heavy dropout, much more than larger embeddings. | supported |
| 6 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | At lr=0.01 batch 4096 gives +1.4 to +2.4 on all groups, while at lower lr it is flat or negative. | supported |
| 7 | Setting dropout = 0 raises recall_at_20 more when lr = high than otherwise | Removing dropout is positive at lr=0.01 but negative on all groups at lr=0.0001. | not supported |
| 8 | Setting emb_dim = 16 raises recall_at_20 more when lr = high than otherwise | Shrinking emb_dim hurts less at lr=0.01 than at lr=0.0001, where it costs up to -6. | not supported |
| 9 | Setting model = gru raises recall_at_20 less when emb_dim = high than otherwise | Large embeddings favour attention or mf-style models, so moving to gru looks relatively worse at emb_dim=256. | contradicted |
| 10 | Setting batch = 4096 raises recall_at_20 less when emb_dim = high than otherwise | Large batch hurts gru at emb_dim=256 (-3.3 on task_c) more than at emb_dim=16. | not supported |

## direct, run 6

10 hypotheses, 2 supported of 10 testable (precision 0.20)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting emb_dim = 256 raises recall_at_20 more when model = transformer than otherwise | Transformers benefit more from wider embeddings than mf, which saturates early. | supported |
| 2 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Large embeddings overfit more, so regularization pays off more. | not supported |
| 3 | Setting dropout = 0.3 raises recall_at_20 less when emb_dim = low than otherwise | A small model is capacity-limited, so dropout helps less. | not supported |
| 4 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | Large batches stabilize the noisy gradients at high learning rates. | supported |
| 5 | Setting lr = 0.001 raises recall_at_20 more when model = transformer than otherwise | Transformers need a moderate lr to train well and gain more from it. | not supported |
| 6 | Setting lr = 0.01 raises recall_at_20 less when model = transformer than otherwise | A high lr destabilizes transformer training. | contradicted |
| 7 | Setting emb_dim = 128 raises recall_at_20 less when lr = low than otherwise | Larger embeddings are undertrained at a very low lr, so the gain shrinks. | not supported |
| 8 | Setting dropout = 0.5 raises recall_at_20 less when model = mf than otherwise | mf has little spare capacity, so heavy dropout hurts it more. | not supported |
| 9 | Setting dropout = 0.1 raises recall_at_20 more when model = gru than otherwise | Light dropout regularizes recurrent models, which overfit the sequences. | not supported |
| 10 | Setting batch = 256 raises recall_at_20 more when emb_dim = high than otherwise | Noisy small-batch updates regularize large-embedding models. | not supported |

## measure, run 6 (93 experiments first)

10 hypotheses, 8 supported of 10 testable (precision 0.80)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting emb_dim = 256 raises recall_at_20 less when model = mf than otherwise | Growing the embedding helps gru and transformer but is flat or negative for mf in the measured runs. | supported |
| 2 | Setting lr = 0.01 raises recall_at_20 less when model = gru than otherwise | lr 0.01 was negative on every group for gru while other models showed mixed or positive effects. | supported |
| 3 | Setting dropout = 0.5 raises recall_at_20 more when emb_dim = low than otherwise | Small embeddings gained strongly from heavy dropout, whereas emb_dim 64 saw flat or negative changes. | supported |
| 4 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | Switching gru to transformer gained far more at lr 0.01 than at lr 0.0001. | supported |
| 5 | Setting model = gru raises recall_at_20 less when lr = high than otherwise | Switching mf to gru was positive at lr 0.0001 but negative at lr 0.01. | supported |
| 6 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | Large batch was positive for gru at lr 0.01 but mostly negative at lower learning rates. | supported |
| 7 | Setting dropout = 0.5 raises recall_at_20 more when model = transformer than otherwise | Dropout 0.5 was consistently positive for transformer but mixed or negative for gru and mf. | not supported |
| 8 | Setting model = transformer raises recall_at_20 more when emb_dim = high than otherwise | Transformer's advantage over mf was much larger at emb_dim 256 than at 16. | supported |
| 9 | Setting emb_dim = 256 raises recall_at_20 more when model = gru than otherwise | Growing the embedding gave gru consistent gains while mf saw little or negative change. | supported |
| 10 | Setting emb_dim = 256 raises recall_at_20 more when dropout = 0.5 than otherwise | For gru, emb_dim 256 gained more at dropout 0.5 than at dropout 0.1. | contradicted |

## direct, run 7

7 hypotheses, 2 supported of 7 testable (precision 0.29)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Large embeddings overfit more, so regularisation helps more at emb_dim 256. | not supported |
| 2 | Setting dropout = 0.3 raises recall_at_20 less when emb_dim = low than otherwise | Small embeddings have little capacity to overfit, so dropout helps less or hurts. | not supported |
| 3 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | Large batches with a tiny lr give too few effective updates, so underfitting worsens. | supported |
| 4 | Setting batch = 256 raises recall_at_20 more when lr = low than otherwise | Small batches give more updates, which helps most when lr is tiny. | supported |
| 5 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = low than otherwise | Heavy dropout on a tiny embedding removes needed capacity. | contradicted |
| 6 | Setting dropout = 0.1 raises recall_at_20 more when lr = high than otherwise | Dropout noise partly stabilises high-lr training, so it helps more at lr 0.01. | not supported |
| 7 | Setting batch = 1024 raises recall_at_20 more when lr = high than otherwise | A larger batch reduces gradient noise, which helps more when lr is high. | not supported |

## measure, run 7 (97 experiments first)

10 hypotheses, 7 supported of 10 testable (precision 0.70)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting lr = 0.003 raises recall_at_20 more when batch = 4096 than otherwise | Raising lr to 0.003 gave +3 to +6.6 gains at batch 4096 but roughly flat or negative gains at batch 256. | supported |
| 2 | Setting batch = 256 raises recall_at_20 less when lr = high than otherwise | Small batch at the highest lr was negative on all groups (runs 67, 69, 71), while it was mostly positive at lr 0.001. | supported |
| 3 | Setting model = gru raises recall_at_20 less when lr = high than otherwise | Switching from mf to gru was negative at lr 0.01 but positive at mid and low lr. | supported |
| 4 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | Switching from gru to transformer gave +6.7, +1.6, +5.6 at lr 0.01, larger than at other lrs. | supported |
| 5 | Setting emb_dim = 256 raises recall_at_20 less when dropout = 0.5 than otherwise | Growing the embedding was negative on nearly every group at dropout 0.5 but strongly positive at dropout 0.1. | supported |
| 6 | Setting emb_dim = 256 raises recall_at_20 more when lr = high than otherwise | The embedding gain rose with lr, notably for mf, where it was negative at lr 0.0001 and positive at 0.01. | not supported |
| 7 | Setting batch = 256 raises recall_at_20 more when emb_dim = high than otherwise | Small batch was positive on all groups at emb_dim 256 across gru, mf and transformer. | not supported |
| 8 | Setting lr = 0.003 raises recall_at_20 more when emb_dim = high than otherwise | The lr increase gave +3 to +4.5 at emb_dim 256 versus about +2 at emb_dim 64. | not supported |
| 9 | Setting model = gru raises recall_at_20 more when lr = low than otherwise | Switching to gru from mf was strongly positive at lr 0.0001 but negative at lr 0.01. | supported |
| 10 | Setting emb_dim = 256 raises recall_at_20 less when model = mf than otherwise | The embedding increase helped mf far less than gru or transformer, which gained +2 to +5. | supported |

## direct, run 8

10 hypotheses, 3 supported of 10 testable (precision 0.30)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting lr = 0.01 raises recall_at_20 less when model = transformer than otherwise | Transformers are unstable at very high learning rates, so raising lr to 0.01 hurts them more than GRU or MF. | contradicted |
| 2 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Large embeddings overfit more, so dropout regularization helps more at emb_dim 256. | not supported |
| 3 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = low than otherwise | Heavy dropout on a tiny embedding removes needed capacity, so it helps less at emb_dim 16. | contradicted |
| 4 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | Large batches with a very low lr mean few effective updates and undertraining, which hurts more. | supported |
| 5 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | Large batches have lower gradient noise and tolerate a high lr, so they gain more when lr is high. | supported |
| 6 | Setting emb_dim = 256 raises recall_at_20 less when lr = low than otherwise | Bigger embeddings need more optimization, so they benefit less when lr is very low. | not supported |
| 7 | Setting emb_dim = 256 raises recall_at_20 more when model = mf than otherwise | MF depends on embedding capacity alone, so increasing dimension helps it more than sequence models. | contradicted |
| 8 | Setting lr = 0.003 raises recall_at_20 more when batch = 4096 than otherwise | A larger batch calls for a larger lr, so raising lr to 0.003 helps more at batch 4096. | supported |
| 9 | Setting dropout = 0.3 raises recall_at_20 more when model = transformer than otherwise | Transformers overfit readily on small synthetic data, so dropout helps them more than MF. | not supported |
| 10 | Setting emb_dim = 256 raises recall_at_20 less when lr = high than otherwise | Large embeddings combined with a high lr are prone to instability and overshoot, reducing the gain from larger dimension. | not supported |

## measure, run 8 (90 experiments first)

9 hypotheses, 5 supported of 9 testable (precision 0.56)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting lr = 0.01 raises recall_at_20 less when model = gru than otherwise | Raising lr to 0.01 hurt GRU on all groups (runs 21, 28, 30) while it helped transformer. | supported |
| 2 | Setting lr = 0.01 raises recall_at_20 more when model = transformer than otherwise | Transformer gained on all groups from lr=0.01 (runs 25, 26), unlike GRU. | supported |
| 3 | Setting emb_dim = 256 raises recall_at_20 more when model = transformer than otherwise | emb_dim=256 gave large gains on all groups for transformer (runs 15, 17) but hurt MF (run 14). | supported |
| 4 | Setting emb_dim = 256 raises recall_at_20 less when model = mf than otherwise | MF saturates on embedding size, so 256 gave a smaller or negative gain (runs 13, 14). | supported |
| 5 | Setting batch = 4096 raises recall_at_20 less when model = gru than otherwise | A larger batch hurt GRU on most groups (runs 41, 42, 50) while helping MF at emb 128 (run 44). | not supported |
| 6 | Setting model = transformer raises recall_at_20 more when emb_dim = high than otherwise | Switching to transformer gave big gains on all groups at emb 256 (runs 32, 39). | supported |
| 7 | Setting dropout = 0.5 raises recall_at_20 less when model = transformer than otherwise | Dropout 0.5 gave small or negative changes for transformer but large gains for GRU and MF (runs 51, 53, 55). | not supported |
| 8 | Setting emb_dim = 256 raises recall_at_20 more when lr = low than otherwise | At lr=0.0001 and emb 16, emb 256 gave very large gains (run 11). | not supported |
| 9 | Setting lr = 0.01 raises recall_at_20 more when emb_dim = high than otherwise | At emb 256, lr=0.01 helped (runs 26, 29), whereas at emb 128 or lower it often hurt (runs 28, 30). | not supported |

## direct, run 9

10 hypotheses, 3 supported of 10 testable (precision 0.30)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Large embeddings overfit more, so moderate dropout helps more at emb_dim 256. | not supported |
| 2 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = low than otherwise | Heavy dropout on a tiny 16-dim embedding removes scarce capacity, so it helps less than at larger sizes. | contradicted |
| 3 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | Large batches with a tiny learning rate undertrain, so they help less at lr 0.0001. | supported |
| 4 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | Large batches reduce gradient noise, stabilizing training at lr 0.01. | supported |
| 5 | Setting lr = 0.01 raises recall_at_20 less when model = transformer than otherwise | Transformers are especially unstable at high learning rates, so raising lr helps less for them. | contradicted |
| 6 | Setting emb_dim = 256 raises recall_at_20 less when dropout = 0 than otherwise | Without dropout, a large embedding overfits, so growing emb_dim helps less. | not supported |
| 7 | Setting lr = 0.003 raises recall_at_20 less when batch = 256 than otherwise | Small batches are noisy, so a higher learning rate helps less than with large batches. | supported |
| 8 | Setting lr = 0.001 raises recall_at_20 more when emb_dim = low than otherwise | Small embeddings need a decently high learning rate to converge in the training budget. | not supported |
| 9 | Setting dropout = 0.1 raises recall_at_20 less when batch = 4096 than otherwise | Large batches already regularize implicitly through stable gradients, but offer less noise, so mild dropout helps less here than expected from small batches. | not supported |
| 10 | Setting emb_dim = 128 raises recall_at_20 less when lr = low than otherwise | At a very low learning rate, extra embedding parameters train too slowly to pay off. | not supported |

## measure, run 9 (95 experiments first)

10 hypotheses, 5 supported of 10 testable (precision 0.50)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting emb_dim = 256 raises recall_at_20 less when model = mf than otherwise | Going to 256 hurts or does not help mf on all groups while gru and transformer mostly gain. | supported |
| 2 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | Switching gru to transformer gave the largest gains at lr 0.01 on all three tasks. | supported |
| 3 | Setting model = mf raises recall_at_20 more when lr = high than otherwise | mf is far less bad relative to gru at high lr than at lr 0.001 or 0.0001. | supported |
| 4 | Setting model = mf raises recall_at_20 less when emb_dim = high than otherwise | mf is much worse than gru at emb 256 than at smaller embedding sizes. | supported |
| 5 | Setting emb_dim = 16 raises recall_at_20 more when model = mf than otherwise | Shrinking to 16 hurts mf far less than transformer or gru, consistent with mf not benefiting from large embeddings. | not supported |
| 6 | Setting dropout = 0.5 raises recall_at_20 more when emb_dim = low than otherwise | Dropout 0.5 gave positive changes at emb 16 but mostly negative ones at emb 64. | supported |
| 7 | Setting model = transformer raises recall_at_20 more when dropout = 0.5 than otherwise | The gru-to-transformer gain was largest on all groups at dropout 0.5. | not supported |
| 8 | Setting lr = 0.003 raises recall_at_20 more when model = mf than otherwise | Raising lr from 0.0001 to 0.003 helped mf more than gru or transformer. | not supported |
| 9 | Setting emb_dim = 256 raises recall_at_20 more when dropout = 0.5 than otherwise | Large embeddings gain most when heavily regularized by dropout 0.5. | contradicted |
| 10 | Setting model = transformer raises recall_at_20 more when batch = 256 than otherwise | The transformer gain over gru was larger at batch 256 than at 1024 or 4096 on the groups measured. | not supported |

## direct, run 10

10 hypotheses, 3 supported of 10 testable (precision 0.30)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting emb_dim = 256 raises recall_at_20 more when dropout = 0.5 than otherwise | Dropout regularizes the larger capacity, so overfitting from a big embedding is reduced. | contradicted |
| 2 | Setting lr = 0.01 raises recall_at_20 less when model = transformer than otherwise | Transformers are unstable at high learning rates. | contradicted |
| 3 | Setting lr = 0.003 raises recall_at_20 more when batch = 4096 than otherwise | Large batches need larger learning rates to compensate for fewer updates. | supported |
| 4 | Setting dropout = 0.5 raises recall_at_20 less when emb_dim = low than otherwise | A small model is capacity-limited and heavy dropout causes underfitting. | contradicted |
| 5 | Setting model = transformer raises recall_at_20 more when emb_dim = high than otherwise | Transformers exploit large embedding capacity better than gru or mf. | supported |
| 6 | Setting batch = 4096 raises recall_at_20 less when lr = low than otherwise | Few updates at a tiny learning rate leads to undertraining. | supported |
| 7 | Setting lr = 0.001 raises recall_at_20 more when emb_dim = low than otherwise | Small embeddings tolerate and benefit from faster learning. | not supported |
| 8 | Setting dropout = 0.3 raises recall_at_20 more when emb_dim = high than otherwise | Larger embeddings overfit more and benefit from regularization. | not supported |
| 9 | Setting emb_dim = 128 raises recall_at_20 less when lr = high than otherwise | High learning rates destabilize larger embeddings. | not supported |
| 10 | Setting model = mf raises recall_at_20 less when lr = low than otherwise | Matrix factorization converges slowly at a tiny learning rate. | not supported |

## measure, run 10 (100 experiments first)

10 hypotheses, 8 supported of 10 testable (precision 0.80)

| # | hypothesis | why (model) | verdict |
|---|---|---|---|
| 1 | Setting lr = 0.01 raises recall_at_20 more when model = transformer than otherwise | Raising lr to 0.01 helps the transformer (mostly positive) but hurts GRU by 3-5 points. | supported |
| 2 | Setting lr = 0.01 raises recall_at_20 less when model = gru than otherwise | GRU is consistently harmed by lr=0.01 on every task, unlike the other models. | supported |
| 3 | Setting model = transformer raises recall_at_20 more when lr = high than otherwise | Switching to the transformer gains most over the alternatives at lr=0.01 (e.g. +5 to +6 over GRU). | supported |
| 4 | Setting model = transformer raises recall_at_20 less when lr = low than otherwise | At the lowest lr the transformer advantage shrinks or reverses compared with other learning rates. | not supported |
| 5 | Setting batch = 4096 raises recall_at_20 more when lr = high than otherwise | Large batches gain at high lr (positive for all models) but lose at lr=0.001. | supported |
| 6 | Setting batch = 256 raises recall_at_20 less when lr = high than otherwise | Going from batch 4096 to 256 hurts at lr=0.01 on all models, while it helps at lower lr. | supported |
| 7 | Setting dropout = 0.5 raises recall_at_20 more when emb_dim = low than otherwise | Strong dropout gives its largest gains for small embeddings, up to +6 to +7 for the transformer. | supported |
| 8 | Setting model = mf raises recall_at_20 less when emb_dim = high than otherwise | Replacing the transformer with MF loses much more at emb_dim=256 (down to -9) than at emb_dim=16. | supported |
| 9 | Setting lr = 0.01 raises recall_at_20 more when emb_dim = high than otherwise | High lr is better tolerated with large embeddings, notably MF and the transformer turning positive. | not supported |
| 10 | Setting emb_dim = 256 raises recall_at_20 less when model = mf than otherwise | Enlarging embeddings is mostly negative for MF but positive for GRU and the transformer. | supported |
