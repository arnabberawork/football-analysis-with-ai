# Thesis results tables

## Naive vs Agentic - retrieval + generation metrics (stages 9/10/13)

| system | precision_at_5 | recall_at_5 | precision_at_10 | recall_at_10 | rouge1 | rouge2 | rougeL | bleu | bertscore_f1 | n_queries |
|---|---|---|---|---|---|---|---|---|---|---|
| agentic | 0.763 | 0.458 | 0.482 | 0.461 | 0.181 | 0.039 | 0.132 | 2.099 | 0.818 | 61 |
| naive | 0.357 | 0.303 | 0.231 | 0.349 | 0.140 | 0.036 | 0.115 | 1.511 | 0.828 | 61 |

_Note: retrieval metrics (P@K/R@K) are averaged only over rows where retrieval actually ran - 36 agentic row(s) routed to StatsAgent (exact counts, no retrieval by design) are excluded from the retrieval-metric average but still included in generation-metric averages and n_queries._

## Embedding model comparison - MiniLM vs E5 (stage 3)

| embedder | k | mean_precision | mean_recall | n_queries |
|---|---|---|---|---|
| minilm | 5 | 0.480 | 0.149 | 15 |
| minilm | 10 | 0.387 | 0.193 | 15 |
| e5 | 5 | 0.493 | 0.157 | 15 |
| e5 | 10 | 0.373 | 0.205 | 15 |

## Ablation study

_Not yet available - run `experiment/run_full_ablation.py`._

## Human evaluation

_Not yet available - run `experiment/human_eval_sample.py` then `human_eval_analyze.py`._

## Error analysis (stage 12)

12 worst-case rows saved to `worst_cases.csv` - see that file for qualitative discussion material (not reproduced here, response texts are long).
