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
| minilm | 5 | 0.440 | 0.283 | 15 |
| minilm | 10 | 0.307 | 0.339 | 15 |
| e5 | 5 | 0.440 | 0.273 | 15 |
| e5 | 10 | 0.287 | 0.316 | 15 |

## Ablation study (stage 12)

| config | precision_at_5 | rougeL | n_rows |
|---|---|---|---|
| full | 0.427 | 0.122 | 15 |
| no_evaluator | 0.427 | 0.131 | 15 |
| no_hybrid | 0.427 | 0.142 | 15 |
| no_reasoning | 0.427 | 0.129 | 15 |
| no_reranker | 0.427 | 0.122 | 15 |

## Human evaluation (stage 11)

| system | relevance | coherence | factual_consistency | fluency |
|---|---|---|---|---|
| agentic | 4.400 | 4.633 | 4.200 | 4.533 |
| naive | 2.617 | 4.300 | 2.767 | 4.650 |

## Error analysis (stage 12)

40 worst-case rows saved to `worst_cases.csv` - see that file for qualitative discussion material (not reproduced here, response texts are long).
