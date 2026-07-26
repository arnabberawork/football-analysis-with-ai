"""Generation metrics (proposal §6.11.3).

ROUGE and BLEU are always available (lightweight, no model download).
BERTScore is lazy-imported - its default model is a large transformer
(~1GB+ download on first use) which the resource-efficiency goal (§6.13)
says not to force on every harness run, so it's only paid for if
`bertscore()` is actually called.
"""

from __future__ import annotations

import sacrebleu
from rouge_score import rouge_scorer

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def rouge_scores(candidate: str, reference: str) -> dict[str, float]:
    """F-measure for each of rouge1/rouge2/rougeL."""
    scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {name: result.fmeasure for name, result in scores.items()}


def bleu_score(candidate: str, reference: str) -> float:
    return sacrebleu.sentence_bleu(candidate, [reference]).score


def bertscore(candidates: list[str], references: list[str]) -> dict[str, float]:
    """Precision/recall/F1 averaged over the batch. Lazy-imports
    `bert_score`, which downloads a large transformer model on first call -
    only invoke this if you specifically want BERTScore."""
    from bert_score import score as bert_score_fn

    precision, recall, f1 = bert_score_fn(candidates, references, lang="en")
    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }
