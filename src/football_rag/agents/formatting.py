"""Human-readable rendering of `graph.run_agentic_pipeline()`'s result
dict - shared by `scripts/run_agentic_pipeline.py`'s console output and
`api/main.py`'s saved-output-file text, so both stay in sync with one
source of truth for the format instead of drifting apart.
"""

from __future__ import annotations


def format_pipeline_result(match_id: str, query: str, result: dict) -> str:
    lines = [f"Match: {match_id}", f"Query: {query}", ""]

    lines.append("1. Query Understanding:")
    lines.append(f"   {result['structured_query']!r}")
    lines.append(f"   where filter: {result.get('where_filter')}")

    lines.append("")
    lines.append("2. Retriever (fused dense + BM25 hits, re-ranked):")
    for hit in result["retrieved_chunks"]:
        score = hit.get("rerank_score", hit.get("fusion_score"))
        # ensure_salient-forced hits (agents/retriever.py::_force_salient) carry
        # no score at all - they bypassed ranking entirely, so there's nothing
        # numeric to show.
        score_str = f"{score:.4f}" if score is not None else "forced"
        lines.append(f"   [{hit['chunk_id']}] ({score_str}) {hit['text']}")

    lines.append("")
    lines.append(f"3. Context Aggregator ({len(result['aggregated_context'])} deduplicated chunks kept)")

    lines.append("")
    lines.append("4. Reasoning insights:")
    for insight in result.get("reasoning_insights", []):
        lines.append(f"   - {insight.insight} (evidence: {', '.join(insight.supporting_chunk_ids or [])})")

    lines.append("")
    lines.append("5. Generated response:")
    lines.append("")
    lines.append(result["response_text"])

    evaluation = result.get("evaluation")
    if evaluation is not None:
        lines.append("")
        lines.append("6. Evaluator verdict:")
        lines.append(
            f"   passed={evaluation.passed} relevance={evaluation.relevance:.2f} "
            f"coherence={evaluation.coherence:.2f} factual_consistency={evaluation.factual_consistency:.2f}"
        )
        lines.append(f"   feedback: {evaluation.feedback}")

    return "\n".join(lines)
