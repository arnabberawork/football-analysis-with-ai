"""Structured prompt templates (proposal §6.10).

Explicitly separating system role, user query, evidence, and task
instructions keeps generation grounded in retrieved evidence and reduces
hallucinated claims, per the proposal's response-generation design.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert football analyst. Use only the provided evidence to "
    "generate the response and avoid unsupported information."
)

TASK_INSTRUCTIONS = (
    "Task:\n"
    "- Generate a concise summary of the match (150-200 words).\n"
    "- Identify key moments with timestamps.\n"
    "- Provide relevant tactical insights.\n"
    '- If the required information is not available in the evidence, respond with "Not enough information."'
)


def build_user_prompt(user_query: str, evidence: list[dict]) -> str:
    """Assemble the user-turn prompt from the query and retrieved chunks.

    `evidence` items are the dicts returned by ChunkVectorStore.query
    (each with `chunk_id` and `text`). Referencing chunk_ids in the prompt
    lets the generated output be traced back to supporting evidence.
    """
    evidence_lines = "\n".join(f"[{item['chunk_id']}] {item['text']}" for item in evidence)
    return (
        f"User request: {user_query}\n\n"
        f"Contextual Evidence:\n{evidence_lines}\n\n"
        f"{TASK_INSTRUCTIONS}"
    )


def build_agentic_user_prompt(user_query: str, evidence: list[dict], insights: list) -> str:
    """Same evidence-block structure as `build_user_prompt`, plus an
    Analytical Insights section carrying the Reasoning Agent's structured
    findings (proposal §6.3.4/§6.3.5) - used by the agentic pipeline's
    Generator Agent, not the naive baseline.
    """
    evidence_lines = "\n".join(f"[{item['chunk_id']}] {item['text']}" for item in evidence)
    insight_lines = "\n".join(f"- {item.insight} (evidence: {', '.join(item.supporting_chunk_ids or [])})" for item in insights)
    return (
        f"User request: {user_query}\n\n"
        f"Contextual Evidence:\n{evidence_lines}\n\n"
        f"Analytical Insights:\n{insight_lines}\n\n"
        f"{TASK_INSTRUCTIONS}"
    )
