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
    "- Directly answer the user's request above, using only the provided evidence.\n"
    "- If the request asks for an overall match summary, structure the answer as a concise "
    "150-200 word narrative, key moments with timestamps, and tactical insights.\n"
    "- For a narrower request (e.g. a specific count, a single player, or a specific phase of "
    "the match), answer only what was asked - do not force in an unrelated full match summary "
    "just because evidence for one happens to be present.\n"
    "- If the required information is not available in the evidence, do not guess or invent an "
    "answer. Say so in one sentence, then briefly note that this system only has access to "
    "on-pitch match-event data (goals, shots, passes, duels, cards, substitutions, and their "
    "timings) - not external context like official awards, weather, or officiating decisions. "
    "Close with one specific alternative question about this match, grounded in a real player, "
    "team, or event actually named in the evidence above, that the evidence could answer instead."
)
"""Confirmed live this needed to be query-adaptive, not a one-size-fits-all
match-summary template: NaiveRAGGenerator previously used this exact
instruction unconditionally for every query, including narrow factual ones
("how many cards were shown") where all the needed evidence was present in
the top-10 retrieved chunks - the model still answered "Not enough
information", apparently because it judged the fixed instruction ("write a
150-200 word summary with tactical insights") unanswerable from
cards-only evidence, even though the actual question asked was answerable.

The "not available" fallback used to be a bare literal "Not enough
information." - confirmed live (asking "Who was the Man of the Match?",
which StatsBomb's open-data genuinely never records - see TESTING.md) that
this was unhelpful: a user has no way to tell "wrong question for this
data" apart from "the system is broken". Now it's phrased as an
instruction rather than a fixed string so the model still grounds its
"try this instead" suggestion in whatever real evidence it was actually
given (a real player/team/event from this match), rather than a generic
suggestion applicable to any match.

Applies to both `build_user_prompt` (naive) and `build_agentic_user_prompt`
(agentic Generator) since both share this constant, keeping the naive-vs
-agentic prompt comparison identical per §6.10's fairness requirement."""


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
