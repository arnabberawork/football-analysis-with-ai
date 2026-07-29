"""Streamlit UI for the 6-agent Agentic RAG pipeline (proposal §6.2.3, §6.3).

Shows the whole pipeline as a flow diagram up front - so someone unfamiliar
with the architecture can see the shape of it before clicking anything -
then lights up box by box, live, as each agent completes (Query
Understanding -> Retriever -> Context Aggregator -> Reasoning -> Generator
-> Evaluator, or the StatsAgent shortcut for counting questions), with a
detailed per-stage log alongside it. The generated response is shown by
default with the full stage-by-stage audit trail one tab away. A pure HTTP
client: all data comes from the FastAPI service in
`src/football_rag/api/main.py` via `scripts/_api_client.py` - this file has
no direct `football_rag` pipeline imports.

Usage:
    uv run python scripts/run_api.py       # in one terminal
    uv run streamlit run scripts/app.py    # in another
"""

from __future__ import annotations

import streamlit as st

import _api_client

DEFAULT_MATCH_ID = 3869685  # 2022 FIFA World Cup Final: Argentina vs France
DEFAULT_QUERY = "Summarize the match and provide key tactical insights."

# (node_id, label) per LangGraph node (agents/graph.py) - node_id is the
# short identifier used inside the Mermaid diagram source.
PIPELINE_NODES: dict[str, tuple[str, str]] = {
    "query_understanding": ("QU", "Query Understanding"),
    "retriever": ("RET", "Retriever"),
    "context_aggregator": ("CA", "Context Aggregator"),
    "reasoning": ("RS", "Reasoning"),
    "generator": ("GEN", "Generator"),
    "evaluator": ("EV", "Evaluator"),
    "stats_node": ("STATS", "StatsAgent"),
}

# (source, target, edge label or None for a plain arrow) - mirrors
# agents/graph.py's conditional edges (route_after_query_understanding,
# route_after_evaluator) closely enough for a progress diagram, without
# being the actual control flow itself.
PIPELINE_EDGES: list[tuple[str, str, str | None]] = [
    ("query_understanding", "retriever", None),
    ("query_understanding", "stats_node", "counting question"),
    ("retriever", "context_aggregator", None),
    ("context_aggregator", "reasoning", None),
    ("reasoning", "generator", None),
    ("generator", "evaluator", None),
    ("evaluator", "generator", "needs revision"),
]


def _initial_stage_status(enable_evaluator: bool) -> dict[str, str]:
    """Fresh status map for a new run - Query Understanding always goes
    first; which branch it leads to isn't known until it completes (see
    `next_stage_hint()` server-side), so both `retriever` and `stats_node`
    start "pending" rather than one being pre-selected. `evaluator` is
    pre-marked "skipped" when disabled since the UI already knows that
    without waiting for a server hint."""
    status = {key: "pending" for key in PIPELINE_NODES}
    status["query_understanding"] = "running"
    if not enable_evaluator:
        status["evaluator"] = "skipped"
    return status


def _build_pipeline_diagram(stage_status: dict[str, str]) -> str:
    """Renders the agent pipeline as a Mermaid flowchart, colored by each
    node's current status - lets someone unfamiliar with the pipeline see
    its whole shape immediately, then watch it light up stage by stage,
    instead of only reading a scrolling text log."""
    lines = ["graph LR"]
    for node_key, (node_id, label) in PIPELINE_NODES.items():
        lines.append(f'    {node_id}["{label}"]')
    for src, dst, edge_label in PIPELINE_EDGES:
        src_id, dst_id = PIPELINE_NODES[src][0], PIPELINE_NODES[dst][0]
        if edge_label:
            lines.append(f"    {src_id} -.->|{edge_label}| {dst_id}")
        else:
            lines.append(f"    {src_id} --> {dst_id}")

    lines.append("    classDef pending fill:#30363d,stroke:#6e7681,color:#8b949e,stroke-width:1px;")
    lines.append("    classDef running fill:#1f6feb,stroke:#58a6ff,color:#ffffff,stroke-width:2px;")
    lines.append("    classDef complete fill:#238636,stroke:#3fb950,color:#ffffff,stroke-width:2px;")
    lines.append("    classDef skipped fill:#161b22,stroke:#30363d,color:#484f58,stroke-width:1px,stroke-dasharray:3 3;")
    lines.append("    classDef error fill:#da3633,stroke:#f85149,color:#ffffff,stroke-width:2px;")
    for node_key, (node_id, _label) in PIPELINE_NODES.items():
        lines.append(f"    class {node_id} {stage_status.get(node_key, 'pending')};")
    return "\n".join(lines)


@st.cache_data(show_spinner="Loading competitions...")
def get_competitions() -> list[dict]:
    return _api_client.get_competitions()


@st.cache_data(show_spinner="Loading matches...")
def get_matches(competition_id: int, season_id: int) -> list[dict]:
    return _api_client.get_matches(competition_id, season_id)


def render_match_picker() -> int:
    competitions = get_competitions()

    comp_names = sorted({c["competition_name"] for c in competitions})
    default_comp_index = comp_names.index("FIFA World Cup") if "FIFA World Cup" in comp_names else 0
    selected_comp = st.sidebar.selectbox("Competition", comp_names, index=default_comp_index)

    seasons = [c for c in competitions if c["competition_name"] == selected_comp]
    season_names = [c["season_name"] for c in seasons]
    default_season_index = season_names.index("2022") if "2022" in season_names else 0
    selected_season_name = st.sidebar.selectbox("Season", season_names, index=default_season_index)

    selected_season = next(c for c in seasons if c["season_name"] == selected_season_name)
    matches = sorted(
        get_matches(selected_season["competition_id"], selected_season["season_id"]),
        key=lambda m: m["match_date"],
    )
    if not matches:
        st.sidebar.warning("No matches available for this competition/season.")
        st.stop()

    labels = [
        f"{m['match_date']}: {m['home_team']['home_team_name']} {m['home_score']}-{m['away_score']} "
        f"{m['away_team']['away_team_name']}"
        for m in matches
    ]
    match_ids = [m["match_id"] for m in matches]
    default_match_index = match_ids.index(DEFAULT_MATCH_ID) if DEFAULT_MATCH_ID in match_ids else 0
    selected_label = st.sidebar.selectbox("Match", labels, index=default_match_index)

    st.session_state["selected_match_label"] = selected_label
    return match_ids[labels.index(selected_label)]


def run_pipeline_with_live_status(
    diagram_slot, ingest_slot, match_id: int, query: str, enable_evaluator: bool, max_retries: int
) -> None:
    """Streams `/pipeline/run/stream`, updating the pipeline diagram (via
    `diagram_slot`, created by the caller so it stays in the same spot on
    screen) and a detailed text log (`st.status()`) as each agent
    completes - two views of the same live progress, one visual and
    at-a-glance, one worded and detailed.
    """
    stage_status = st.session_state["stage_status"]

    with st.status("Starting the agentic pipeline...", expanded=True) as status:
        try:
            for event in _api_client.stream_pipeline(
                match_id, query, enable_evaluator=enable_evaluator, max_retries=max_retries
            ):
                if event["event"] == "done":
                    for key, value in stage_status.items():
                        if value == "running":
                            stage_status[key] = "complete"
                    diagram_slot.mermaid_chart(_build_pipeline_diagram(stage_status), width="stretch")
                    st.session_state["result"] = event["result"]
                    st.session_state["result_match_label"] = st.session_state["selected_match_label"]
                    st.session_state["result_query"] = query
                    status.update(label="Pipeline complete", state="complete")
                    continue

                stage, next_stage = event["stage"], event.get("next")

                if stage == "ingest":
                    ingest_slot.caption(f":material/cloud_download: {event['detail']}")
                    status.update(label="Ingesting match data...")
                    continue
                ingest_slot.empty()

                if stage in stage_status:
                    stage_status[stage] = "complete"

                if stage == "query_understanding":
                    # Intent is now known - gray out whichever branch (StatsAgent
                    # vs. the full retrieval chain) this run did NOT take.
                    if next_stage == "retriever":
                        skip_chain = ["stats_node"]
                    else:
                        skip_chain = ["retriever", "context_aggregator", "reasoning", "generator", "evaluator"]
                    for key in skip_chain:
                        if stage_status.get(key) != "complete":
                            stage_status[key] = "skipped"

                if next_stage and stage_status.get(next_stage) not in ("skipped", "complete"):
                    stage_status[next_stage] = "running"
                elif next_stage == "generator":
                    # Evaluator-fail retry loop: Generator is already
                    # "complete" from its first pass, but it's about to run
                    # again, so light it back up.
                    stage_status["generator"] = "running"

                diagram_slot.mermaid_chart(_build_pipeline_diagram(stage_status), width="stretch")
                # event["label"] is the stage that just finished (now green) - the
                # status line should name whichever stage is now running (now blue),
                # i.e. next_stage, not the one that just completed.
                if next_stage and stage_status.get(next_stage) == "running":
                    next_label = PIPELINE_NODES.get(next_stage, (None, next_stage))[1]
                    status.update(label=f"Running: {next_label}...")
                else:
                    status.update(label="Finishing up...")
                st.write(f":material/check_circle: **{event['label']}** — {event['detail']}")
        except (RuntimeError, ConnectionError) as e:
            for key, value in stage_status.items():
                if value == "running":
                    stage_status[key] = "error"
            diagram_slot.mermaid_chart(_build_pipeline_diagram(stage_status), width="stretch")
            status.update(label="Pipeline failed", state="error")
            st.error(str(e))


def render_response_tab(result: dict) -> None:
    st.markdown(result["response_text"])

    evaluation = result.get("evaluation")
    if evaluation is None:
        return

    st.space("small")
    if evaluation["passed"]:
        st.badge("Passed review", icon=":material/check_circle:", color="green")
    else:
        st.badge("Flagged for revision", icon=":material/warning:", color="orange")

    cols = st.columns(3)
    cols[0].metric("Relevance", f"{evaluation['relevance']:.2f}")
    cols[1].metric("Coherence", f"{evaluation['coherence']:.2f}")
    cols[2].metric("Factual consistency", f"{evaluation['factual_consistency']:.2f}")
    st.caption(evaluation["feedback"])


def render_audit_tab(result: dict) -> None:
    is_statistic = result["structured_query"]["intent"] == "statistic"

    with st.expander("1. Query understanding", icon=":material/quiz:", expanded=True):
        st.json(result["structured_query"])
        st.write("Where filter:", result.get("where_filter"))

    if is_statistic:
        st.caption(
            ":material/info: This was a counting/statistic question — answered directly by StatsAgent "
            "from the full event list, bypassing retrieval, reasoning, and generation entirely."
        )
        return

    with st.expander("2. Retriever — fused dense + BM25 hits", icon=":material/search:", expanded=True):
        hits = result["retrieved_chunks"]
        st.dataframe(
            [
                {
                    "chunk_id": h["chunk_id"],
                    # ensure_salient-forced hits (agents/retriever.py::_force_salient)
                    # carry no score at all - they bypassed ranking entirely. Always a
                    # str (never a raw float) so this column stays a single dtype -
                    # a mixed float/"forced" column made PyArrow's dataframe
                    # serialization fail with ArrowTypeError on every run that
                    # included a forced hit.
                    "score": str(round(h["rerank_score"], 4)) if "rerank_score" in h else (
                        str(round(h["fusion_score"], 4)) if "fusion_score" in h else "forced"
                    ),
                    "text": h["text"],
                }
                for h in hits
            ],
            width="stretch",
        )

    with st.expander("3. Context aggregator", icon=":material/filter_alt:"):
        st.write(f"{len(result['aggregated_context'])} deduplicated chunks kept (from {len(hits)} retrieved)")

    with st.expander("4. Reasoning insights", icon=":material/psychology:"):
        insights = result["reasoning_insights"]
        if not insights:
            st.caption("No reasoning insights (the reasoning stage was skipped for this run).")
        for insight in insights:
            st.markdown(f"- {insight['insight']}  \n  _evidence: {', '.join(insight.get('supporting_chunk_ids') or [])}_")


def main() -> None:
    st.set_page_config(page_title="Football Agentic RAG", page_icon=":material/sports_soccer:", layout="wide")
    st.title("Football Agentic RAG")
    st.caption(
        "6-agent pipeline: Query Understanding → Retriever → Context Aggregator → "
        "Reasoning → Generator → Evaluator (counting questions bypass retrieval via StatsAgent)"
    )

    try:
        match_id = render_match_picker()
    except (RuntimeError, ConnectionError) as e:
        st.error(f"Couldn't reach the API at {_api_client.API_BASE_URL}: {e}")
        st.info("Start it with: `uv run python scripts/run_api.py`")
        return

    query = st.sidebar.text_area("Query", value=DEFAULT_QUERY)

    with st.sidebar.expander("Advanced settings", icon=":material/tune:"):
        enable_evaluator = st.checkbox("Enable evaluator agent", value=True)
        max_retries = st.number_input("Max retries on evaluator fail", min_value=0, max_value=5, value=1)

    if "stage_status" not in st.session_state:
        st.session_state["stage_status"] = _initial_stage_status(enable_evaluator)

    diagram_slot = st.empty()
    diagram_slot.mermaid_chart(_build_pipeline_diagram(st.session_state["stage_status"]), width="stretch")
    ingest_slot = st.empty()

    if st.sidebar.button("Run pipeline", type="primary", icon=":material/play_arrow:"):
        st.session_state["stage_status"] = _initial_stage_status(enable_evaluator)
        diagram_slot.mermaid_chart(_build_pipeline_diagram(st.session_state["stage_status"]), width="stretch")
        run_pipeline_with_live_status(diagram_slot, ingest_slot, match_id, query, enable_evaluator, int(max_retries))

    result = st.session_state.get("result")
    if not result:
        st.info("Pick a match and query in the sidebar, then click **Run pipeline**.")
        return

    st.subheader(f"Match: {st.session_state['result_match_label']}")
    st.caption(f"Query: {st.session_state['result_query']}")
    if result.get("output_file"):
        st.caption(f":material/save: Saved to {result['output_file']} (on the machine running the API)")

    response_tab, audit_tab = st.tabs([":material/summarize: Response", ":material/manage_search: Audit trail"])
    with response_tab:
        render_response_tab(result)
    with audit_tab:
        render_audit_tab(result)


main()
