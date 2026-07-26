"""Streamlit UI for the 6-agent Agentic RAG pipeline (proposal §6.2.3, §6.3).

Shows every agent's output as it happens - structured query, fused
retrieval hits, deduplicated context, reasoning insights, response, and the
evaluator's verdict - so the pipeline's behavior is easy to inspect, not
just its final answer. A pure HTTP client: all data comes from the FastAPI
service in `src/football_rag/api/main.py` via `scripts/_api_client.py` -
this file has no direct `football_rag` pipeline imports.

Usage:
    uv run python scripts/run_api.py       # in one terminal
    uv run streamlit run scripts/app.py    # in another
"""

from __future__ import annotations

import streamlit as st

import _api_client

DEFAULT_MATCH_ID = 3869685  # 2022 FIFA World Cup Final: Argentina vs France
DEFAULT_QUERY = "Summarize the match and provide key tactical insights."


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


def main() -> None:
    st.set_page_config(page_title="Football Agentic RAG", layout="wide")
    st.title("Football Agentic RAG — pipeline walkthrough")
    st.caption(
        "6-agent pipeline (proposal §6.2.3/§6.3): Query Understanding → Retriever → "
        "Context Aggregator → Reasoning → Generator → Evaluator"
    )

    try:
        match_id = render_match_picker()
    except (RuntimeError, ConnectionError) as e:
        st.error(f"Couldn't reach the API at {_api_client.API_BASE_URL}: {e}")
        st.info("Start it with: `uv run python scripts/run_api.py`")
        return

    query = st.sidebar.text_area("Query", value=DEFAULT_QUERY)

    with st.sidebar.expander("Advanced settings"):
        enable_evaluator = st.checkbox("Enable Evaluator agent", value=True)
        max_retries = st.number_input("Max retries on evaluator fail", min_value=0, max_value=5, value=1)

    if st.sidebar.button("Run pipeline", type="primary"):
        try:
            with st.spinner("Running the 6-agent pipeline (first run per match also ingests + embeds)..."):
                result = _api_client.run_pipeline(
                    match_id, query, enable_evaluator=enable_evaluator, max_retries=int(max_retries)
                )
            st.session_state["result"] = result
            st.session_state["result_match_label"] = st.session_state["selected_match_label"]
            st.session_state["result_query"] = query
        except (RuntimeError, ConnectionError) as e:
            st.error(str(e))

    result = st.session_state.get("result")
    if not result:
        st.info("Pick a match and query in the sidebar, then click **Run pipeline**.")
        return

    st.subheader(f"Match: {st.session_state['result_match_label']}")
    st.write(f"Query: {st.session_state['result_query']}")
    if result.get("output_file"):
        st.caption(f"Saved to {result['output_file']} (on the machine running the API)")

    with st.expander("1. Query Understanding", expanded=True):
        st.json(result["structured_query"])
        st.write("where filter:", result.get("where_filter"))

    with st.expander("2. Retriever — fused dense + BM25 hits", expanded=True):
        hits = result["retrieved_chunks"]
        st.dataframe(
            [{"chunk_id": h["chunk_id"], "fusion_score": round(h["fusion_score"], 4), "text": h["text"]} for h in hits],
            use_container_width=True,
        )

    with st.expander("3. Context Aggregator", expanded=True):
        st.write(f"{len(result['aggregated_context'])} deduplicated chunks kept (from {len(hits)} retrieved)")

    with st.expander("4. Reasoning insights", expanded=True):
        for insight in result["reasoning_insights"]:
            st.markdown(f"- {insight['insight']}  \n  _evidence: {', '.join(insight['supporting_chunk_ids'])}_")

    st.subheader("5. Generated response")
    st.markdown(result["response_text"])

    evaluation = result.get("evaluation")
    if evaluation is not None:
        st.subheader("6. Evaluator verdict")
        cols = st.columns(4)
        cols[0].metric("Passed", "Yes" if evaluation["passed"] else "No")
        cols[1].metric("Relevance", f"{evaluation['relevance']:.2f}")
        cols[2].metric("Coherence", f"{evaluation['coherence']:.2f}")
        cols[3].metric("Factual consistency", f"{evaluation['factual_consistency']:.2f}")
        st.caption(evaluation["feedback"])


main()
