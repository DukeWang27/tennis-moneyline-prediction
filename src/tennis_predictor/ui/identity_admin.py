from __future__ import annotations

import streamlit as st

from tennis_predictor.storage import IdentityStore
from tennis_predictor.ui.data import load_app_config


def render_identity_review() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">DATA QUALITY CONTROL</div>
          <h1>Alias Review Queue</h1>
          <p>Resolve unmatched Polymarket or historical names to canonical player IDs with source-specific mappings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    config = load_app_config()
    store = IdentityStore(config.history.identity_database_path)
    pending = store.pending_reviews(limit=200)
    st.metric("Pending aliases", len(pending))
    if not pending:
        st.success("No unresolved aliases are waiting for review.")
        return

    index = st.number_input(
        "Queue position",
        min_value=1,
        max_value=len(pending),
        value=1,
        step=1,
    )
    item = pending[int(index) - 1]
    st.subheader(str(item["raw_name"]))
    st.caption(
        f"Source: {item['source']} · Tour: {item['tour'] or 'unknown'} · "
        f"Suggested: {item.get('suggested_name') or 'none'} · "
        f"Score: {item.get('fuzzy_score') or 'n/a'}"
    )

    query = st.text_input("Search canonical players", value=str(item["raw_name"]))
    candidates = store.search_players(query, tour=str(item.get("tour") or ""), limit=50)
    if not candidates and query:
        candidates = store.search_players("", tour=str(item.get("tour") or ""), limit=50)
    if not candidates:
        st.warning("No candidate player IDs are available. Run the historical backfill first.")
        return

    labels = {
        f"{row['canonical_name']} · ID {row['canonical_player_id']} · {row['tour']}": row
        for row in candidates
    }
    selected = st.selectbox("Map alias to", list(labels))
    row = labels[selected]
    if st.button("Confirm alias mapping", type="primary", use_container_width=True):
        store.resolve_review(int(item["review_id"]), int(row["canonical_player_id"]))
        st.success(
            f"Mapped {item['raw_name']} to {row['canonical_name']} "
            f"(canonical ID {row['canonical_player_id']})."
        )
        st.rerun()
