from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from locallens.service import LocalLensService

MUNI_URL = "https://www.sfmta.com/muni"
MUNIMOBILE_URL = "https://www.sfmta.com/munimobile"
CLIPPER_URL = "https://mtc.ca.gov/operations/traveler-services/clipper"
BAY_WHEELS_URL = "https://mtc.ca.gov/operations/traveler-services/bay-wheels-bike-share-program"


def enrich_tip_text(tip: str, response) -> str:
    location = ""
    if response.citations:
        location = response.citations[0].get("location", "")
    lowered = tip.lower().strip()

    if "muni" in lowered and "website" in lowered:
        return (
            f"Use [Muni Trip Planner]({MUNI_URL}) or "
            f"[MuniMobile]({MUNIMOBILE_URL}) to plan your route."
        )
    if "clipper" in lowered:
        return f"Consider getting a [Clipper card]({CLIPPER_URL}) for convenient Bay Area travel."
    if "bike-share" in lowered or "bike share" in lowered:
        if location in {"San Francisco", "San Jose", "Oakland", "Berkeley"}:
            return (
                f"If you want a first/last-mile option, use "
                f"[Bay Wheels]({BAY_WHEELS_URL}), the regional bike-share program."
            )
        return tip
    if "munimobile" in lowered:
        return f"Use [MuniMobile]({MUNIMOBILE_URL}) for mobile tickets and trip planning."
    return tip

SERVICE_CACHE_VERSION = "2026-04-20-ui-contrast-fix-v2"


def metric_card(value: str, label: str, detail: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-detail">{detail}</div>
    </div>
    """


def citation_jump_card(citation: dict[str, str], index: int) -> str:
    title = escape(str(citation.get("title", "")))
    location = escape(str(citation.get("location", "")))
    topic = escape(str(citation.get("topic", "")))
    source_type = escape(str(citation.get("source_type", "")))
    chunk_id = escape(str(citation.get("chunk_id", "")))
    evidence_anchor = escape(str(citation.get("evidence_anchor", "#")), quote=True)
    return f"""
    <div class="citation-nav-card">
        <a class="citation-nav-link" href="{evidence_anchor}">
            [{index}] Jump to exact chunk
        </a><br/>
        <strong>{title}</strong><br/>
        <span>{location} | {topic} | {source_type}</span><br/>
        <span>Chunk ID: {chunk_id}</span>
    </div>
    """


def evidence_record_card(citation: dict[str, str], index: int) -> str:
    title = escape(str(citation.get("title", "")))
    location = escape(str(citation.get("location", "")))
    topic = escape(str(citation.get("topic", "")))
    source_type = escape(str(citation.get("source_type", "")))
    chunk_id = escape(str(citation.get("chunk_id", "")))
    doc_id = escape(str(citation.get("doc_id", "")))
    timestamp = escape(str(citation.get("timestamp", "")))
    passage_index = escape(str(citation.get("passage_index", "")))
    score = escape(str(citation.get("score", "")))
    evidence_dom_id = escape(str(citation.get("evidence_dom_id", "")), quote=True)
    passage_text = escape(str(citation.get("passage_text", "")))
    source_url = str(citation.get("source_url", "")).strip()
    source_link = ""
    if source_url:
        safe_source_url = escape(source_url, quote=True)
        source_link = (
            f'<a href="{safe_source_url}" target="_blank" rel="noopener noreferrer">'
            "Open original source page"
            "</a>"
        )
    return f"""
    <div id="{evidence_dom_id}" class="evidence-card evidence-record-card">
        <div class="evidence-record-header">Evidence Record [{index}]</div>
        <strong>{title}</strong><br/>
        <div class="evidence-meta">
            <span>Chunk ID: {chunk_id}</span>
            <span>Document ID: {doc_id}</span>
            <span>Passage: {passage_index}</span>
            <span>Score: {score}</span>
        </div>
        <div class="evidence-meta">
            <span>{location}</span>
            <span>{topic}</span>
            <span>{source_type}</span>
            <span>{timestamp}</span>
        </div>
        <div class="evidence-passage">{passage_text}</div>
        <div class="evidence-footer">
            This is the exact stored chunk retrieved for the answer.
            {source_link}
        </div>
    </div>
    """

st.set_page_config(
    page_title="LocalLens",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #18212b;
        --muted: #5c6773;
        --sand: #f5efe4;
        --cream: #fbf8f2;
        --mist: #edf2f4;
        --sun: #f4c58e;
        --coral: #de7b63;
        --berry: #954f57;
        --line: rgba(24, 33, 43, 0.10);
        --shadow: 0 22px 60px rgba(27, 34, 43, 0.08);
        --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        --sans: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 12% 10%, rgba(244, 197, 142, 0.42), transparent 22%),
            radial-gradient(circle at 84% 8%, rgba(132, 169, 196, 0.22), transparent 24%),
            linear-gradient(180deg, #f7f1e7 0%, #fbf8f2 42%, #eef2f4 100%);
        color: var(--ink);
        font-family: var(--sans);
    }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] .st-emotion-cache-ue6h4q,
    [data-testid="stAppViewContainer"] .st-emotion-cache-1wivap2,
    [data-testid="stAppViewContainer"] .st-emotion-cache-1r6slb0 {
        color: var(--ink) !important;
    }
    h1, h2, h3 {
        letter-spacing: -0.03em;
        color: var(--ink);
    }
    .hero {
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.86), rgba(255,255,255,0.72)),
            radial-gradient(circle at top right, rgba(244, 197, 142, 0.14), transparent 24%);
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        padding: 1.8rem 1.8rem 1.4rem 1.8rem;
        margin-bottom: 1.1rem;
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        color: var(--berry);
        margin-bottom: 0.55rem;
        font-weight: 700;
    }
    .hero-title {
        font-family: var(--serif);
        font-size: 4rem;
        line-height: 0.95;
        margin: 0 0 0.7rem 0;
        font-weight: 700;
    }
    .hero-copy {
        font-size: 1.13rem;
        line-height: 1.65;
        color: var(--ink);
        max-width: 58rem;
        margin: 0;
    }
    .hero-subcopy {
        font-size: 0.98rem;
        line-height: 1.7;
        color: var(--muted);
        max-width: 56rem;
        margin-top: 0.75rem;
    }
    .metric-card {
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        padding: 1rem 1rem 0.95rem 1rem;
        min-height: 132px;
    }
    .metric-value {
        font-family: var(--serif);
        font-size: 2rem;
        line-height: 1;
        margin-bottom: 0.35rem;
        color: var(--ink);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--berry);
        margin-bottom: 0.4rem;
        font-weight: 700;
    }
    .metric-detail {
        color: var(--muted);
        line-height: 1.45;
        font-size: 0.92rem;
    }
    .section-kicker {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        color: var(--berry);
        margin: 0.8rem 0 0.3rem 0;
        font-weight: 700;
    }
    .section-title {
        font-family: var(--serif);
        font-size: 2.2rem;
        margin: 0 0 0.25rem 0;
    }
    .section-copy {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 0.9rem;
        max-width: 52rem;
    }
    .search-note {
        border-radius: 18px;
        background: rgba(255,255,255,0.68);
        border: 1px solid var(--line);
        padding: 0.95rem 1rem;
        color: var(--muted);
        margin: 0.35rem 0 1rem 0;
    }
    .place-card, .evidence-card {
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid var(--line);
        box-shadow: 0 18px 48px rgba(20, 33, 43, 0.06);
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    .citation-nav-card {
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid var(--line);
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .citation-nav-link {
        color: #9b4f46;
        font-weight: 700;
        text-decoration: none;
    }
    .citation-nav-link:hover {
        text-decoration: underline;
    }
    .evidence-record-card {
        scroll-margin-top: 1.2rem;
    }
    .evidence-record-header {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.76rem;
        color: var(--berry);
        margin-bottom: 0.55rem;
        font-weight: 700;
    }
    .evidence-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem 0.9rem;
        color: var(--muted);
        font-size: 0.88rem;
        margin: 0.35rem 0 0.45rem 0;
    }
    .evidence-passage {
        white-space: pre-wrap;
        line-height: 1.68;
        color: var(--ink);
        background: rgba(245, 239, 228, 0.65);
        border-radius: 14px;
        padding: 0.9rem 0.95rem;
        border: 1px solid rgba(24, 33, 43, 0.08);
        margin-top: 0.65rem;
    }
    .evidence-footer {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: 0.7rem;
    }
    .evidence-footer a {
        margin-left: 0.45rem;
        color: #9b4f46;
        font-weight: 600;
    }
    .place-img {
        width: 100%;
        border-radius: 14px;
        margin-bottom: 0.8rem;
    }
    .stButton button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #d97a5d, #b55a5a);
        color: #fff;
        font-weight: 700;
        box-shadow: 0 14px 28px rgba(181, 90, 90, 0.18);
        padding: 0.72rem 1.2rem;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #cb6b50, #a54c50);
        color: #fff;
    }
    .stTextArea textarea, div[data-baseweb="select"] > div, .stTextInput input {
        border-radius: 18px !important;
        border: 1px solid rgba(24, 33, 43, 0.12) !important;
        background: rgba(255, 255, 255, 0.76) !important;
        box-shadow: 0 12px 28px rgba(20, 33, 43, 0.04);
    }
    .stTextArea textarea {
        min-height: 8.5rem;
        font-size: 1rem;
        line-height: 1.6;
    }
    .stTextArea textarea,
    .stTextInput input,
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder,
    .stSelectbox label,
    .stTextArea label,
    .stTextInput label,
    div[data-baseweb="select"] *,
    div[data-baseweb="popover"] *,
    ul[role="listbox"] *,
    li[role="option"] * {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"],
    ul[role="listbox"],
    li[role="option"] {
        background: rgba(255, 255, 255, 0.96) !important;
    }
    div[data-baseweb="select"] svg,
    div[data-baseweb="popover"] svg {
        fill: var(--ink) !important;
        color: var(--ink) !important;
    }
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] input {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        caret-color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input,
    [data-testid="stSidebar"] [data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, 0.96) !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder,
    [data-testid="stSidebar"] [data-testid="stTextArea"] textarea::placeholder {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #232430 0%, #2a2d39 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #f5efe8;
    }
    .sidebar-shell {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-title {
        font-family: var(--serif);
        font-size: 1.6rem;
        margin: 0 0 0.4rem 0;
        color: #fff;
    }
    .sidebar-copy {
        color: rgba(245,239,232,0.78);
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }
    .status-chip {
        display: inline-block;
        margin: 0.2rem 0.35rem 0.2rem 0;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        color: #f7f0e7;
        font-size: 0.82rem;
    }
    .status-chip strong {
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_service(_cache_version: str) -> LocalLensService:
    return LocalLensService()


service = get_service(SERVICE_CACHE_VERSION)
stats = service.stats()

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">Local Knowledge, Not Just Search Results</div>
        <h1 class="hero-title">LocalLens</h1>
        <p class="hero-copy">
            A traveler-focused local intelligence desk for the questions people usually only
            learn how to answer after living somewhere for a while.
        </p>
        <p class="hero-subcopy">
            Ask about transit, neighborhoods, hidden friction, food, timing, safety, or
            practical local tradeoffs. LocalLens retrieves grounded evidence, ranks places,
            and turns it into a narrative answer with clickable citations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(4, gap="medium")
metric_columns[0].markdown(
    metric_card(str(stats["location_count"]), "Destinations", "U.S. cities and parks covered in the live corpus."),
    unsafe_allow_html=True,
)
metric_columns[1].markdown(
    metric_card(f"{stats['raw_documents']:,}", "Source Docs", "Guides, community threads, and background documents."),
    unsafe_allow_html=True,
)
metric_columns[2].markdown(
    metric_card(f"{stats['chunks']:,}", "Passages", "Citation-ready text chunks used by the retriever."),
    unsafe_allow_html=True,
)
metric_columns[3].markdown(
    metric_card(f"{stats['places']:,}", "Place Records", "Structured restaurants, parks, transit nodes, and more."),
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-shell">
            <div class="sidebar-title">Filters</div>
            <p class="sidebar-copy">
                Narrow the answer if you already know the city or topic. If you leave these open,
                LocalLens will infer context from the question itself.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_location = st.selectbox(
        "Location",
        options=[""] + list(stats["locations"]),
        format_func=lambda value: "Auto-detect from query" if value == "" else value,
    )
    selected_topic = st.selectbox(
        "Topic",
        options=[""] + list(stats["topics"]),
        format_func=lambda value: "Auto-detect from query" if value == "" else value.title(),
    )
    st.markdown(
        f"""
        <div>
            <span class="status-chip"><strong>Embeddings</strong>: {stats['embedding_backend']}</span>
            <span class="status-chip"><strong>Ollama</strong>: {stats['ollama_generation_available']}</span>
            <span class="status-chip"><strong>Maps key</strong>: {stats['google_places_available']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("What does `Build / Refresh Corpus` do?"):
        st.markdown(
            """
            It reruns the entire data pipeline:

            - fetches source material from travel guides, background pages, Reddit, and place data
            - rebuilds the local SQLite database
            - recomputes the dense embedding index used for retrieval

            It takes time because it is doing network fetches plus embedding computation over thousands
            of passages. You only need it when the underlying dataset changes or after a fresh checkout.
            """
        )
    if st.button("Rebuild Data + Embeddings"):
        with st.spinner("Refetching sources, rebuilding the database, and recomputing embeddings..."):
            refreshed = service.rebuild_assets()
            get_service.clear()
            st.session_state["rebuild_done"] = True
        st.success(
            f"Rebuilt {refreshed['documents']} documents, {refreshed['chunks']} chunks, and {refreshed['places']} places."
        )

if st.session_state.get("rebuild_done"):
    service = get_service(SERVICE_CACHE_VERSION)
    stats = service.stats()
    st.session_state.pop("rebuild_done", None)

st.markdown(
    """
    <div class="section-kicker">Plan A Query</div>
    <h2 class="section-title">Ask About A Place Like A Local</h2>
    <p class="section-copy">
        Search is fast because it uses the existing local database and embedding index.
        The rebuild action is separate and only needed when the data itself changes.
    </p>
    """,
    unsafe_allow_html=True,
)

sample_queries = service.sample_queries(
    location=selected_location,
    topic=selected_topic,
)
sample_query_key = "sample_query_select"
sample_scope = (selected_location, selected_topic)
previous_scope = st.session_state.get("sample_query_scope")
if previous_scope != sample_scope:
    st.session_state[sample_query_key] = ""
st.session_state["sample_query_scope"] = sample_scope
if st.session_state.get(sample_query_key, "") not in [""] + sample_queries:
    st.session_state[sample_query_key] = ""

example_query = st.selectbox(
    "Try a sample query",
    options=[""] + sample_queries,
    format_func=lambda value: "Select a sample question" if value == "" else value,
    key=sample_query_key,
)
default_query = example_query if example_query else ""
st.markdown(
    """
    <div class="search-note">
        Try something practical and specific: a neighborhood to stay in, a transit question,
        a sunset spot, a first-week itinerary, or a restaurant query with constraints like rating or cuisine.
    </div>
    """,
    unsafe_allow_html=True,
)
query = st.text_area(
    "Ask LocalLens",
    value=default_query,
    height=120,
    placeholder=(
        "Examples: Best taco place in San Jose over 4.5, "
        "what should I know before moving to Seattle, "
        "or where is a good sunset spot in San Francisco?"
    ),
)

run = st.button("Search and Answer", type="primary")

if run and query.strip():
    with st.spinner("Retrieving evidence, ranking places, and composing a grounded narrative..."):
        response = service.answer(
            query.strip(),
            location=selected_location,
            topic=selected_topic,
        )

    left, right = st.columns([1.35, 0.9], gap="large")
    with left:
        st.subheader("Answer")
        st.write(response.answer)
        st.subheader("Why This Recommendation")
        st.write(response.why_this_recommendation)
        st.subheader("Key Tips")
        for tip in response.key_tips:
            st.markdown(f"- {enrich_tip_text(tip, response)}")
        st.caption(response.confidence_note)

        if response.gallery_images:
            st.subheader("Visual Context")
            image_columns = st.columns(min(len(response.gallery_images), 3))
            for index, image in enumerate(response.gallery_images[:3]):
                with image_columns[index % len(image_columns)]:
                    st.image(image["url"], caption=image["title"], use_container_width=True)

        if response.place_cards:
            st.subheader("Recommended Places")
            for card in response.place_cards:
                st.markdown(
                    f"""
                    <div class="place-card">
                        <strong>{card['name']}</strong><br/>
                        <span>{card['location']} | {card['category']} | {card['source_provider']}</span><br/>
                        <span>Rating: {card['rating'] if card['rating'] is not None else 'n/a'} | Reviews: {card['review_count'] or 'n/a'}</span><br/>
                        <span>{card['address'] or 'Address unavailable'}</span><br/><br/>
                        <span>{card['why']}</span><br/><br/>
                        <a href="{card['source_url']}" target="_blank">Open place source</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if card["review_snippets"]:
                    for snippet in card["review_snippets"][:2]:
                        st.caption(snippet)

    with right:
        st.subheader("Retrieval Notes")
        filter_items = [f"{key}: {value}" for key, value in response.filters_applied.items() if value]
        st.write("Applied filters: " + (", ".join(filter_items) if filter_items else "none"))
        st.write("Source mix: " + response.source_summary)
        st.write(
            "Generation mode: "
            + ("Local Ollama model" if response.used_local_llm else "Grounded fallback narrative")
        )

    if response.citations:
        st.subheader("Citations")
        st.caption(
            "Each citation below jumps to the exact stored chunk retrieved for this answer. "
            "The external source page is shown only as secondary context inside the evidence record."
        )
        for index, citation in enumerate(response.citations, start=1):
            st.markdown(citation_jump_card(citation, index), unsafe_allow_html=True)

        st.subheader("Evidence Records")
        for index, citation in enumerate(response.citations, start=1):
            st.markdown(evidence_record_card(citation, index), unsafe_allow_html=True)
else:
    st.info(
        "If this is a fresh checkout, press `Rebuild Data + Embeddings` once to ingest the data and create the local SQLite database and retrieval index."
    )
