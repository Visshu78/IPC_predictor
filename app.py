import streamlit as st
import time

# ─────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="IPC Predictor — AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS — premium dark legal theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* ── Root / Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0d14 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #0f1420 !important;
    border-right: 1px solid #1e2a40;
}

/* ── Main heading ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f6c90e 0%, #e8a815 40%, #c8860a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 1.8rem;
    font-weight: 400;
}

/* ── Input area ── */
.stTextArea textarea {
    background: #111827 !important;
    border: 1.5px solid #1e3a5f !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 14px !important;
    transition: border-color 0.25s ease;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: #f6c90e !important;
    box-shadow: 0 0 0 3px rgba(246,201,14,0.12) !important;
}
.stTextArea label {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #f6c90e, #d4a017) !important;
    color: #0a0d14 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2.2rem !important;
    cursor: pointer !important;
    transition: transform 0.15s ease, box-shadow 0.2s ease !important;
    letter-spacing: 0.02em;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(246,201,14,0.28) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(145deg, #111827, #0f1e30);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #f6c90e;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    position: relative;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: slideIn 0.4s ease forwards;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 36px rgba(246,201,14,0.12);
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ipc-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f6c90e, #c8860a);
    color: #0a0d14;
    font-weight: 700;
    font-size: 1rem;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 0.7rem;
    letter-spacing: 0.03em;
}
.offense-title {
    font-size: 1.13rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.9rem;
    line-height: 1.4;
}
.info-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    margin-bottom: 0.8rem;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 3px 11px;
    border-radius: 999px;
    letter-spacing: 0.02em;
}
.badge-green  { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.badge-red    { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }
.badge-blue   { background: rgba(96,165,250,0.12);  color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }
.badge-yellow { background: rgba(246,201,14,0.12);  color: #f6c90e; border: 1px solid rgba(246,201,14,0.25); }
.badge-gray   { background: rgba(148,163,184,0.1);  color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }

.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.25rem;
    margin-top: 0.7rem;
}
.punishment-text {
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.5;
}
.description-text {
    color: #94a3b8;
    font-size: 0.87rem;
    line-height: 1.6;
    margin-top: 0.3rem;
    font-style: italic;
}

/* ── Severity bar ── */
.sev-bar-bg {
    background: #1e2a40;
    border-radius: 999px;
    height: 7px;
    width: 100%;
    margin-top: 4px;
    overflow: hidden;
}
.sev-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Confidence chip ── */
.conf-chip {
    position: absolute;
    top: 1.2rem;
    right: 1.4rem;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 999px;
    letter-spacing: 0.04em;
}
.conf-high   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
.conf-medium { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid #fbbf24; }
.conf-low    { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid #f87171; }

/* ── Divider ── */
.gold-divider {
    border: none;
    border-top: 1px solid #1e2a40;
    margin: 1.5rem 0;
}

/* ── Sidebar ── */
.sidebar-title {
    font-size: 1rem;
    font-weight: 700;
    color: #f6c90e;
    margin-bottom: 0.5rem;
}
.sidebar-body {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.65;
}
.example-query {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.83rem;
    color: #94a3b8;
    cursor: pointer;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 2rem;
    background: #111827;
    border: 1px solid #1e2a40;
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    margin-bottom: 1.6rem;
}
.stat-item { text-align: center; }
.stat-val {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f6c90e;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 3px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 1rem;
    color: #475569;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.empty-msg { font-size: 1.05rem; }

/* ── No results ── */
.no-result-card {
    background: #111827;
    border: 1px dashed #1e3a5f;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: #64748b;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0d14; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 999px; }

/* ── Streamlit overrides ── */
[data-testid="stSpinner"] > div { color: #f6c90e !important; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MODEL LOADING (cached across sessions)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ipc_engine():
    from ipc_core import get_models
    get_models()
    return True


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def severity_color(val: float) -> str:
    if val >= 0.75:
        return "#f87171"
    if val >= 0.5:
        return "#fbbf24"
    return "#34d399"


def confidence_class(score: float) -> str:
    if score >= 0.72:
        return "conf-high"
    if score >= 0.58:
        return "conf-medium"
    return "conf-low"


def confidence_label(score: float) -> str:
    if score >= 0.72:
        return f"🎯 {score*100:.0f}% High"
    if score >= 0.58:
        return f"✅ {score*100:.0f}% Good"
    return f"⚠️ {score*100:.0f}% Low"


def cognizable_badge(val: str) -> str:
    v = val.lower()
    if "non" in v:
        return '<span class="badge badge-gray">🔵 Non-Cognizable</span>'
    if "cognizable" in v:
        return '<span class="badge badge-red">🔴 Cognizable</span>'
    return f'<span class="badge badge-gray">{val}</span>'


def bailable_badge(val: str) -> str:
    v = val.lower()
    if "non" in v:
        return '<span class="badge badge-red">🔒 Non-Bailable</span>'
    if "bailable" in v or v == "yes":
        return '<span class="badge badge-green">🔓 Bailable</span>'
    return f'<span class="badge badge-gray">{val}</span>'


def court_badge(val: str) -> str:
    return f'<span class="badge badge-blue">🏛️ {val}</span>'


def render_result_card(match: dict, rank: int):
    """Render a result card using two st.markdown calls to avoid Streamlit HTML cutoff."""
    ipc    = match["ipc"]
    offense = match["offense"]
    punish  = match["punishment"]
    cog    = match["cognizable"]
    bail   = match["bailable"]
    court  = match["court"]
    raw_desc = match.get("description", "")
    # Strip case citations — everything from "Cited by" onwards
    if "Cited by" in raw_desc:
        raw_desc = raw_desc.split("Cited by")[0].strip()
    desc = raw_desc[:280] + ("…" if len(raw_desc) > 280 else "")

    score      = match["score"]
    crime_sev  = match["crime_severity"]
    punish_sev = match["punishment_severity"]
    sev_match  = match["severity_match"]

    cs_color = severity_color(crime_sev)
    ps_color = severity_color(punish_sev)
    sm_color = "#34d399" if sev_match >= 0.72 else "#fbbf24" if sev_match >= 0.5 else "#f87171"

    conf_cls = confidence_class(score)
    conf_lbl = confidence_label(score)

    crime_sev_lbl  = "Extreme" if crime_sev  >= 0.85 else "High" if crime_sev  >= 0.65 else "Medium" if crime_sev  >= 0.4 else "Low"
    punish_sev_lbl = "Life/Death" if punish_sev >= 0.95 else "Severe" if punish_sev >= 0.65 else "Moderate" if punish_sev >= 0.4 else "Minor"
    sev_match_lbl  = "Excellent" if sev_match  >= 0.85 else "Good" if sev_match  >= 0.65 else "Fair" if sev_match  >= 0.45 else "Mismatch"

    # ── Part 1: card header (ipc badge, offense, badges, punishment, description)
    part1 = f"""
<div class="result-card">
  <span class="conf-chip {conf_cls}">{conf_lbl}</span>
  <div class="ipc-badge">{ipc}</div>
  <div class="offense-title">{offense}</div>
  <div class="info-row">
    {cognizable_badge(cog)}
    {bailable_badge(bail)}
    {court_badge(court)}
  </div>
  <div class="section-label">⛓️ Punishment</div>
  <div class="punishment-text">{punish}</div>
  <div class="section-label">📝 In Simple Words</div>
  <div class="description-text">{desc}</div>
</div>"""

    # ── Part 2: severity bars (separate call to avoid Streamlit 10k char limit)
    part2 = f"""
<div style="background:linear-gradient(145deg,#111827,#0f1e30);border:1px solid #1e3a5f;
  border-left:4px solid #f6c90e;border-top:none;border-radius:0 0 14px 14px;
  padding:0.9rem 1.6rem 1.2rem;margin-top:-16px;margin-bottom:1.2rem;">
  <hr class="gold-divider" style="margin:0 0 0.8rem">
  <div style="display:flex;gap:1.4rem;flex-wrap:wrap;">
    <div style="flex:1;min-width:110px;">
      <div class="section-label">Crime Severity</div>
      <div style="font-size:0.82rem;color:{cs_color};font-weight:600;margin-bottom:3px;">{crime_sev_lbl} ({crime_sev:.0%})</div>
      <div class="sev-bar-bg"><div class="sev-bar-fill" style="width:{crime_sev*100:.0f}%;background:{cs_color};"></div></div>
    </div>
    <div style="flex:1;min-width:110px;">
      <div class="section-label">Punishment Severity</div>
      <div style="font-size:0.82rem;color:{ps_color};font-weight:600;margin-bottom:3px;">{punish_sev_lbl} ({punish_sev:.0%})</div>
      <div class="sev-bar-bg"><div class="sev-bar-fill" style="width:{punish_sev*100:.0f}%;background:{ps_color};"></div></div>
    </div>
    <div style="flex:1;min-width:110px;">
      <div class="section-label">Severity Match</div>
      <div style="font-size:0.82rem;color:{sm_color};font-weight:600;margin-bottom:3px;">{sev_match_lbl} ({sev_match:.0%})</div>
      <div class="sev-bar-bg"><div class="sev-bar-fill" style="width:{sev_match*100:.0f}%;background:{sm_color};"></div></div>
    </div>
  </div>
</div>"""

    st.markdown(part1, unsafe_allow_html=True)
    st.markdown(part2, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚖️ IPC Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-body">AI-powered Indian Penal Code section finder. Describe a crime in plain English and get the most relevant IPC sections instantly.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">💡 Example Queries</div>', unsafe_allow_html=True)

    examples = [
        "A person stabbed his neighbor with a knife over a property dispute",
        "Someone hacked into my bank account and transferred money",
        "A group broke into a house at night and stole jewellery",
        "My employer hasn't paid my salary for three months",
        "A man was spreading false rumors about another person online",
        "Someone was caught selling counterfeit currency notes",
        "A drunk driver crashed into a pedestrian causing serious injury",
    ]

    if "query" not in st.session_state:
        st.session_state.query = ""

    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state.query = ex
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-title">🔍 How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-body">
<b>1. Semantic Search</b> — SentenceBERT finds conceptually similar IPC sections.<br><br>
<b>2. Keyword Matching</b> — TF-IDF ensures important legal keywords aren't missed.<br><br>
<b>3. Severity Calibration</b> — Crime intensity is matched against punishment severity to rank sections more accurately.<br><br>
<b>Hybrid Score</b> = 60% Semantic + 40% Keyword, penalty-adjusted for severity fit.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div class="sidebar-body" style="font-size:0.78rem;">
⚠️ <b>Disclaimer:</b> This tool is for educational and informational purposes only. It is not a substitute for professional legal advice. Always consult a qualified lawyer for legal matters.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">⚖️ IPC Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Describe a crime in plain language — get matching Indian Penal Code sections with full legal details, severity analysis, and confidence scores.</div>', unsafe_allow_html=True)

# Stats bar
st.markdown("""
<div class="stats-bar">
  <div class="stat-item"><div class="stat-val">511+</div><div class="stat-lbl">IPC Sections</div></div>
  <div class="stat-item"><div class="stat-val">2</div><div class="stat-lbl">AI Models</div></div>
  <div class="stat-item"><div class="stat-val">Hybrid</div><div class="stat-lbl">Search Engine</div></div>
  <div class="stat-item"><div class="stat-val">⚡ Fast</div><div class="stat-lbl">Cached Embeddings</div></div>
</div>
""", unsafe_allow_html=True)

# Input
# ── Slider row (above textarea)
slider_col1, slider_col2, slider_spacer = st.columns([2, 2, 3])
with slider_col1:
    top_k = st.select_slider(
        "🔢 Number of Results",
        options=[1, 2, 3, 5, 7, 10],
        value=5,
        help="How many IPC sections to return",
    )
with slider_col2:
    conf_filter = st.slider(
        "🎯 Min. Confidence",
        min_value=0,
        max_value=90,
        value=50,
        step=5,
        format="%d%%",
        help="Filter out results below this confidence threshold",
    )

# ── Full-width text area
query = st.text_area(
    "Crime Description",
    value=st.session_state.query,
    placeholder="e.g. A person was assaulted with an iron rod outside a shop in broad daylight...",
    height=120,
    label_visibility="collapsed",
    key="query_input",
)

# ── Button row — search button large, clear button ghost secondary
st.markdown("""
<style>
/* Ghost style for Streamlit secondary buttons */
[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: #64748b !important;
    border: 1.5px solid #334155 !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    transition: border-color 0.2s ease, color 0.2s ease !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(248,113,113,0.06) !important;
    border-color: #f87171 !important;
    color: #f87171 !important;
    box-shadow: none !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)

search_clicked = st.button("🔍  Find IPC Sections", use_container_width=True)
if st.button("✕  Clear", use_container_width=True, type="secondary"):
    st.session_state.query = ""
    st.rerun()


st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────
# SEARCH + RESULTS
# ─────────────────────────────────────────
if search_clicked or (st.session_state.get("query") and search_clicked):
    user_query = query.strip()

    if not user_query:
        st.warning("⚠️  Please enter a crime description before searching.")
    elif len(user_query) < 10:
        st.warning("⚠️  Description is too short. Please provide more detail for accurate results.")
    else:
        # Load models (cached)
        with st.spinner("⏳ Loading AI models (first run only — takes ~30 seconds)…"):
            load_ipc_engine()

        from ipc_core import find_ipc_sections

        with st.spinner("🔎 Analysing description and matching IPC sections…"):
            t_start = time.time()
            results = find_ipc_sections(user_query, top_k=top_k)
            elapsed = time.time() - t_start

        # Apply confidence filter
        min_score = conf_filter / 100.0
        filtered = [r for r in results if r["score"] >= min_score]

        # Summary header
        st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
  <div style="font-size:1.05rem;font-weight:600;color:#f1f5f9;">
    {'📋 ' + str(len(filtered)) + ' section' + ('s' if len(filtered) != 1 else '') + ' matched'}
    <span style="font-size:0.8rem;color:#64748b;font-weight:400;margin-left:0.5rem;">
      ({elapsed*1000:.0f} ms)
    </span>
  </div>
  <div style="font-size:0.8rem;color:#64748b;">
    Query: <em style="color:#94a3b8;">"{user_query[:60]}{'…' if len(user_query)>60 else ''}"</em>
  </div>
</div>
""", unsafe_allow_html=True)

        if not filtered:
            if results:
                closest = results[0]
                st.markdown(f"""
<div class="no-result-card">
  <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
  <div style="color:#94a3b8;font-size:1rem;margin-bottom:0.4rem;">No sections above {conf_filter}% confidence.</div>
  <div style="font-size:0.85rem;color:#64748b;">
    Closest match: <strong style="color:#f6c90e;">{closest['ipc']}</strong> — {closest['offense']}
    (Confidence: {closest['score']*100:.0f}%)
  </div>
  <div style="margin-top:0.8rem;font-size:0.82rem;color:#475569;">
    💡 Try lowering the Min. Confidence slider or adding more detail to your description.
  </div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div class="no-result-card">
  <div style="font-size:2rem;margin-bottom:0.5rem;">❌</div>
  <div style="color:#94a3b8;">No relevant IPC sections found. Try a more detailed description.</div>
</div>
""", unsafe_allow_html=True)
        else:
            # Show individual confidence guide once
            st.markdown("""
<div style="font-size:0.78rem;color:#475569;margin-bottom:1.1rem;">
  🎯 ≥72% High confidence &nbsp;|&nbsp; ✅ 58–71% Good &nbsp;|&nbsp; ⚠️ &lt;58% Low — review carefully
</div>
""", unsafe_allow_html=True)

            for i, match in enumerate(filtered):
                render_result_card(match, i)

            # Footer note if any results were filtered
            hidden = len(results) - len(filtered)
            if hidden > 0:
                st.markdown(f"""
<div style="text-align:center;font-size:0.8rem;color:#475569;margin-top:0.5rem;">
  ℹ️ {hidden} additional result{'s' if hidden>1 else ''} hidden (below {conf_filter}% confidence threshold).
  Lower the slider to see them.
</div>
""", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
<div class="empty-state">
  <div class="empty-icon">⚖️</div>
  <div class="empty-msg">Enter a crime description above and click <strong style="color:#f6c90e;">Find IPC Sections</strong></div>
  <div style="font-size:0.88rem;margin-top:0.6rem;color:#334155;">
    Or pick an example from the left sidebar to get started instantly.
  </div>
</div>
""", unsafe_allow_html=True)
