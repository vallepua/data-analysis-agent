# app.py - Data Analysis Agent (decision-support version)
import io
import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.chat import answer_question
from agent.data_loader import load_data
from agent.pipeline import run_analysis
from agent.stats_engine import column_kinds, suggest_target

st.set_page_config(page_title="Data Analysis Agent", page_icon="📊", layout="wide")


def _flag(name, default="1"):
    """Read an on/off setting from the environment or Streamlit secrets."""
    val = os.getenv(name)
    if val is None:
        try:
            val = st.secrets[name]
        except Exception:
            val = default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


ENABLE_CHAT = _flag("ENABLE_CHAT", "1")   # set ENABLE_CHAT=0 for public deployments
MAX_UPLOAD_MB = 10


def md(text):
    """Streamlit treats $...$ as math, so escape dollar signs."""
    return text.replace("$", "\\$")


@st.cache_data(show_spinner=False)
def load_cached(name, data):
    buf = io.BytesIO(data)
    buf.name = name
    return load_data(buf)


st.title("📊 Data Analysis Agent")
st.markdown("*Upload a CSV or Excel file, tell the agent what you are deciding, and get a verified, "
            "decision-oriented report with charts you can download as a PDF.*")

with st.sidebar:
    st.header("⚙️ How it works")
    steps = ("1. 📁 Upload your data\n"
             "2. 🎯 Say what you want to decide\n"
             "3. 🔢 Statistics are computed in Python\n"
             "4. 📊 Charts are chosen from the top drivers\n"
             "5. 🤖 AI writes the summary from those verified numbers\n"
             "6. ✅ Every figure it writes is re-checked\n")
    steps += ("7. 💬 Ask follow-up questions about your data" if ENABLE_CHAT
              else "7. 📥 Download the PDF report")
    st.markdown(steps)
    st.divider()
    st.info("Decision support, not proof. Results show associations in your file; "
            "they cannot show causes or factors that were not recorded.")

uploaded = st.file_uploader("📁 Upload your dataset", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.info("👆 Upload a CSV or Excel file to get started.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Works for any data**\n- Sales, finance, surveys, operations\n- Any columns, any size")
    c2.markdown("**You get**\n- Data trust check\n- Ranked drivers with confidence\n- Charts with explanations")
    c3.markdown("**And**\n- Segment comparisons\n- Recommendations\n- Chat with your data")
    st.stop()

raw = uploaded.getvalue()
if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"❌ File is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller file or a sample of it.")
    st.stop()
df, error = load_cached(uploaded.name, raw)
if error:
    st.error(f"❌ {error}")
    st.stop()

file_key = f"{uploaded.name}-{len(raw)}"
if st.session_state.get("file_key") != file_key:
    st.session_state["file_key"] = file_key
    st.session_state["result"] = None
    st.session_state["chat"] = []

kinds = column_kinds(df)
st.success(f"✅ Loaded **{uploaded.name}**: {df.shape[0]} rows x {df.shape[1]} columns")
with st.expander("👀 Preview data", expanded=False):
    st.dataframe(df.head(10), width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", df.shape[0])
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing values", int(df.isnull().sum().sum()))

# ------------------------------------------------------------------ goal
st.subheader("🎯 What are you trying to decide?")
numeric_cols = [c for c in df.columns if kinds[c] == "numeric"]
NONE_LABEL = "(No outcome - just explore the data)"
suggested = suggest_target(df, kinds)
options = numeric_cols + [NONE_LABEL]
default_idx = options.index(suggested) if suggested in options else len(options) - 1

c1, c2 = st.columns([1, 2])
with c1:
    choice = st.selectbox("Which column is the outcome you care about?", options, index=default_idx,
                          help="For example revenue, rent, churn score, profit. The agent will find what drives it.")
with c2:
    goal = st.text_area("Describe the decision or question (optional but recommended)",
                        placeholder="e.g. We are choosing which properties to renovate next year to raise rent.",
                        height=90)

target = None if choice == NONE_LABEL else choice
target_source = "auto-suggested, confirmed by user" if (target and target == suggested) else "chosen by user"

if st.button("🚀 Analyze my data", type="primary", width="stretch"):
    with st.status("Working...", expanded=True) as status:
        def progress(msg):
            st.write(f"⏳ {msg}")
        try:
            st.session_state["result"] = run_analysis(df, uploaded.name, target, target_source,
                                                      goal.strip(), progress)
            st.session_state["chat"] = []
            status.update(label="✅ Analysis complete", state="complete")
        except Exception as e:  # noqa
            status.update(label="❌ Something went wrong", state="error")
            st.error(f"{e}")

res = st.session_state.get("result")
if not res:
    st.stop()

facts, charts = res["facts"], res["charts"]
q = facts["profile"]

# ------------------------------------------------------------------ headline
badge = st.success if q["score"] >= 85 else st.warning if q["score"] >= 60 else st.error
badge(f"**Data trust: {q['verdict']}** (quality score {q['score']}/100)")
ver = res["verification"]
if ver.get("ai_used"):
    st.caption(f"AI-written text re-checked: {ver['verified']} of {ver['checked']} figures matched the computed "
               f"statistics" + (f"; {ver['removed']} unverifiable statement(s) removed." if ver["removed"] else "."))
else:
    st.warning("The AI service was unavailable (rate limit or key problem), so the summary was generated by rules "
               "from the verified numbers. " + ver.get("error", ""))

tab_names = ["📄 Report", "📊 Charts"] + (["💬 Ask your data"] if ENABLE_CHAT else []) + ["🔬 Evidence tables"]
tabs = st.tabs(tab_names)
tab_report, tab_charts, tab_evidence = tabs[0], tabs[1], tabs[-1]
tab_chat = tabs[2] if ENABLE_CHAT else None

with tab_report:
    st.markdown(md(res["report_md"]))
    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        if res["pdf_bytes"]:
            st.download_button("📥 Download full report (PDF)", data=res["pdf_bytes"],
                               file_name=f"report_{uploaded.name}.pdf", mime="application/pdf",
                               width="stretch")
        else:
            st.warning("The PDF could not be generated.")
    with d2:
        table = facts["driver_table"]
        if not table.empty:
            st.download_button("📥 Download driver table (CSV)", data=table.to_csv(index=False),
                               file_name=f"drivers_{uploaded.name}.csv", mime="text/csv", width="stretch")

with tab_charts:
    if not charts:
        st.info("No charts could be drawn for this dataset.")
    for i, c in enumerate(charts, 1):
        st.subheader(f"Chart {i}: {c['title']}")
        st.markdown(f"**What it shows:** {md(c['aim'])}")
        st.image(c["bytes"], width="stretch")
        st.markdown(f"**So what:** {md(c['so_what'])}")
        st.divider()

if tab_chat is not None:
    with tab_chat:
        st.markdown("Ask anything the report did not cover, for example *'average rent for buildings older than "
                    "10 years'* or *'plot rent against age'*. The agent writes pandas code, runs it, and fixes its "
                    "own mistakes.")
        for item in st.session_state["chat"]:
            with st.chat_message("user"):
                st.write(item["q"])
            with st.chat_message("assistant"):
                if item["text"]:
                    st.markdown(md(item["text"]))
                if item["table"] is not None:
                    st.dataframe(item["table"], width="stretch")
                if item["image"]:
                    st.image(item["image"])
                if item["error"] and not item["text"]:
                    st.error(item["error"])
                if item["code"]:
                    with st.expander("Show the code the agent ran"):
                        st.code(item["code"], language="python")
        question = st.chat_input("Ask a question about your data...")
        if question:
            with st.spinner("Thinking..."):
                out = answer_question(question, df, kinds, facts, goal.strip())
            out["q"] = question
            st.session_state["chat"].append(out)
            st.rerun()

with tab_evidence:
    st.markdown("#### Ranked drivers")
    if facts["driver_table"].empty:
        st.info("No outcome column was selected.")
    else:
        st.dataframe(facts["driver_table"], width="stretch")
    st.markdown("#### Data quality issues")
    st.dataframe(pd.DataFrame(q["issues"]), width="stretch")
    st.markdown("#### Column profile")
    st.dataframe(pd.DataFrame(q["columns"]), width="stretch")
    with st.expander("Exact fact sheet given to the AI"):
        st.code(res["fact_text"])
