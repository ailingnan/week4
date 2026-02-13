
import streamlit as st
import pandas as pd
from pathlib import Path

from src.rag_core import Week3RAG, rag_query_logged

st.set_page_config(page_title="Capstone RAG Module", page_icon="🧠", layout="wide")
st.title("🧠 Capstone RAG Module (Week 4)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K evidence", 1, 10, 5)

    st.markdown("---")
    st.header("Monitoring (logs/product_metrics.csv)")
    log_path = Path("logs/product_metrics.csv")
    if log_path.exists():
        df = pd.read_csv(log_path)
        st.metric("Total queries", len(df))
        st.metric("Avg latency (ms)", f"{df['latency_ms'].astype(float).mean():.0f}")
        st.metric("Avg confidence", f"{df['confidence'].astype(float).mean():.2f}")
    else:
        st.info("No logs yet. Run a query.")

@st.cache_resource
def load_rag():
    docs_dir = "project_data_mm/docs"
    rag = Week3RAG(docs_dir=docs_dir)
    rag.build()
    return rag

rag = load_rag()

# Main UI panels
query = st.text_input("Query input", placeholder="Ask something about the uploaded UMKC PDFs...")

col1, col2 = st.columns([2,1], gap="large")

if st.button("Run", type="primary") and query:
    resp = rag_query_logged(rag, query, top_k=top_k)

    with col1:
        st.subheader("Response panel")
        st.write(resp["answer"])

        st.subheader("Evidence display")
        if not resp["evidence"]:
            st.warning("No evidence matched the query.")
        else:
            for i, e in enumerate(resp["evidence"], 1):
                with st.expander(f"[{i}] {e['source']} p{e['page']} | score={e['score']:.3f} | {e['evidence_id']}"):
                    st.write(e["text"])

    with col2:
        st.subheader("Metrics panel")
        st.metric("Latency (ms)", f"{resp['latency_ms']:.0f}")
        st.metric("Confidence", f"{resp['confidence']:.2f}")
        st.caption("Trust indicator: evidence + source/page citation shown above.")

st.markdown("---")
st.subheader("Recent log rows (tail)")
log_path = Path("logs/product_metrics.csv")
if log_path.exists():
    df = pd.read_csv(log_path).tail(10)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No logs yet.")
