import streamlit as st
from config import TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
from rag_engine import (
    rebuild_index, smart_search, multi_search, add_neighbours, 
    ask_llm, rewrite_query, INSUFFICIENT
)

st.set_page_config(
    page_title="NED Academy Digital Assistant",
    page_icon="🎓",
    layout="wide"
)

# --- THEME STYLING (Professional White & Red Theme with Light/Dark Mode Toggle) ---
theme_mode = st.sidebar.selectbox("🎨 Theme Mode", ["Light (Clean White & Red)", "Dark (Rich Charcoal & Red)"], index=0)

if theme_mode.startswith("Light"):
    bg_color, card_bg, text_color, border_color = "#FFFFFF", "#F8FAFC", "#1E293B", "#E2E8F0"
    primary_red, hover_red, sidebar_bg = "#DC2626", "#B91C1C", "#F1F5F9"
else:
    bg_color, card_bg, text_color, border_color = "#0F172A", "#1E293B", "#F8FAFC", "#334155"
    primary_red, hover_red, sidebar_bg = "#1E293B", "#DC2626", "#1E293B"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar_bg}; border-right: 1px solid {border_color}; }}
    h1, h2, h3, h4 {{ color: {primary_red} !important; font-family: 'Inter', sans-serif; }}
    .stButton>button {{
        background-color: {primary_red}; color: white; border-radius: 8px; border: none; font-weight: 600; padding: 0.5rem 1rem;
    }}
    .stButton>button:hover {{ background-color: {hover_red}; box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3); }}
    .stExpander {{ background-color: {card_bg}; border: 1px solid {border_color}; border-radius: 8px; }}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🎓 NED Academy Digital Assistant")
st.markdown("Powered by **Advanced RAG (Query Rewriting, Hybrid Search, Cross-Encoder Reranking, Multi-Document Fusion & Long-Context Packing)** for NED University training and professional programmes.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    k = st.slider("Retrieved Chunks (k)", min_value=1, max_value=8, value=TOP_K)
    
    st.divider()
    st.subheader("🔍 Advanced RAG Toggles")
    use_query_rewrite = st.checkbox("Query Rewriting", value=True)
    use_multi_doc = st.checkbox("Multi-Document Fusion", value=True)
    use_neighbours = st.checkbox("Context Neighbour Expansion", value=True)
    
    st.divider()
    st.subheader("📁 Index Maintenance")
    st.caption("Rebuild after editing or adding documents in `knowledge_base/`.")
    if st.button("Rebuild Vector Index", use_container_width=True):
        with st.spinner("Embedding knowledge files and building SQLite index..."):
            try:
                count = rebuild_index(CHUNK_SIZE, CHUNK_OVERLAP)
                st.success(f"Index rebuilt successfully! Stored {count} chunks.")
            except Exception as exc:
                st.error(f"Index rebuild failed: {exc}")
    st.divider()
    st.caption(f"Chunk Size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")

# --- QUERY INTERFACE ---
st.markdown("### Ask a Question about NED Academy")
question = st.text_area(
    "Type your query below:",
    placeholder="Examples: What is the fee for UGD?, How do I apply for PGD?, What laboratories are available?, Contact details...",
    height=100
)

if st.button("Get Answer", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Running Advanced RAG pipeline..."):
                # 1. Query Rewriting
                processed_q = rewrite_query(question) if use_query_rewrite else question
                if use_query_rewrite and processed_q != question:
                    st.info(f"🔄 **Optimized Query:** `{processed_q}`")

                # 2. Search & Retrieval (Smart Search / Multi-Doc)
                if use_multi_doc and (" and " in question or "," in question or " & " in question):
                    sub_queries = [q.strip() for q in question.replace(" and ", ",").replace(" & ", ",").split(",") if q.strip()]
                    hits = multi_search(sub_queries, k_per_part=2)
                else:
                    hits = smart_search(processed_q, top_n=k)

                # 3. Neighbour Expansion
                if use_neighbours and hits:
                    hits = add_neighbours(hits)

                # 4. LLM Generation
                answer = ask_llm(question, hits)

            st.markdown("---")
            st.subheader("💡 Answer")
            st.write(answer)

            st.subheader("📚 Supporting Evidence & Excerpts")
            if hits:
                seen = set()
                for hit in hits:
                    filename = hit.get("filename", "Document")
                    score = hit.get("score", 0.0)
                    key = (filename, hit["text"][:80])
                    if key in seen: continue
                    seen.add(key)
                    
                    with st.expander(f"📄 [{filename}] · Score: {score}"):
                        st.write(hit["text"])
            else:
                st.info(INSUFFICIENT)

        except Exception as exc:
            st.error(f"Execution error: {exc}")
            st.info("Ensure your `.env` API key is set and you have clicked **Rebuild Vector Index** in the sidebar.")