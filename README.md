# 🎓 NED Academy Digital Assistant

An enterprise-grade Advanced RAG (Retrieval-Augmented Generation) assistant built for **NED Academy** (the Training & Professional Development Wing of NED University of Engineering & Technology, Karachi)[cite: 27]. 

This application provides reliable, evidence-backed answers to student and public inquiries regarding admissions, course fees, UGD/PGD programmes, facilities, attendance policies, and contact channels, while strictly avoiding hallucinations and handling prompt injections[cite: 27, 39, 40, 41].

---

## 🚀 Key Features
- **Advanced RAG Pipeline (Module 09):**
  - **Query Rewriting:** Expands colloquial Roman Urdu and shorthand terms into formal English search queries.
  - **Hybrid Search:** Combines dense vector semantic search (SentenceTransformers) with sparse keyword matching (BM25Okapi).
  - **Reciprocal Rank Fusion (RRF):** Merges semantic and keyword rankings robustly.
  - **Cross-Encoder Reranking:** Re-scores retrieved candidates using `ms-marco-MiniLM-L-6-v2` for high precision.
  - **Multi-Document Fusion:** Splits compound questions to search across multiple knowledge files.
  - **Long-Context Neighbour Expansion:** Automatically includes adjacent text chunks to prevent rule fragmentation.
- **Strict Evidence Guardrails:** Answers exclusively from uploaded documents, citing source files and excerpts, and outputs exact fallback messages (`INSUFFICIENT`) when evidence is missing[cite: 27, 38].
- **Professional UI:** Styled with a clean White & Red professional theme supporting a Light/Dark Mode toggle.
- **Local SQLite Vector Database:** Zero external paid vector databases required; runs entirely locally using SQLite and Python.

---
---

## 🎥 Demonstration Video
You can watch the complete 3-5 minute demo video covering all required test scenarios (factual lookups, multi-doc fusion, policy comparison, missing info, and prompt-injection resistance) here:
👉 **[Watch NED Academy Assistant Demo Video](https://drive.google.com/file/d/1_rCZ_jiSnT_EeSRklDB1QckSxaZvnh-J/view?usp=sharing)**

## 📂 Project Structure
```text
NED/
├── knowledge_base/               # Official PDF, CSV, and Excel knowledge documents
│   ├── NED_Academy_GPT_Master_Knowledge_v1.2.1.pdf
│   ├── NED_Academy_FAQ_Master_v1.2.csv
│   └── NED_Academy_Source_Index_and_Update_Log_v1.0.xlsx
├── reports/                      # Evaluation and test results
│   ├── test_results.csv          # 20-question comprehensive test suite
│   └── retrieval_experiments.csv # k=1,3,5 retrieval ablation logs
├── .env                          # Environment variables (Groq API key)
├── .env.example                  # Template for environment variables
├── .gitignore                    # Git exclusion rules
├── README.md                     # Project documentation
├── requirements.txt              # Python package dependencies
├── config.py                     # Centralized configuration
├── rag_engine.py                 # Advanced RAG pipeline logic
├── build_index.py                # Script to build/rebuild SQLite vector index
└── app.py                        # Streamlit web application frontend