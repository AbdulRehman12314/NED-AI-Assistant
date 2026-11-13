# Short Report: NED Academy Advanced RAG Assistant

## 1. Business Use Case
NED Academy operates as the Training & Professional Development Wing of NED University of Engineering & Technology, Karachi[cite: 15]. The academy offers professional diplomas (Undergraduate Diplomas - UGD, Postgraduate Diplomas - PGD), Postgraduate Certificate Courses (PGCC), short certifications, and specialized vocational programmes[cite: 16]. 

Prospective students, working professionals, and current trainees frequently submit inquiries regarding:
* Admission workflows and eligibility prerequisites[cite: 17].
* Programme-specific fee structures and lump-sum discounts[cite: 18].
* Attendance thresholds (e.g., mandatory 75% attendance) and examination rules[cite: 18].
* Physical and online class schedules and contact channels[cite: 19].

To streamline student support, reduce manual overhead, and eliminate unverified claims, this project implements an enterprise-grade **Advanced RAG (Retrieval-Augmented Generation)** assistant. It provides verified, evidence-backed answers with strict source traceability, version-controlled policy handling, and robust fallback mechanisms.

---

## 2. Pipeline Architecture
The system integrates an advanced 7-stage RAG pipeline inspired by Module 09 coursework:
1. **Document Ingestion & Chunking:** Parses unstructured and semi-structured documents (PDFs, CSVs, and Excel tracking logs) from the `knowledge_base/` folder. Text is segmented into semantic chunks with controlled character overlap to preserve contextual continuity.
2. **Query Rewriting:** Normalizes shorthand queries, typos, and Roman Urdu phrases (e.g., *"admision ki fee"* or *"UGD rate"*) into clean, standardized English search terms.
3. **Hybrid Search:** Combines dense vector semantic search (`sentence-transformers/all-MiniLM-L6-v2`) with sparse keyword matching (`BM25Okapi`) to ensure exact keyword matches (such as specific course IDs or fee figures) are never missed alongside semantic meaning.
4. **Reciprocal Rank Fusion (RRF):** Harmonizes semantic and keyword ranking lists into a unified relevance score.
5. **Cross-Encoder Reranking:** Re-scores the top candidate chunks using `ms-marco-MiniLM-L-6-v2` to prioritize the most precise context blocks.
6. **Context Neighbour Expansion & Packing:** Automatically appends adjacent chunks from the same file to prevent rule fragmentation and packs the final context window efficiently.
7. **LLM Generation:** Queries the Groq LLM (`llama-3.3-70b-versatile`) under strict system prompt constraints that enforce evidence-only answers, exact source filename citations in brackets, and precise fallback messaging when evidence is absent[cite: 17, 18, 41].

---

## 3. Retrieval Investigation & Experiments
We evaluated retrieval performance across $k=1$, $k=3$, and $k=5$ using core test queries:
* **$k=1$:** Highly efficient for direct single-fact lookups (such as registration fees), but frequently failed when queries spanned multiple documents or required broader institutional context.
* **$k=3$ to $k=5$:** Established the optimal operational range. Capturing 3 to 5 chunks ensured that supporting evidence was reliably present even when buried across tables or multi-page policy sections.
* **Chunk Size & Overlap Tuning:** Reducing chunk size from 850 to 600 characters with an 80-character overlap improved retrieval precision for specific line items in tabular fee schedules, though it occasionally fragmented multi-step policy workflows. This fragmentation was successfully resolved by incorporating our **Neighbour Expansion** module.

---

## 4. Failures, Limitations & Production Readiness
While the prototype performs reliably under test conditions, several limitations must be addressed before enterprise deployment:
* **Dynamic Data Lag:** Institutional details such as live batch openings, daily seat availability, and class start dates change dynamically. Static RAG documents cannot reflect live registration status without direct API integrations with the university's admissions portal.
* **Table & Spreadsheet Parsing:** Dense tabular data inside Excel tracking logs and CSV files required careful formatting during ingestion to preserve row-column relationships across text chunks.
* **Classroom Prototype Scope:** This system is implemented as an offline-capable, local SQLite prototype running on free-tier APIs. A production rollout would require enterprise access control, audit logging, and automated document synchronisation with official university CMS databases.