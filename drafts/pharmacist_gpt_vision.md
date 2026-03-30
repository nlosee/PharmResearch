# Vision: Pharmacist Coaching GPT & Regulatory Knowledge Graph

The Pharma Regulatory Pulse project is evolving from a weekly email newsletter into a **Retrieval-Augmented Generation (RAG)** knowledge engine. The ultimate goal is to build an interactive Educational GPT that pharmacists can consult to get real-time, compliance-approved coaching on patient consultations based on the latest guidelines, clinical trials, and FDA/EMA enforcement actions.

---

## 🏗️ Architecture Overview

The system transitions from a linear pipeline (Research → Normalize → Email) into a three-tiered RAG architecture:

1. **The Ingestion Engine (Currently Active)**: The existing pipeline acts as an automated data scraper and enricher. It pulls from PubMed, OpenFDA, and RSS feeds, and uses `gpt-5.4-nano` to score, tag keywords, and flag guideline changes.
2. **The Knowledge Database (Next Step)**: A Vector Database (e.g., Pinecone/Qdrant) and optionally a Graph Database (e.g., Neo4j). This stores the "embeddings" (mathematical representations) of every enriched article.
3. **The Chat Agent (Final Phase)**: A user-facing application (e.g., built with Streamlit, Next.js, or Vercel AI SDK) where a pharmacist asks a natural language question. The agent retrieves the top 5 most relevant articles from the Knowledge Database and uses an LLM (e.g., `gpt-5.4-mini` or `gpt-4.1-mini`) to synthesize a coaching response.

---

## 🚀 Action Plan

### ✅ Phase 1: Data Preservation (Completed)
Instead of discarding raw articles after the newsletter is sent, the pipeline now archives the fully enriched JSON payload (including `audience_fit`, `trust_score`, `keywords`, and `_guideline_boost`) to the `archive/` directory. 
- **Status:** Done. We are now passively building the Knowledge Corpus with every weekly run.

### ⏳ Phase 2: Vector Database Integration & Social Listening (Upcoming)
**Social Media Commentary Engine:**
- **Action 1:** Create a Google Sheet containing the "Social Roster" (X handles, LinkedIn hashtags, orgs). Go to `File -> Share -> Publish to web -> CSV`.
- **Action 2:** Build an Apify integration into `research.py`. Every Monday, it downloads the published Google Sheet CSV and scrapes the top commentary from those specific pharmacists/organizations over the last 7 days.
- **Action 3:** Update `synthesize.py` (`gpt-5.4-mini`) to merge this commentary into the clinical updates.

**Vector DB Enablement:**
- **Action 4:** Choose a Vector Database (Pinecone is easiest, Qdrant is great for local).
- **Action 5:** Write an `upload_to_vector_db.py` script. The script iterates through `.json` files in `archive/`, generates embeddings (`text-embedding-3-small`), and upserts them.
- **Action 6:** Update the GitHub Actions workflow to run this script automatically.

### ⏳ Phase 3: The Educational GPT Agent (Future)
Build the actual chat interface for pharmacists.
- **Action 1:** Create a simple Streamlit or Gradio web app.
- **Action 2:** Define the **System Prompt** for the coaching agent (e.g., *"You are an expert clinical pharmacist coach. Based strictly on the provided retrieved guidelines, advise the user on how to consult their patient."*).
- **Action 3:** Implement the RAG loop: 
  1. User asks question. 
  2. Embed question. 
  3. Query Vector DB for top 5 matches. 
  4. Pass matches to LLM. 
  5. Stream response back to user.

### 🔮 Phase 4: GraphRAG Expansion (Advanced Vision)
For highly complex queries (e.g., "What are all the cascading effects of the new KDIGO renal guidelines across multiple drug classes?"), expand from pure Vector Search to a **Knowledge Graph**.
- Use an LLM to extract entities (`[FDA] -> [issued] -> [Boxed Warning] -> [for Drug X]`) from the archived JSON.
- Store relationships in Neo4j.
- The Chat Agent can traverse these graph relationships to provide incredibly comprehensive multi-hop answers.
