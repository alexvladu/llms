# AI-Powered City Recommendation System: Architecture Presentation
**Milestone #2 | Due April 10, 2026** | Week 5-6 Technical Design

---

## 1. System Architecture Overview

Our conversational city recommendation system comprises three integrated layers: **(1) Dialogue Layer** – Qwen 9B LLM for natural preference elicitation; **(2) Retrieval Layer** – RAG pipeline for contextual city knowledge; **(3) Ranking Layer** – weighted multi-criteria evaluation framework. The architecture enables progressive preference aggregation, transitioning autonomously from exploratory conversation to recommendation mode upon reaching preference-certainty thresholds. Users interact through natural language, with the system inferring explicit and implicit preferences across 8 evaluation dimensions (employment, climate, cost-of-living, infrastructure, air quality, taxation, language, culture).

## 2. Dataset Selection

**Primary Dataset:** Romanian Cities Corpus (315 cities, 100% coverage)
- **Content:** Wikipedia-derived HTML profiles containing comprehensive city metadata (demographics, economy, climate, infrastructure, cultural factors)
- **Format:** Structured HTML with semantic sections (geography, history, demographics, economy, infrastructure, culture)

**Domain Specifications:**
- Coverage: 315 Romanian cities (≥100% of administrative units)
- Information Density: 8+ structured evaluation dimensions per city
- Quality Baseline: Wikipedia-grade sourcing, real-time validation capability

## 3. Chunking Strategy (with overlap)

- Chunk size: 1,000–2,000 tokens per semantic section (balance between context density and retrieval precision)
- Overlap: 200-token windows to preserve boundary context

## 4. Vector Database & Retrieval Infrastructure

**Technology:** Chroma (open-source, in-process vector DB)
- **Rationale:** (1) Eliminates external service dependencies for on-premises deployment; (2) native embedding integration (Sentence Transformers); (3) persistent SQLite backend for reproducibility; (4) sub-millisecond query latency for real-time dialogue
- **Embedding Model:** Sentence Transformers (`all-MiniLM-L6-v2`) – 384-dim embeddings, optimized for semantic similarity at inference scale

**Configuration:**
- Distance metric: cosine similarity
- Index persistence: `chroma_cities/chroma.sqlite3`
- Query parameters: Top-5 retrieval per user preference signal

## 5. Model Choices & Dialogue Engine

**Primary Model Strategy:**
- **Base Model:** Qwen 9B – efficient LLM with superior reasoning capabilities, optimized inference latency <1s per turn, locally deployable
- **Training Approach:** Supervised preference elicitation via prompt engineering and dialogue evaluation; training data pairs user queries → optimal preference questions and city recommendations
- **Architecture Rationale:** (1) Better contextual understanding for complex preference trade-offs vs. smaller SLMs; (2) cost-effective inference on consumer GPU; (3) strong performance on multi-dimensional reasoning (city scoring); (4) on-premises deployment without external service dependencies

## 6. RAG Strategies & Implementation Pipeline

**Primary Strategy: Preference-Conditioned Retrieval**
1. User provides preference signal → embed into vector space
2. Query Chroma DB with dimension-filtered retrieval (e.g., climate-focused queries → geography/climate chunks)
3. Qwen 9B generates follow-up preference questions or synthesizes dimensions into explanations

**Ranking & Recommendation Generation:**
- Weighted aggregation across 8 dimensions using normalized city indices
- Preference weights derived from dialogue history (explainable multi-criteria decision analysis)
- Confidence threshold check: transition to recommendation mode at 70%+ preference confidence

**Quality Assurance & Fallbacks:**
- Chunk retrieval validation: if <3 relevant chunks returned, broaden query scope
- Dialogue state tracking: maintain preference-history DAG to detect contradictions, request clarification
