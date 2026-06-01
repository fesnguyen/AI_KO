# RAG.md

# Retrieval-Augmented Generation (RAG)

RAG = Retrieve relevant documents and inject them into the LLM context before generation.

```text
Question
    ↓
Embedding
    ↓
Vector Search
    ↓
Relevant Chunks
    ↓
Prompt Construction
    ↓
LLM Answer
```

---

# Basic RAG

## Indexing

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

docs = [
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(docs)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype=np.float32))
```

---

## Retrieval

```python
query = "capital of japan"

query_embedding = model.encode([query])

distances, indices = index.search(
    np.array(query_embedding, dtype=np.float32),
    k=3
)

context = "\n".join(
    docs[i]
    for i in indices[0]
)
```

---

## Generation

```python
prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""
```

---

# Standard Production RAG

Nobody uses:

```text
query
→ vector search
→ LLM
```

anymore.

Typical production pipeline:

```text
Query
 ↓
Query Rewrite
 ↓
Hybrid Search
 ↓
Reranker
 ↓
Context Compression
 ↓
LLM
```

---

# Query Rewriting

Improve retrieval quality.

User:

```text
When was he born?
```

Bad retrieval.

Rewrite:

```text
When was Elon Musk born?
```

Example:

```python
rewrite_prompt = f"""
Rewrite for retrieval:

{query}
"""
```

---

# Multi-Query Retrieval

Generate multiple search queries.

```python
queries = [
    query,
    "alternative wording",
    "related keywords"
]
```

Retrieve for all.

Merge results.

Popularized by LangChain.

---

# Hybrid Search

Combine:

* Dense Retrieval
* BM25 Keyword Search

```text
score =
0.5 * dense_score
+
0.5 * bm25_score
```

Why?

Vector search misses:

* IDs
* model numbers
* exact keywords

BM25 catches them.

---

# Example

Dense search:

```text
RTX 5090
```

might match:

```text
latest GPU
```

BM25 retrieves exact:

```text
RTX 5090
```

documents.

---

# Reranking

One of the highest ROI improvements.

---

## Step 1

Retrieve:

```text
Top 50
```

---

## Step 2

Cross-encoder reranker:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

scores = reranker.predict(
    [(query, doc) for doc in candidates]
)
```

---

## Step 3

Keep:

```text
Top 5
```

---

# Why?

Embedding retrieval is approximate.

Reranker performs exact relevance scoring.

---

# Context Compression

Retrieved chunks often contain junk.

Compress before LLM.

---

## Example

Original:

```text
2000 tokens
```

Compressed:

```text
300 tokens
```

Methods:

* LLM summarization
* extractive filtering
* keyword extraction

---

# Parent-Child Retrieval

Store:

```text
small chunks
```

Retrieve:

```text
large parent chunk
```

---

## Index

```text
paragraph chunks
```

---

## Return

```text
whole section
```

instead of tiny fragments.

Improves coherence.

---

# Metadata Filtering

Attach metadata.

```python
{
    "source": "finance",
    "year": 2025
}
```

Filter:

```python
year >= 2024
```

before retrieval.

---

# Self-Query Retrieval

LLM extracts filters.

User:

```text
Show finance reports from 2025
```

Generated filter:

```python
{
    "department": "finance",
    "year": 2025
}
```

Applied directly to vector DB.

---

# Recursive Retrieval

If context insufficient:

```text
Retrieve
→ Analyze
→ Retrieve Again
→ Answer
```

Agentic RAG.

---

# Graph RAG

Instead of vectors:

```text
Entities
Relationships
Knowledge Graph
```

Example:

```text
OpenAI
   ↓ founded by
Sam Altman
```

Useful for:

* enterprise docs
* research
* legal

---

# Knowledge Graph + Vector Search

Common hybrid:

```text
Graph Search
    +
Vector Search
```

Graph:

* relationships

Vectors:

* semantic similarity

---

# RAPTOR

Recursive summarization tree.

```text
Documents
   ↓
Summaries
   ↓
Higher Summaries
   ↓
Root Summary
```

Retrieve at multiple levels.

Good for:

* books
* long reports
* research papers

---

# ColBERT

Late interaction retrieval.

Instead of:

```text
1 embedding
```

Use:

```text
token embeddings
```

for retrieval.

Benefits:

* much higher recall
* better precision

Popular in state-of-the-art RAG systems.

---

# Contextual Retrieval

Popularized recently.

Instead of embedding:

```text
chunk only
```

Embed:

```text
chunk
+
document summary
```

Example:

```text
This chunk comes from:
NVIDIA 2025 annual report.
```

Retrieval improves dramatically.

---

# Agentic RAG

Pipeline:

```text
Question
 ↓
Planner
 ↓
Retrieve
 ↓
Analyze
 ↓
Retrieve More
 ↓
Answer
```

Instead of:

```text
single retrieval
```

Agent decides what to search next.

---

# Web + RAG

Modern assistants:

```text
Vector DB
+
Web Search
+
Reranker
```

Workflow:

```text
User Query
 ↓
Search Internal Docs
 ↓
Search Web
 ↓
Merge
 ↓
Rerank
 ↓
Answer
```

---

# Chunking Strategies

Bad chunking destroys RAG quality.

---

## Fixed

```text
512 tokens
```

Simple.

---

## Overlapping

```text
512 size
128 overlap
```

Most common.

---

## Semantic Chunking

Split by:

* headings
* paragraphs
* sections

instead of token count.

Usually better.

---

# Embedding Models

Common choices:

```python
BAAI/bge-large-en-v1.5
```

```python
intfloat/e5-large-v2
```

```python
nomic-ai/nomic-embed-text-v1
```

```python
sentence-transformers/all-MiniLM-L6-v2
```

---

# Modern RAG Stack (2026)

```text
User Query
    ↓
Query Rewrite
    ↓
Multi Query Expansion
    ↓
Hybrid Search
        ├─ BM25
        └─ Dense Retrieval
    ↓
Reranker
    ↓
Metadata Filter
    ↓
Context Compression
    ↓
LLM
```

---

# Highest ROI Improvements

If basic RAG already works:

1. Better chunking
2. Hybrid search
3. Reranker
4. Query rewriting
5. Metadata filtering
6. Context compression
7. Multi-query retrieval
8. Agentic retrieval
9. Graph RAG
10. ColBERT

---

# Rule of Thumb

Most beginners improve:

```text
Embedding Model
```

when the real bottleneck is:

```text
Retrieval Quality
```

In production, reranking and hybrid search often improve answer quality more than switching to a larger embedding model.

---
