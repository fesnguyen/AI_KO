# RAG_search_technique.md

# Search Techniques for RAG

This document focuses on retrieval/search strategies, not query rewriting.

```text
Query
  ↓
Search
  ↓
Relevant Documents
  ↓
RAG
```

---

# 1. Dense Vector Search

Most common beginner approach.

---

## How It Works

```text
Query
 ↓
Embedding
 ↓
Vector Similarity
 ↓
Top K Documents
```

---

## Without LangChain

```python
query_embedding = embedder.encode(
    query
)

distances, indices = faiss_index.search(
    query_embedding,
    k=5
)
```

---

## Advantages

* Semantic search
* Synonym matching
* Concept matching

---

## Weaknesses

Bad at:

```text
RTX 5090
Invoice #12345
SKU-A-991
Error Code 0x80070005
```

Exact identifiers.

---

# 2. BM25 Search

Traditional keyword search.

Still heavily used.

---

## How It Works

Scores documents using:

* term frequency
* inverse document frequency

---

## Example

Query:

```text
RTX 5090 VRAM
```

BM25 excels.

---

## Python

```python
from rank_bm25 import BM25Okapi

tokenized_docs = [
    doc.split()
    for doc in docs
]

bm25 = BM25Okapi(
    tokenized_docs
)

scores = bm25.get_scores(
    query.split()
)
```

---

## Advantages

Excellent for:

* IDs
* Product names
* Error codes
* Exact terms

---

## Weaknesses

Poor semantic understanding.

---

# 3. Hybrid Search

Most common production retrieval strategy.

---

## Idea

Combine:

```text
Dense Search
+
BM25
```

---

## Why?

Dense catches meaning.

BM25 catches exact keywords.

---

## Example

Query:

```text
RTX 5090 memory
```

Dense finds:

```text
latest GPU memory architecture
```

BM25 finds:

```text
RTX 5090 VRAM specs
```

---

## Fusion

```python
final_score = (
    0.5 * dense_score
    +
    0.5 * bm25_score
)
```

---

## Production Reality

Many enterprise RAG systems use:

```text
Hybrid Search
+
Reranker
```

---

# 4. Reciprocal Rank Fusion (RRF)

Popular hybrid technique.

---

## Formula

```text
1/(k + rank)
```

---

## Example

Dense:

```text
Doc A Rank 1
Doc B Rank 2
```

BM25:

```text
Doc B Rank 1
Doc A Rank 2
```

---

## Fusion

Produces:

```text
Doc A
Doc B
```

based on combined ranking.

---

## Why?

Often better than score averaging.

---

# 5. Metadata Filtering

Search only relevant subsets.

---

## Example Metadata

```python
{
    "department": "finance",
    "year": 2025,
    "region": "US"
}
```

---

## Query

```text
Finance reports from 2025
```

---

## Search

```python
results = vectordb.search(
    query,
    filters={
        "department": "finance",
        "year": 2025
    }
)
```

---

## Huge ROI

Often improves retrieval more than changing embedding models.

---

# 6. Parent-Child Retrieval

Store small chunks.

Return larger context.

---

## Index

```text
Paragraph 1
Paragraph 2
Paragraph 3
```

---

## Retrieve

```text
Paragraph 2
```

---

## Return

```text
Entire Section
```

---

## Why?

Prevents fragmented context.

---

# 7. Multi-Vector Retrieval

Instead of:

```text
1 chunk
→ 1 embedding
```

Use:

```text
1 chunk
→ multiple embeddings
```

---

## Example

Store:

```text
Summary embedding

Keyword embedding

Full text embedding
```

---

## Search

Across all representations.

---

## Benefit

Higher recall.

---

# 8. ColBERT

Late Interaction Retrieval.

Widely used in state-of-the-art systems.

---

## Traditional Retrieval

```text
Chunk
 ↓
Single Vector
```

---

## ColBERT

```text
Chunk
 ↓
Token Embeddings
```

---

## Matching

Compare:

```text
query token
vs
document token
```

instead of:

```text
query vector
vs
document vector
```

---

## Benefits

* Better precision
* Better recall

---

## Cost

More storage.

More compute.

---

# 9. Graph Search

Knowledge graph retrieval.

---

## Example

```text
OpenAI
  ↓ founded_by
Sam Altman

Sam Altman
  ↓ attended
Stanford
```

---

## Search

Traverse relationships.

---

## Useful For

* Enterprise knowledge
* Legal systems
* Research databases

---

# 10. Graph RAG

Combine:

```text
Knowledge Graph
+
Vector Search
```

---

## Workflow

```text
Query
 ↓
Entity Extraction
 ↓
Graph Traversal
 ↓
Vector Search
 ↓
Merge Results
```

---

## Strength

Multi-hop reasoning.

---

# 11. RAPTOR Retrieval

Hierarchical retrieval.

---

## Build Tree

```text
Documents
 ↓
Chunk Summaries
 ↓
Higher Summaries
 ↓
Root Summary
```

---

## Search

Retrieve:

```text
Leaf Nodes
+
Summary Nodes
```

---

## Best For

* Books
* Reports
* Research papers

---

# 12. Recursive Retrieval

Retrieve repeatedly.

---

## Workflow

```text
Search
 ↓
Analyze
 ↓
Need More Info?
 ↓
Search Again
```

---

## Example

Question:

```text
How did NVIDIA's AI revenue grow?
```

First retrieval finds:

```text
Revenue report
```

Second retrieval finds:

```text
AI product breakdown
```

---

# 13. Agentic Search

LLM decides search strategy.

---

## Workflow

```text
Query
 ↓
Planner
 ↓
Search
 ↓
Evaluate
 ↓
Search Again
 ↓
Answer
```

---

## Example

Agent may decide:

```text
Search company reports

Search earnings call

Search news

Merge findings
```

---

# 14. Contextual Retrieval

Popularized recently.

---

## Problem

Chunk:

```text
Revenue increased 12%.
```

Without context:

```text
Revenue of what?
```

---

## Solution

Embed:

```text
This chunk comes from
NVIDIA 2025 Annual Report.

Revenue increased 12%.
```

---

## Benefit

Significantly better retrieval.

---

# 15. Search + Rerank

Probably the highest ROI upgrade.

---

## Workflow

```text
Retrieve Top 50
 ↓
Cross Encoder
 ↓
Keep Top 5
```

---

## Without Reranking

```text
Approximate relevance
```

---

## With Reranking

```text
Precise relevance
```

---

## Example

```python
scores = reranker.predict([
    (query, doc)
    for doc in docs
])

top_docs = sort_by_score(
    docs,
    scores
)
```

---

# Modern Production Stack

Most serious RAG systems look like:

```text
Query
 ↓
Query Rewrite
 ↓
Hybrid Search
      ├── BM25
      └── Dense Search
 ↓
Metadata Filtering
 ↓
RRF Fusion
 ↓
Reranker
 ↓
Context Compression
 ↓
LLM
```

---

# ROI Ranking

If building RAG today:

```text
1. Hybrid Search
2. Reranking
3. Metadata Filtering
4. Parent-Child Retrieval
5. Multi-Query Retrieval
6. Contextual Retrieval
7. ColBERT
8. Graph RAG
9. RAPTOR
10. Agentic Search
```

---

# What Most Companies Actually Deploy

Despite all the research papers, most production systems are:

```text
Hybrid Search
+
Metadata Filters
+
Reranker
+
Good Chunking
```

That combination often delivers 80-90% of the achievable retrieval quality with a fraction of the complexity.

---
