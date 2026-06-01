# RAG_query_rewrite.md

# Query Rewrite Techniques

Query rewriting improves retrieval quality before vector search.

```text
User Query
    ↓
Query Rewrite
    ↓
Retriever
    ↓
RAG
```

---

# 1. Conversational Rewrite

## Problem

User:

```text
When was he born?
```

History:

```text
Tell me about Elon Musk.
```

Retriever sees:

```text
When was he born?
```

Poor retrieval.

---

## Rewrite

```text
When was Elon Musk born?
```

---

## Without LangChain

```python
prompt = f"""
Rewrite the query for retrieval.

History:
{history}

Query:
{query}

Only output rewritten query.
"""

rewritten_query = llm(prompt)
```

---

## LangChain

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Rewrite query for retrieval.

History:
{history}

Query:
{query}
""")

rewritten = llm.invoke(
    prompt.format(
        history=history,
        query=query
    )
)
```

---

# 2. Multi Query Retrieval

Generate multiple search queries.

---

## Example

Input:

```text
How does RAG work?
```

Generate:

```text
How does Retrieval Augmented Generation work?

RAG retrieval pipeline

Vector search in RAG

Document retrieval for LLMs
```

---

## Without LangChain

```python
prompt = f"""
Generate 4 search queries.

Question:
{query}

Return one query per line.
"""

queries = llm(prompt).split("\n")
```

---

## Retrieval

```python
all_docs = []

for q in queries:
    docs = vector_store.search(q)
    all_docs.extend(docs)
```

---

## LangChain

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=llm
)
```

---

# 3. HyDE

Hypothetical Document Embedding.

One of the most effective rewrite techniques.

---

## Idea

Question:

```text
What are benefits of Graph RAG?
```

Generate:

```text
Graph RAG combines vector retrieval
with knowledge graph traversal...
```

Embed generated answer.

Retrieve using that embedding.

---

## Without LangChain

```python
hyde_prompt = f"""
Write a detailed answer.

Question:
{query}
"""

fake_doc = llm(hyde_prompt)

query_embedding = embedder.encode(
    fake_doc
)

results = vectordb.search(
    query_embedding
)
```

---

## LangChain

```python
from langchain.chains import HypotheticalDocumentEmbedder
```

(Usually custom implementations are preferred.)

---

# 4. Step-Back Prompting

Generate a higher-level query.

---

## Input

```text
How does RAPTOR improve retrieval?
```

---

## Rewrite

```text
What are hierarchical retrieval systems?
```

---

## Code

```python
prompt = f"""
Generate a broader version
of the question.

Question:
{query}
"""

step_back_query = llm(prompt)
```

---

## Retrieval

```python
docs1 = retrieve(query)

docs2 = retrieve(
    step_back_query
)

merged = docs1 + docs2
```

---

# 5. Query Decomposition

Split a complex query.

---

## Input

```text
Compare Graph RAG and RAPTOR.
```

---

## Rewrite

```text
What is Graph RAG?

What is RAPTOR?

Graph RAG advantages

RAPTOR advantages
```

---

## Without LangChain

```python
prompt = f"""
Break into retrieval questions.

Question:
{query}
"""

sub_queries = llm(prompt).split("\n")
```

---

## Retrieval

```python
all_docs = []

for q in sub_queries:
    docs = retrieve(q)
    all_docs.extend(docs)
```

---

# 6. Keyword Expansion

Useful for BM25.

---

## Input

```text
RTX 5090 memory
```

---

## Rewrite

```text
RTX 5090 VRAM memory specifications
```

---

## Code

```python
prompt = f"""
Expand keywords.

Query:
{query}
"""

expanded = llm(prompt)
```

---

# 7. Metadata Extraction

Instead of rewriting.

Extract filters.

---

## Input

```text
Finance reports from 2025
```

---

## Output

```json
{
  "department": "finance",
  "year": 2025
}
```

---

## Code

```python
prompt = f"""
Extract metadata filters.

Query:
{query}

Output JSON.
"""

filters = json.loads(
    llm(prompt)
)
```

---

## Search

```python
vectordb.search(
    query,
    filters=filters
)
```

---

# 8. Agentic Rewrite

Retriever retries when retrieval quality is poor.

---

```text
Query
 ↓
Retrieve
 ↓
Low score?
 ↓
Rewrite
 ↓
Retrieve Again
```

---

## Example

First query:

```text
CUDA memory
```

Results weak.

Rewrite:

```text
CUDA GPU memory allocation
optimization techniques
```

---

## Code

```python
docs = retrieve(query)

if max_score < 0.6:
    rewritten = llm(
        rewrite_prompt
    )

    docs = retrieve(
        rewritten
    )
```

---

# Production Query Rewrite Pipeline

Typical setup:

```text
User Query
      ↓
Conversation Rewrite
      ↓
Multi Query Expansion
      ↓
Metadata Extraction
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

# Most Valuable Techniques

Ranked by practical impact:

```text
1. Conversational Rewrite
2. Multi Query Retrieval
3. Metadata Extraction
4. Query Decomposition
5. HyDE
6. Step-Back Prompting
7. Agentic Rewrite
```

---

# What Most Production Teams Actually Use

```text
Conversational Rewrite
+
Multi Query
+
Hybrid Search
+
Reranker
```

Everything else is usually added only after retrieval quality becomes a bottleneck.
