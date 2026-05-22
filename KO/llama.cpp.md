# llama.cpp Fundamentals

This file focuses on:
- What llama.cpp actually is
- Core architecture
- How it relates to GGUF
- How it connects to Ollama
- How it connects to Unsloth
- The complete beginner mental model

Minimal hype.
Focus on fundamentals.

---

# 1. What is llama.cpp?

llama.cpp is a lightweight inference engine for large language models (LLMs).

It is mainly written in:
- C
- C++

Core purpose:

```text
Run LLMs efficiently on local hardware.
```

Especially:
- CPUs
- consumer GPUs
- laptops

---

# 2. Important Clarification

llama.cpp is NOT:
- a model
- a training framework
- a dataset

It is:

```text
An inference runtime.
```

Meaning:
```text
A system that executes trained models.
```

---

# 3. The Big Picture

Modern LLM ecosystem simplified:

```text
Training
    ↓
Fine-tuning
    ↓
Quantization
    ↓
Inference
```

llama.cpp belongs mainly to:

```text
Inference
```

---

# 4. Core Pipeline

Typical llama.cpp workflow:

```text
Pretrained Model
    ↓
Quantization
    ↓
GGUF file
    ↓
llama.cpp
    ↓
Text generation
```

---

# 5. Why llama.cpp Became Important

Originally:
- LLMs required expensive GPUs
- deployment was difficult
- inference frameworks were heavy

llama.cpp changed this by making:
- local inference practical
- CPU inference fast
- quantized inference efficient

---

# 6. Fundamental Transformer Computation

Transformers mostly do:

```python
y = Wx
```

and:

```python
Attention(Q, K, V)
```

These become:
- matrix multiplications
- tensor operations

llama.cpp is heavily optimized for these operations.

---

# 7. What llama.cpp Actually Executes

Suppose model layer:

```python
y = Wx
```

llama.cpp:
- loads weights
- performs matrix multiplication
- applies activations
- computes attention
- predicts next token

It is essentially:
```text
A highly optimized transformer executor.
```

---

# 8. Why C/C++ Matters

Python is convenient but slower.

llama.cpp uses:
- low-level memory control
- SIMD instructions
- optimized CPU kernels

This allows:
- high speed
- low memory overhead

---

# 9. CPU Optimization

llama.cpp heavily uses:
- AVX
- AVX2
- AVX512
- ARM NEON

These are CPU vector instructions.

---

# 10. Vectorization Intuition

Instead of:

```python
a*b
a*b
a*b
```

one-by-one,

SIMD computes many operations simultaneously.

Very important for:
- matrix multiplication
- transformer inference

---

# 11. Why Quantization Matters So Much

Transformer weights are huge.

Example:

| Model | FP16 Size |
|---|---|
| 7B | ~14GB |
| 13B | ~26GB |

llama.cpp became practical because of:
- GGUF
- quantization
- memory optimization

---

# 12. Core Quantization Idea

Instead of:

```python
float16
```

weights become:

```python
4-bit
5-bit
8-bit
```

This reduces:
- RAM
- VRAM
- bandwidth

---

# 13. Why Bandwidth Matters

Transformer inference is often memory-bandwidth limited.

Meaning:
```text
Moving weights is expensive.
```

Smaller weights:
- move faster
- cache better
- reduce memory pressure

---

# 14. Memory Mapping (Very Important)

llama.cpp supports:

```text
mmap
```

(memory mapping)

Meaning:
- model stays partially on disk
- OS loads only needed sections

This reduces RAM usage dramatically.

---

# 15. GGUF Relationship

GGUF is the model format.

llama.cpp is the runtime.

Relationship:

```text
GGUF = optimized model file
llama.cpp = engine that runs it
```

---

# 16. Why GGUF Was Created

PyTorch checkpoints are:
- large
- fragmented
- training-oriented

GGUF is:
- inference-oriented
- quantization-friendly
- memory-efficient

---

# 17. Token Generation Process

Transformer inference loop:

```text
Input tokens
    ↓
Embedding lookup
    ↓
Attention layers
    ↓
MLP layers
    ↓
Next token logits
    ↓
Sampling
    ↓
Generate token
```

llama.cpp executes this loop repeatedly.

---

# 18. KV Cache

One of the most important inference concepts.

Transformer attention needs previous tokens.

Without KV cache:
- recompute everything every step

Very slow.

---

# 19. KV Cache Mathematics

Attention:

```python
softmax(QKᵀ)V
```

Previous:
- K vectors
- V vectors

are cached.

So future tokens reuse them.

Huge speed improvement.

---

# 20. Why Context Length Costs Memory

Longer context:
- larger KV cache

Memory grows approximately:

```text
O(sequence_length)
```

for KV cache storage.

---

# 21. Why Attention is Expensive

Attention matrix:

```python
QKᵀ
```

has complexity:

```text
O(n²)
```

where:
```python
n = sequence length
```

This is why:
- long contexts are expensive
- VRAM usage rises quickly

---

# 22. GPU Offloading

llama.cpp can split work:

```text
Some layers on GPU
Some layers on CPU
```

This is called:
```text
GPU offloading
```

---

# 23. Example

```bash
./main -m model.gguf -ngl 20
```

Where:

```text
-ngl
```

means:
```text
Number of GPU layers
```

---

# 24. Why Hybrid Execution Matters

Many users:
- lack enough VRAM
- have strong CPUs

Hybrid execution allows:
- larger models
- lower VRAM usage

---

# 25. Sampling

Transformer outputs logits.

Example:

```python
[2.1, 5.4, 0.7]
```

Sampling converts logits into next tokens.

Common methods:
- greedy
- top-k
- top-p
- temperature

---

# 26. Temperature

Controls randomness.

Mathematically:

```python
softmax(logits / T)
```

Where:

| T | Effect |
|---|---|
| Low | More deterministic |
| High | More random |

---

# 27. Relationship to Ollama

Ollama is NOT a replacement for llama.cpp.

Instead:

```text
Ollama uses llama.cpp internally.
```

Important beginner insight.

---

# 28. Ollama Simplified

Without Ollama:

```text
Manual GGUF management
Manual commands
CLI complexity
```

With Ollama:

```text
Easy model management
Easy APIs
Simple commands
```

---

# 29. Relationship Diagram

```text
GGUF Model
    ↓
llama.cpp
    ↓
Ollama
```

---

# 30. Ollama's Role

Ollama acts like:
- model manager
- deployment wrapper
- API server

Built on top of inference runtimes like:
- llama.cpp

---

# 31. Example Ollama Command

```bash
ollama run mistral
```

Internally:
- model loads
- llama.cpp inference executes
- tokens generated

---

# 32. Relationship to Unsloth

Completely different role.

---

# Unsloth

Mainly for:
```text
Training / Fine-tuning optimization
```

---

# llama.cpp

Mainly for:
```text
Inference optimization
```

---

# 33. Full Ecosystem Picture

---

# Training Side

```text
Transformers
+
PyTorch
+
Unsloth
+
QLoRA
```

Purpose:
- train
- fine-tune

---

# Inference Side

```text
GGUF
+
llama.cpp
+
Ollama
```

Purpose:
- deploy
- run locally

---

# 34. Beginner Mental Model

---

# Unsloth

```text
Makes training cheaper/faster.
```

---

# GGUF

```text
Compressed deployable model format.
```

---

# llama.cpp

```text
Runs the model efficiently.
```

---

# Ollama

```text
Makes local deployment easy.
```

---

# 35. Typical Real Workflow

---

## Fine-Tuning

```text
Base model
    ↓
Unsloth + QLoRA
    ↓
Fine-tuned model
```

---

## Deployment

```text
Fine-tuned model
    ↓
Convert to GGUF
    ↓
Run with llama.cpp or Ollama
```

---

# 36. Why Conversion is Needed

Training frameworks use:
- PyTorch tensors
- full precision weights

Inference frameworks prefer:
- quantized formats
- optimized layouts

So conversion happens.

---

# 37. Why llama.cpp Became Revolutionary

Because it showed:

```text
Consumer hardware can run serious LLMs.
```

This changed:
- AI learning
- local AI
- private inference
- offline assistants

massively.

---

# 38. Important Limitation

llama.cpp is AMAZING for inference,
but not intended for:
- large-scale distributed training
- advanced backpropagation workflows

Different engineering problem.

---

# 39. Core Engineering Insight

Modern AI systems separate:

| Stage | Tooling |
|---|---|
| Training | PyTorch ecosystem |
| Deployment | Optimized runtimes |

llama.cpp belongs strongly to:
```text
deployment/runtime engineering
```

---

# 40. Final Mental Model

Think of:

---

# PyTorch

```text
Flexible research laboratory
```

---

# Unsloth

```text
Efficient fine-tuning accelerator
```

---

# GGUF

```text
Compressed deployable model package
```

---

# llama.cpp

```text
Optimized transformer execution engine
```

---

# Ollama

```text
User-friendly deployment layer
```

---

# 41. Final Notes

Understanding llama.cpp teaches important AI engineering concepts:
- inference optimization
- quantization
- memory bandwidth
- runtime systems
- deployment engineering

This is a major part of real-world LLM engineering.

Training models is only half the story.

Efficiently RUNNING them matters just as much.

---

# Useful Tools

| Tool | Purpose |
|---|---|
| llama.cpp | Inference runtime |
| Ollama | Easy deployment |
| Unsloth | Efficient fine-tuning |
| Transformers | Model training ecosystem |
| GGUF | Optimized model format |

---

# Transformer Refresher

Original Transformer paper:

https://arxiv.org/abs/1706.03762