# Gemma 2B Tokenizer

## 1. Overview

`Gemma 2B` uses SentencePiece to perform subword tokenization using a BPE-like segmentation strategy.

Tokenizer converts raw text into:
- token IDs ([14562, 9821, 53])
- attention masks ([1, 1, 1, 0, 0])
- special tokens (<BOS>, <EOS>, <PAD>)

before feeding data into the transformer.

---

# 2. Why Tokenization Exists

Neural networks cannot process raw text directly.

Example:

```text
"Hello world"
```

becomes:

```python
[2, 10994, 2416, 1]
```

Each integer maps to:
- vocabulary entry
- embedding vector

---

# 3. Core Idea of Subword Tokenization

Tokenizer splits text into:
- full words
- partial words
- sometimes characters

Example:

```text
unbelievable
```

might become:

```text
["un", "believ", "able"]
```

---

## Why Subwords Matter

Without subwords:
- vocabulary becomes enormous
- unknown words become impossible to handle

Subwords solve:
- rare words
- spelling variations
- multilingual support
- code tokens

---

# 4. SentencePiece (Critical)

Gemma uses SentencePiece.

Unlike classic tokenizers:
- it does NOT rely heavily on whitespace splitting
- whitespace becomes part of tokens

Example:

```text
" hello"
```

internally may become:

```text
"▁hello"
```

The `▁` symbol means:
> token begins after whitespace

---

# 5. Vocabulary

Gemma tokenizer has a fixed vocabulary size.

Approximate:
- ~256k tokens (depends on version)

Vocabulary contains:
- words
- subwords
- punctuation
- symbols
- code fragments
- multilingual pieces

---

# 6. Special Tokens

| Token | Purpose |
|------|------|
| `<bos>` | beginning of sequence |
| `<eos>` | end of sequence |
| `<pad>` | padding |
| `<unk>` | unknown token |

---

# 7. BOS Token (Very Important)

Gemma commonly expects:

```text
<bos>
```

at the start of sequences.

Without BOS:
- generation quality may drop
- training distribution mismatch occurs

---

# 8. EOS Token

EOS means:
> sequence ended

During training:
- model predicts next token until EOS

During inference:
- generation stops at EOS

---

# 9. Padding

Different sequences have different lengths.

Example:

```python
[1, 5, 9]
[1, 8]
```

Need equal batch size:

```python
[1, 5, 9]
[1, 8, 0]
```

where:

```text
0 = pad token
```

---

# 10. Attention Mask

Padding should not affect attention.

Example:

```python
[1, 1, 1]
[1, 1, 0]
```

- `1` = real token
- `0` = ignore

---

# 11. Token IDs

Tokenizer maps tokens into integer IDs.

Example:

```text
"hello" → 10994
```

Transformer never sees raw text directly.

It only sees tensors:

```python
tensor([[2, 10994, 2416]])
```

---

# 12. Embedding Layer Connection

Each token ID indexes embedding matrix:

$$
\text{Embedding}(token\_id) \rightarrow vector
$$

Example:

```python
10994 → [0.21, -1.3, ...]
```

Embeddings are learned during training.

---

# 13. Unknown Tokens (`<unk>`)

Rare in SentencePiece.

Reason:
- unknown words can be decomposed into subwords

Example:

```text
"hypertransformermagic"
```

becomes:

```text
["hyper", "transform", "er", "magic"]
```

instead of `<unk>`.

---

# 14. Why Token Count Matters

Transformers process tokens, NOT words.

Example:

```text
"Hello"
```

may be:
- 1 token

while:

```text
"antidisestablishmentarianism"
```

may become:
- many tokens

---

## Consequences

More tokens means:
- more VRAM usage
- slower training
- slower inference
- shorter effective context

---

# 15. Context Length

Gemma has maximum token window.

Example:
- 8k context

Meaning:

```text
input_tokens + generated_tokens <= context_limit
```

---

# 16. Important Tokenization Quirks

## ❗ Spaces Matter

These tokenize differently:

```text
"hello"
```

vs

```text
" hello"
```

because SentencePiece encodes whitespace.

---

## ❗ Capitalization Matters

```text
"cat"
```

vs

```text
"Cat"
```

can produce different token IDs.

---

## ❗ Repeated Spaces Matter

```text
"hello world"
```

vs

```text
"hello  world"
```

produce different token sequences.

---

## ❗ Formatting Matters

LLMs are sensitive to formatting patterns seen during training.

Example:

```text
User:
Assistant:
```

often performs better than random formats.

---

# 17. Base vs Instruction-Tuned Model

## Base Model
- raw continuation prediction

## Instruction-Tuned (`-it`)
- trained on conversational patterns

Tokenizer stays the same,
BUT prompting style differs.

---

# 18. HuggingFace Usage

Load tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2b"
)
```

---

## Tokenization

```python
tokens = tokenizer(
    "Hello world",
    return_tensors="pt"
)
```

Output:

```python
{
    "input_ids": ...,
    "attention_mask": ...
}
```

---

# 19. Decoding

Convert token IDs back into text:

```python
tokenizer.decode(ids)
```

---

# 20. Important Fine-Tuning Rule

You almost always:
- keep original tokenizer
- keep original vocabulary

Changing tokenizer after pretraining can:
- break embeddings
- destroy pretrained alignment
- reduce performance heavily

---

# 21. Why Tokenizer Design Is Hard

Tokenizer must balance:
- vocabulary size
- compression efficiency
- multilingual support
- code support
- memory efficiency

---

## Too Small Vocabulary

Problems:
- too many tokens
- slower inference
- larger sequence lengths

---

## Too Large Vocabulary

Problems:
- sparse learning
- harder embedding training
- wasted parameters

---

# 22. BPE vs SentencePiece

| Feature | BPE | SentencePiece |
|------|------|------|
| Needs pre-tokenization | Often yes | No |
| Handles whitespace directly | Usually no | Yes |
| Raw text training | Limited | Yes |

Gemma benefits from:
- whitespace-aware tokenization
- multilingual robustness

---

# 23. Common Beginner Mistakes

## ❌ Assuming 1 word = 1 token

False.

---

## ❌ Ignoring BOS/EOS

Can reduce generation quality.

---

## ❌ Changing tokenizer after training

Usually catastrophic.

---

## ❌ Ignoring tokenizer during fine-tuning

Tokenizer mismatch can destroy performance.

---

# 24. Mental Model

Tokenizer is:

> a compression + segmentation system converting text into neural-readable units

Transformer never understands words directly.

It only understands:
- token IDs
- embeddings
- probabilities

---

# 25. Key Takeaway

Tokenizer quality directly affects:
- efficiency
- VRAM usage
- context length
- multilingual capability
- reasoning quality
- code generation quality

Tokenizer design is far more important than most beginners realize.