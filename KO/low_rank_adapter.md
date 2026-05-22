# LoRA and QLoRA — Mathematical Explanation

This file focuses on:
- Mathematical intuition
- Matrix decomposition
- Why LoRA works
- Why QLoRA works

Minimal marketing.
Minimal deployment discussion.

---

# 1. Start From Transformer Linear Layers

Transformer layers heavily use:

```python
y = Wx
```

Where:

| Symbol | Meaning |
|---|---|
| W | Weight matrix |
| x | Input vector |
| y | Output vector |

---

# 2. Example Dimensions

Suppose:

```python
W ∈ ℝ^(4096 × 4096)
```

This matrix contains:

```python
4096 × 4096
= 16,777,216 parameters
```

for ONE linear layer.

Large transformers contain many such layers.

---

# 3. Full Fine-Tuning

Traditional fine-tuning updates:

```python
W
```

directly.

So during training:

```python
W_new = W + ΔW
```

Where:

| Symbol | Meaning |
|---|---|
| W | Original pretrained weights |
| ΔW | Learned update |

---

# 4. Problem

The update matrix:

```python
ΔW
```

has SAME SIZE as:

```python
W
```

So memory cost remains huge.

---

# 5. LoRA Core Idea

LoRA assumes:

```text
The update matrix ΔW is low-rank.
```

Meaning:

```text
The important changes live in a much smaller subspace.
```

---

# 6. Matrix Rank Intuition

Suppose:

```python
ΔW ∈ ℝ^(4096 × 4096)
```

Maximum rank:

```python
4096
```

But LoRA assumes:

```python
rank(ΔW) << 4096
```

Example:

```python
rank = 8
```

or:

```python
rank = 16
```

---

# 7. Low-Rank Decomposition

Instead of learning:

```python
ΔW
```

directly,

LoRA factorizes it:

```python
ΔW ≈ BA
```

Where:

| Matrix | Shape |
|---|---|
| B | (d × r) |
| A | (r × k) |

and:

```python
r << d,k
```

---

# 8. Example Shapes

Suppose:

```python
W ∈ ℝ^(4096 × 4096)
```

Choose:

```python
r = 8
```

Then:

```python
B ∈ ℝ^(4096 × 8)

A ∈ ℝ^(8 × 4096)
```

Parameter count becomes:

```python
4096×8 + 8×4096
= 65,536
```

instead of:

```python
16,777,216
```

Huge reduction.

---

# 9. LoRA Forward Pass

Original layer:

```python
y = Wx
```

LoRA layer:

```python
y = Wx + BAx
```

Important:

```python
W
```

is frozen.

Only:

```python
A and B
```

are trainable.

---

# 10. Another Common Form

Often written as:

```python
W_new = W + α/r · BA
```

Where:

| Symbol | Meaning |
|---|---|
| α | Scaling factor |
| r | LoRA rank |

---

# 11. Why Scaling Exists

Without scaling:

```python
BA
```

may become too large.

Scaling stabilizes:
- gradients
- update magnitude
- optimization

---

# 12. Initialization Trick

Usually:

```python
B initialized randomly
A initialized near zero
```

So initially:

```python
BA ≈ 0
```

meaning:
- pretrained model behavior preserved at start

---

# 13. Why Low Rank Makes Sense

Pretrained transformers already contain:
- rich representations
- strong language understanding

Fine-tuning often only needs:
- small directional adjustments

Not a full rewrite of weights.

---

# 14. Geometric Interpretation

Original weight space:

```text
Very high dimensional
```

LoRA assumes:

```text
Useful adaptation lies in a lower-dimensional manifold.
```

So instead of updating:
```text
all directions
```

we update:
```text
a compressed subspace
```

---

# 15. Why LoRA Usually Targets Attention Layers

Common targets:

```python
q_proj
k_proj
v_proj
o_proj
```

Reason:
- attention strongly controls behavior
- adapting attention often sufficient

---

# 16. Memory Mathematics

Full fine-tuning stores:

```python
optimizer states
+
gradients
+
weights
```

for ALL parameters.

LoRA stores them only for:

```python
A and B
```

Huge reduction.

---

# 17. Backpropagation in LoRA

Gradient flows only through:

```python
A and B
```

while:

```python
W
```

remains frozen.

---

# 18. Important Mathematical Observation

LoRA does NOT reduce inference complexity much.

Because during inference:

```python
Wx + BAx
```

still needs computation.

But:

```python
BA
```

can often be merged into:

```python
W
```

after training.

---

# 19. Merging LoRA

After training:

```python
W_final = W + BA
```

Now inference becomes:

```python
y = W_final x
```

No separate LoRA computation needed.

---

# 20. Why Rank Matters

Small rank:

```python
r = 4
```

- smaller memory
- less expressive

Large rank:

```python
r = 64
```

- more expressive
- more parameters

---

# 21. Capacity Interpretation

Rank controls:
```text
How much adaptation information can be stored.
```

Higher rank:
- larger adaptation space

Lower rank:
- more constrained updates

---

# 22. Why LoRA Sometimes Surprisingly Works With Tiny Rank

Transformers are heavily overparameterized.

Often:
- only small directional corrections needed
- pretrained representations already strong

---

# 23. QLoRA Begins Here

QLoRA =

```text
Quantized LoRA
```

Key difference:

```python
W
```

is stored in:
```python
4-bit quantized form
```

while:

```python
A and B
```

remain trainable.

---

# 24. QLoRA Forward Pass

Mathematically:

```python
y = Q(W)x + BAx
```

Where:

| Symbol | Meaning |
|---|---|
| Q(W) | Quantized pretrained weights |
| BA | LoRA update |

---

# 25. Important Distinction

LoRA:
```python
W stored normally
```

QLoRA:
```python
W stored quantized
```

---

# 26. Why QLoRA Still Works

Quantized weights:

```python
Q(W)
```

already approximate original model well.

LoRA adapters then learn:
- corrections
- compensations
- task-specific shifts

---

# 27. QLoRA Memory Reduction

Suppose original model uses:

```python
FP16 = 16 bits
```

QLoRA stores base weights in:

```python
4 bits
```

Compression:

```python
16 / 4 = 4× smaller
```

while training only tiny LoRA matrices.

---

# 28. Why Gradients Are Still Stable

In QLoRA:
- quantized weights are frozen
- gradients mainly flow through LoRA adapters

So:
- low precision base model does not destabilize training heavily

---

# 29. NF4 Quantization

QLoRA commonly uses:

```python
NF4
```

Meaning:

```text
NormalFloat4
```

NF4 assumes:
- weights follow normal distribution

This improves reconstruction quality.

---

# 30. Double Quantization

QLoRA also introduced:

```text
Double Quantization
```

Even scaling constants themselves are quantized.

Further memory reduction.

---

# 31. Why QLoRA Was a Big Deal

Before QLoRA:

Fine-tuning large models required:
- huge VRAM
- enterprise GPUs

QLoRA showed:

```text
Quantized frozen base
+
small trainable adapters
```

works surprisingly well.

---

# 32. Full Mathematical Summary

---

# Full Fine-Tuning

```python
W_new = W + ΔW
```

Train ALL parameters.

---

# LoRA

```python
ΔW ≈ BA
```

Train low-rank matrices only.

Forward:

```python
y = Wx + BAx
```

---

# QLoRA

Quantized base model:

```python
y = Q(W)x + BAx
```

Where:
- Q(W) stored in 4-bit
- BA trainable in higher precision

---

# 33. Final Core Insight

LoRA works because:

```text
Fine-tuning updates often lie in a low-rank subspace.
```

QLoRA works because:

```text
Pretrained transformers tolerate quantization surprisingly well,
while LoRA adapters learn the needed corrections.
```

---

# 34. Final Intuition

Think of:

```python
W
```

as:
```text
A giant pretrained knowledge base
```

LoRA learns:
```text
Small directional edits
```

instead of rewriting the entire model.

That is the central mathematical intuition.

---

# Transformer Refresher

Original Transformer paper:

https://arxiv.org/abs/1706.03762