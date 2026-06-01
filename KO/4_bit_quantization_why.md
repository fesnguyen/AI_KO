# Why 4-Bit Quantization Works — Mathematical Explanation

This file focuses on the mathematical intuition behind why quantization works for neural networks and transformers.

The key question:

```text
How can a model still work after reducing weights to only 4 bits?
```

The answer:
```
Quantizer divide weights into groups of 64 weights each which share the same scale(Common system)
Support group 1 [0.12, 0.18, -0.09, 0.25,...] * a scale number like 0.03
=> So Model only need to store [4, 6, -3, 8,...]
Because 4 * 0.03 = 0.12, 6 * 0.03 = 0.18,...
Saving lost of bits
```

Trade-off:
```
* One shared scale for 64 numbers cannot perfectly represent all values simultaneously.
* So some weights get approximated poorly.
=> slight reduce the accuracy, luckily, NNs are fundamentally approximation machines.
```

---

# 1. Start From the Core Neural Network Equation

A neural network layer is mostly:

```python
y = Wx + b
```

Where:

| Symbol | Meaning |
|---|---|
| W | Weight matrix |
| x | Input vector |
| b | Bias |
| y | Output |

---

# 2. Example Without Quantization

Suppose:

```python
W = [0.82, -0.44, 1.21]

x = [2.0, 1.5, -0.7]
```

Dot product:

```python
y = (0.82)(2.0)
  + (-0.44)(1.5)
  + (1.21)(-0.7)
```

Compute:

```python
y = 1.64 - 0.66 - 0.847
```

Final:

```python
y = 0.133
```

---

# 3. Now Quantize the Weights

Suppose we compress:

```python
0.82  -> 0.8
-0.44 -> -0.4
1.21  -> 1.2
```

Now compute again:

```python
y_quantized =
(0.8)(2.0)
+ (-0.4)(1.5)
+ (1.2)(-0.7)
```

Compute:

```python
y_quantized =
1.6 - 0.6 - 0.84
```

Final:

```python
y_quantized = 0.16
```

---

# 4. Compare The Error

Original:

```python
0.133
```

Quantized:

```python
0.16
```

Difference:

```python
0.027
```

This error is tiny.

The neuron behavior remains very similar.

---

# 5. Key Insight:
# Neural Networks Are Approximate Functions

Neural networks are NOT exact symbolic systems.

They approximate functions:

```python
f(x) ≈ y
```

Tiny perturbations in weights often produce:
- Tiny output changes
- Similar activations
- Similar predictions

---

# 6. Error Propagation is Often Small

Suppose quantization introduces error:

```python
W_quantized = W + ε
```

Where:

| Symbol | Meaning |
|---|---|
| ε | Quantization error |

Then:

```python
y_quantized = (W + ε)x
```

Expand:

```python
y_quantized = Wx + εx
```

Notice:

```python
Wx
```

is the original output.

And:

```python
εx
```

is the extra quantization error.

---

# 7. Why Error Often Stays Small

Because:

```python
ε
```

is usually small.

So:

```python
εx
```

also stays small.

This is one of the central mathematical reasons quantization works.

---

# 8. High Dimensional Averaging Effect

Transformers operate in huge dimensions.

Example:

```python
hidden_size = 2048
```

or:

```python
4096
8192
```

A neuron output becomes:

```python
y = Σ(wᵢxᵢ)
```

Many tiny quantization errors:
- Mix together
- Average out
- Partially cancel

This creates strong robustness.

---

# 9. Statistical Cancellation

Suppose errors are random:

```python
ε₁, ε₂, ε₃ ...
```

Some positive:

```python
+0.01
```

Some negative:

```python
-0.01
```

Summation effect:

```python
Σ εᵢ ≈ 0
```

This cancellation is extremely important.

---

# 10. Why Large Models Quantize Better

Large models contain huge redundancy.

Suppose:

```python
7 billion parameters
```

Knowledge is distributed across:
- Millions of neurons
- Many layers
- Multiple attention heads

So losing tiny precision in individual weights:
- Rarely destroys overall behavior

---

# 11. Distributed Representation

In deep learning:

```text
Knowledge is NOT stored in one weight.
```

Instead:

```text
Patterns are distributed across many parameters.
```

Example:

The concept:
```text
"dog"
```

is not stored in:
```python
one neuron
```

It emerges from:
- many layers
- many activations
- many interactions

This distributed storage makes compression possible.

---

# 12. Matrix Multiplication is Naturally Robust

Transformer computation is mostly:

```python
QKᵀ
```

and:

```python
softmax(QKᵀ)V
```

These are giant matrix operations.

Tiny numerical perturbations rarely change:
- the highest attention scores
- the dominant semantic patterns

---

# 13. Softmax Dampens Small Differences

Softmax:

```python
softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)
```

Small quantization changes often produce:
- similar probabilities
- similar token rankings

unless logits are extremely close.

---

# 14. Why Attention Still Works

Suppose original attention scores:

```python
[2.1, 5.8, 1.4]
```

After quantization:

```python
[2.0, 5.7, 1.5]
```

Largest value remains:

```python
5.7
```

Attention behavior barely changes.

---

# 15. Quantization is NOT Just Rounding

Modern quantization uses scaling.

Instead of storing:

```python
0.183
```

directly,

we store:

```python
quantized_value
+
scale_factor
```

---

# 16. Core Quantization Equation

Suppose:

```python
w = original weight
```

Quantized representation:

```python
q = round(w / s)
```

Where:

| Symbol | Meaning |
|---|---|
| q | Quantized integer |
| s | Scale factor |

Recovery:

```python
ŵ = q × s
```

Where:

```python
ŵ
```

is the reconstructed approximate weight.

---

# 17. Example

Suppose:

```python
w = 0.82
s = 0.1
```

Quantize:

```python
q = round(0.82 / 0.1)
q = round(8.2)
q = 8
```

Recover:

```python
ŵ = 8 × 0.1
ŵ = 0.8
```

Very close to original.

---

# 18. Why Scale Factors Matter

Without scaling:

4-bit only stores:

```python
0 → 15
```

Very limited.

Scaling allows those integers to represent:
- tiny ranges
- large ranges
- adaptive ranges

This dramatically increases precision.

---

# 19. Group Quantization

Modern methods split weights into groups.

Example:

```python
128 weights per group
```

Each group gets:
- its own scale
- its own normalization

This greatly reduces error.

---

# 20. Why NF4 Works Better

NF4 assumes weights follow:

```python
Normal Distribution
```

Which is usually true in neural networks.

Most weights cluster near:

```python
0
```

So NF4 allocates:
- more precision near zero
- less precision for rare extremes

This is mathematically efficient.

---

# 21. Information Theory Perspective

Large models are overparameterized.

Meaning:

```text
The network contains more information capacity than necessary.
```

Quantization removes:
- redundant precision
- unnecessary detail

while preserving:
- important structure

---

# 22. Training vs Inference

During training:

Tiny numerical changes affect:
- gradients
- optimization trajectory

Higher precision matters.

---

During inference:

Weights are already optimized.

We only perform:

```python
forward pass
```

Inference is much more tolerant to approximation.

---

# 23. Why 2-Bit Starts Breaking Down

At very low precision:

```python
2^2 = 4 values
```

Quantization error becomes too large.

Then:

```python
εx
```

is no longer small.

Predictions degrade noticeably.

---

# 24. Why 4-Bit Became the Sweet Spot

4-bit gives:

```python
2^4 = 16 values
```

Combined with:
- scaling
- grouping
- normalization

this becomes sufficient for:
- semantic understanding
- text generation
- reasoning

while massively reducing memory.

---

# 25. VRAM Mathematics

FP16:

```python
16 bits per weight
```

4-bit:

```python
4 bits per weight
```

Compression ratio:

```python
16 / 4 = 4× smaller
```

Approximate memory reduction:

```python
75%
```

---

# 26. Final Core Mathematical Insight

Quantization works because:

```text
Neural networks depend more on global statistical structure
than exact individual weight precision.
```

Tiny weight perturbations usually produce:
- small activation changes
- similar attention patterns
- nearly identical predictions

This robustness is one of the hidden superpowers of deep learning.

---

# 27. Final Intuition

A transformer is NOT:
```text
a fragile calculator
```

It is:
```text
a massive distributed statistical system
```

That is why:
- approximation works
- compression works
- quantization works

far better than most people initially expect.

---

# Transformer Refresher

Original Transformer paper:

https://arxiv.org/abs/1706.03762
