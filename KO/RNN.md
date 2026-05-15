# Recurrent Neural Network (RNN)

# 1. Overview

A Recurrent Neural Network (RNN) is a neural network designed for:

- sequential data
- temporal dependencies
- ordered information

Unlike feedforward networks, RNNs have:

> memory of previous inputs

---

# 2. Why RNN Exists

Traditional neural networks assume:
- inputs are independent

But many real-world tasks are sequential:

| Task | Sequential Dependency |
|------|------|
| Language | previous words matter |
| Speech | previous sounds matter |
| Time series | previous values matter |
| Music | previous notes matter |

RNN solves this by carrying hidden state across time.

---

# 3. Core Idea

At time step \( t \):

RNN takes:
- current input \( x_t \)
- previous hidden state \( h_{t-1} \)

and produces:
- new hidden state \( h_t \)
- optional output \( y_t \)

---

# 4. Mathematical Formulation

## Hidden State Update

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

Where:
- \( x_t \) = current input
- \( h_{t-1} \) = previous memory
- \( W_x \) = input weights
- \( W_h \) = recurrent weights
- \( b \) = bias

---

## Output

$$
y_t = W_y h_t
$$

---

# 5. Key Intuition

RNN repeatedly applies:
```text
current information + previous memory
```

This creates temporal memory.

---

# 6. Hidden State (Very Important)

Hidden state acts like:
> compressed memory of past sequence

Example:

Sentence:
```text
"The sky is blue"
```

When reading `"blue"`:
- hidden state already contains information about:
  - `"The"`
  - `"sky"`
  - `"is"`

---

# 7. Unrolling Through Time

RNN is conceptually unfolded:

```text
x1 → h1 → h2 → h3 ...
```

Each time step shares:
- SAME weights
- SAME architecture

---

## Important

Weights are reused across time.

This:
- reduces parameters
- enables variable-length sequences

---

# 8. Weight Sharing

Unlike dense networks:
- RNN uses same weights at every time step

Meaning:

$$
W_x, W_h
$$

are reused repeatedly.

---

# 9. Sequence Types

## One-to-One
Traditional neural network.

Example:
- image classification

---

## One-to-Many

Example:
- image captioning

```text
image → sentence
```

---

## Many-to-One

Example:
- sentiment analysis

```text
sentence → positive/negative
```

---

## Many-to-Many

Example:
- machine translation
- speech recognition

---

# 10. Why RNN Was Revolutionary

Before transformers:
- RNN was dominant for NLP and speech

Reason:
- could model temporal dependencies naturally

---

# 11. Vanishing Gradient Problem (CRITICAL)

Main weakness of vanilla RNN.

During backpropagation through time:

Gradients repeatedly multiply.

If values < 1:

$$
0.5^{100} \rightarrow 0
$$

Gradient disappears.

---

## Consequences

Model forgets long-term dependencies.

Example:
```text
"The movie I watched 3 hours ago was ..."
```

RNN struggles remembering early context.

---

# 12. Exploding Gradient Problem

Opposite issue.

If repeated multiplication > 1:

$$
2^{100} \rightarrow \infty
$$

Gradients explode.

---

## Consequences

- unstable training
- NaNs
- divergence

---

# 13. Why Vanishing Happens

Repeated chain rule multiplication:

$$
\frac{\partial h_t}{\partial h_{t-1}}
\cdot
\frac{\partial h_{t-1}}{\partial h_{t-2}}
\cdot ...
$$

Small derivatives collapse exponentially.

---

# 14. Backpropagation Through Time (BPTT)

RNN training uses:
> Backpropagation Through Time

RNN is unfolded across time,
then normal backpropagation is applied.

---

# 15. Truncated BPTT

Full BPTT is expensive.

Practical training often truncates:

Example:
- only backpropagate last 50 steps

---

## Tradeoff

- faster
- less memory
- weaker long-term learning

---

# 16. LSTM (Long Short-Term Memory)

Created to fix vanishing gradients.

Adds:
- memory cell
- gates controlling information flow

---

## Gates

| Gate | Purpose |
|------|------|
| Forget gate | remove information |
| Input gate | add information |
| Output gate | expose information |

---

# 17. GRU (Gated Recurrent Unit)

Simplified version of LSTM.

- fewer parameters
- faster
- often similar performance

---

# 18. Why LSTM Works Better

LSTM creates:
> more stable gradient flow

Allows learning:
- long dependencies
- long context sequences

---

# 19. Bidirectional RNN

Processes sequence:
- forward
- backward

Example:
```text
left → right
right → left
```

Useful because:
- future context helps understanding

---

# 20. Example

Sentence:
```text
"I went to the bank"
```

Future words help determine:
- river bank
- financial bank

---

# 21. RNN in NLP

Applications:
- language modeling
- translation
- sentiment analysis
- speech recognition

---

# 22. RNN in Time Series

Applications:
- stock prediction
- weather forecasting
- sensor prediction

---

# 23. Teacher Forcing

During training:
- true previous token is fed

instead of model prediction.

---

## Why Important

Improves:
- convergence speed
- stability

---

# 24. Exposure Bias

Problem:
- during inference model sees its own predictions
- during training it sees true labels

Mismatch causes error accumulation.

---

# 25. Computational Weakness

RNN is sequential:

$$
h_t \text{ depends on } h_{t-1}
$$

Meaning:
- cannot parallelize efficiently

---

# 26. Why Transformers Replaced RNNs

Transformers:
- parallelize computation
- capture long-range dependencies better
- avoid recurrent bottleneck

---

## Important

RNN still useful for:
- lightweight systems
- streaming tasks
- low-resource environments

---

# 27. Memory Compression Problem

Hidden state is fixed-size vector.

Long sequence:
- too much information compressed
- older information lost

---

# 28. Attention Mechanism

Attention was invented partly because:
- RNN bottlenecked all memory into hidden state

Attention allows:
> direct access to previous states

---

# 29. Sequence Length Difficulty

Longer sequence means:
- harder optimization
- more gradient decay
- more memory usage

---

# 30. Common Activations

Typical RNN activations:
- tanh
- sigmoid

Rarely ReLU in vanilla RNN due to instability.

---

# 31. Common Beginner Misunderstandings

## ❌ RNN remembers everything

False.

Vanilla RNN memory is weak.

---

## ❌ Hidden state stores raw text

False.

It stores compressed learned representations.

---

## ❌ LSTM completely solves long-term memory

Not fully.

Still struggles with extremely long context.

---

## ❌ RNN processes whole sentence simultaneously

False.

It processes step-by-step sequentially.

---

# 32. Time Complexity

RNN complexity:

$$
O(T)
$$

where:
- \( T \) = sequence length

But sequential dependency prevents efficient parallelism.

---

# 33. Comparison

| Model | Parallelizable | Long Memory | Speed |
|------|------|------|------|
| RNN | ❌ | Weak | Slow |
| LSTM | ❌ | Better | Slow |
| Transformer | ✅ | Strong | Fast |

---

# 34. Mental Model

RNN behaves like:

> reading a sentence one word at a time while carrying a small memory notebook

---

# 35. Key Takeaway

RNN introduced:
- temporal memory
- sequence learning
- recurrent computation

But suffers from:
- vanishing gradients
- sequential bottleneck
- weak long-range memory

LSTM and GRU improved RNN,
and transformers eventually replaced them in most large-scale NLP tasks.