# PEFT (Parameter-Efficient Fine-Tuning)

---

# 1. What is PEFT?

PEFT stands for:

> **Parameter-Efficient Fine-Tuning**

It is a set of techniques for fine-tuning Large Language Models (LLMs) while training only a **small subset** of parameters instead of the entire model.

---

# 2. Why PEFT Exists

Modern LLMs are enormous.

Example:

| Model | Parameters |
|---|---|
| Gemma 2B | 2 billion |
| Llama 7B | 7 billion |
| GPT-scale | 100B+ |

Training all parameters requires:
- huge VRAM
- massive compute
- expensive GPUs
- long training times

PEFT solves this by:
> freezing most model weights and training only small additional components.

---

# 3. Core Idea

Traditional fine-tuning:

```text
Update ALL model weights
```

PEFT:

```text
Freeze base model
+
Train small adapter layers
```

This dramatically reduces:
- VRAM usage
- training time
- storage size

---

# 4. Most Popular PEFT Method: LoRA

LoRA =:

> Low-Rank Adaptation

Instead of changing original weights directly:

```text
Original Weight Matrix W
```

LoRA adds:

```text
W + ΔW
```

where:

```text
ΔW = A × B
```

and:
- A and B are very small trainable matrices
- original weights remain frozen

---

# 5. Why LoRA Works

Neural network updates often lie in:
> lower-dimensional subspaces.

Meaning:
- full giant updates are often unnecessary
- small structured updates can approximate learning well

This is the key insight behind LoRA.

---

# 6. Benefits of PEFT

| Benefit | Explanation |
|---|---|
| Lower VRAM usage | Train tiny parameter subset |
| Faster training | Fewer gradients and optimizer states |
| Smaller checkpoints | Adapters are tiny |
| Reusable adapters | Multiple tasks can share same base model |
| Consumer GPU friendly | Enables local fine-tuning |

---

# 7. Typical PEFT Workflow

```text
Load pretrained model
↓
Freeze base weights
↓
Attach adapters (LoRA)
↓
Train only adapters
↓
Save adapters
```

---

# 8. PEFT vs Full Fine-Tuning

| Aspect | Full Fine-Tuning | PEFT |
|---|---|---|
| Trainable params | All | Small subset |
| VRAM usage | Huge | Much smaller |
| Storage | Large | Tiny |
| Speed | Slower | Faster |
| Consumer GPU friendly | Usually no | Often yes |

---

# 9. QLoRA

QLoRA combines:
- PEFT (LoRA)
- quantization (4-bit)

Workflow:

```text
4-bit frozen base model
+
small LoRA adapters
```

This enables:
> training large LLMs on consumer GPUs.

---

# 10. Common PEFT Techniques

| Method | Idea |
|---|---|
| LoRA | Low-rank adapters |
| QLoRA | LoRA + quantization |
| Prefix Tuning | Learn prompt-like vectors |
| Prompt Tuning | Train soft prompts |
| IA3 | Learn scaling vectors |
| AdaLoRA | Adaptive LoRA ranks |

LoRA is currently the most popular.

---

# 11. Hugging Face PEFT Library

Main library:

```python
from peft import ...
```

Common functions:

```python
LoraConfig
get_peft_model
prepare_model_for_kbit_training
```

---

# 12. Basic LoRA Example

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA adapters
lora_config = LoraConfig(
    r=8,                     # Rank of adapter matrices
    lora_alpha=16,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers to adapt
    lora_dropout=0.05,       # Regularization dropout
    bias="none",
    task_type="CAUSAL_LM"
)

# Attach LoRA adapters to model
model = get_peft_model(model, lora_config)

# Print trainable parameter summary
model.print_trainable_parameters()
```

---

# 13. Important LoRA Parameters

## `r`

Adapter rank.

Higher:
- more expressive
- more memory

Lower:
- lighter
- smaller

Typical:
```text
4-64
```

---

## `lora_alpha`

Scaling factor for LoRA updates.

Controls:
> influence strength of adapter updates.

---

## `target_modules`

Which layers receive LoRA adapters.

Common transformer targets:

```text
q_proj
k_proj
v_proj
o_proj
```

---

## `lora_dropout`

Regularization to reduce overfitting.

---

# 14. What Actually Gets Saved?

Usually:
- ONLY LoRA adapters

NOT:
- full base model

This keeps checkpoints tiny.

Example:

| Type | Size |
|---|---|
| Full model | GBs |
| LoRA adapters | MBs |

---

# 15. During Inference

Two possibilities:

## Option A — Dynamic Adapter Loading

```text
Base model
+
LoRA adapters
```

loaded together.

---

## Option B — Merge Adapters

Merge LoRA weights into base model permanently.

Useful for deployment.

---

# 16. Why PEFT Became So Important

Without PEFT:
- local fine-tuning nearly impossible
- GPU requirements enormous

PEFT democratized LLM training.

It is one of the most impactful practical innovations in modern LLM engineering.

---

# 17. Mental Model

Think:

```text
Base model = giant frozen brain
LoRA adapters = tiny task-specific skill patches
```

---

# 18. Common Beginner Mistakes

## Accidentally Full Fine-Tuning

If LoRA setup fails:
- entire model may train
- instant VRAM explosion

Always verify:

```python
model.print_trainable_parameters()
```

---

## Forgetting Quantization

Without 4-bit/8-bit loading:
VRAM may still overflow.

---

## Wrong Target Modules

Different architectures use different layer names.

---

# 19. PEFT + Quantization + Gradient Checkpointing

Modern local LLM training commonly combines:

```text
Quantization
+
PEFT
+
Gradient checkpointing
```

Together they massively reduce memory usage.

---

# 20. Final Takeaway

PEFT allows:
> adapting huge pretrained models efficiently without retraining everything.

This transformed:
- research
- open-source AI
- local LLM experimentation
- consumer GPU fine-tuning