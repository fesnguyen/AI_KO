# OOM (Out Of Memory) Debugging Journey Summary

---

# 1. Initial Problem

Training Gemma 2B caused:

```text
CUDA Out Of Memory (OOM)
```

Meaning:

```text
GPU VRAM exhausted during training
```

---

# 2. Important Realization

Successfully loading a model:
```text
≠
successfully training a model
```

Training requires much more memory because of:
- activations
- gradients
- optimizer states
- temporary CUDA buffers

---

# 3. Main VRAM Consumers

| Component | VRAM Impact | Notes |
|---|---|---|
| Model weights | large | reduced by quantization |
| Activations | enormous | biggest training memory consumer |
| Gradients | large | created during backward pass |
| Optimizer states | large | AdamW especially expensive |
| Padding waste | moderate-large | unnecessary tokens still consume compute |

---

# 4. Initial Situation

Gemma 2B loaded normally consumed:

```text
~75% VRAM
```

on RTX 4060 Ti 8GB.

Meaning:
> almost no memory remained for training.

---

# 5. Quantization

## Purpose

Reduce:
> model weight memory.

---

# Technique Used

```python
BitsAndBytesConfig(
    load_in_4bit=True
)
```

---

# Result

VRAM usage dropped dramatically:

```text
~75%
→
~3.5GB
```

---

# Common Quantization Levels

| Precision | Memory Usage | Common Use |
|---|---|---|
| fp32 | very high | research/training |
| fp16 | high | common GPU inference/training |
| 8bit | medium | lightweight inference |
| 4bit | low | QLoRA/local training |

---

# Drawbacks

| Drawback | Explanation |
|---|---|
| slower kernels sometimes | quantized ops may be less optimized |
| slight accuracy degradation | usually small |
| more complex setup | bitsandbytes/configuration |
| some models unsupported | architecture compatibility issues |

---

# Important Insight

Quantization reduces:
```text
weight memory
```

NOT:
```text
activation memory
```

---

# 6. PEFT / LoRA / QLoRA

---

# PEFT

General category:

> Parameter-Efficient Fine-Tuning

---

# LoRA

Most popular PEFT method.

Idea:

```text
Freeze base model
+
Train tiny adapter matrices
```

---

# QLoRA

Combination of:
- LoRA
- 4-bit quantization

---

# Relationship

```text
PEFT
└── LoRA
    └── QLoRA
```

---

# Common LoRA Parameters

| Parameter | Common Values | Purpose |
|---|---|---|
| r | 4, 8, 16, 32 | adapter rank |
| lora_alpha | 16, 32, 64 | scaling strength |
| dropout | 0.0-0.1 | regularization |

---

# Current Suggested Config

```python
r=8
lora_alpha=16
lora_dropout=0.05
```

---

# Drawbacks

| Drawback | Explanation |
|---|---|
| less expressive than full FT | limited adaptation capacity |
| architecture-dependent modules | target layer naming differs |
| some tasks may underperform | compared to full FT |
| adapters still consume memory | though much smaller |

---

# Important Insight

LoRA reduces:
```text
trainable parameter memory
```

NOT:
```text
activation memory
```

---

# 7. Gradient Checkpointing

---

# Purpose

Reduce:
> activation memory.

---

# Technique

```python
gradient_checkpointing=True
```

or:

```python
model.gradient_checkpointing_enable()
```

---

# How It Works

Instead of storing activations:
> recompute them during backward pass.

---

# Benefit

Much lower VRAM usage.

Especially important for:
- transformers
- long sequences

---

# Drawbacks

| Drawback | Explanation |
|---|---|
| slower training | recomputation overhead |
| more GPU computation | repeated forward calculations |
| debugging harder | more complex execution |

---

# Common Usage

| Setting | Usage |
|---|---|
| True | memory-constrained training |
| False | faster training if VRAM sufficient |

---

# Important Insight

Checkpointing reduces:
```text
activation memory
```

NOT:
```text
model weight memory
```

---

# 8. Sequence Length

Controlled by:

```python
max_length=...
```

---

# Purpose

Controls:
> maximum token sequence length.

---

# Why Important

Transformer attention complexity scales roughly:

Longer sequences dramatically increase:
- attention tensors
- activations
- compute cost

---

# Common Values

| Length | Typical Usage |
|---|---|
| 128 | lightweight local training |
| 256 | balanced |
| 512 | common |
| 1024+ | expensive |
| 8K+ | large-context models |

---

# Drawbacks of Long Sequences

| Drawback | Explanation |
|---|---|
| huge VRAM increase | quadratic attention scaling |
| slower training | more attention computation |
| fewer samples/sec | throughput reduction |

---

# 9. Batch Size

Controlled by:

```python
per_device_train_batch_size
```

---

# Purpose

Controls:
> simultaneous training examples.

---

# Common Values

| Batch Size | Typical Usage |
|---|---|
| 1 | very low VRAM |
| 2-4 | consumer GPUs |
| 8-32 | stronger GPUs |
| 64+ | distributed training |

---

# Tradeoffs

| Larger Batch | Smaller Batch |
|---|---|
| faster throughput | lower VRAM |
| more stable gradients | noisier gradients |
| higher VRAM | slower training |

---

# 10. Gradient Accumulation

Controlled by:

```python
gradient_accumulation_steps
```

---

# Purpose

Simulate larger effective batches.

effective_batch_size =
per_device_train_batch_size
×
gradient_accumulation_steps
×
number_of_devices

---

# Common Values

| Value | Typical Usage |
|---|---|
| 1 | direct updates |
| 2-8 | consumer GPUs |
| 16+ | extreme memory constraints |

---

# Drawbacks

| Drawback | Explanation |
|---|---|
| slower optimizer updates | more mini-steps needed |
| slower wall-clock training | delayed updates |
| GPU utilization inefficiency | small real batches |

---

# 11. Static Padding

Original approach:

```python
padding="max_length"
```

---

# Problem

Every sample padded to:
```text
max_length
```

even tiny examples.

---

# Drawbacks

| Drawback | Explanation |
|---|---|
| wasted VRAM | padding tokens consume tensors |
| slower attention | useless token computation |
| lower throughput | unnecessary operations |

---

# 12. Dynamic Padding / DataCollator

---

# Purpose

Pad only to:
> longest sample in current batch.

---

# Example

Instead of:
```text
all samples → 128 tokens
```

Use:
```text
current batch longest → maybe 42 tokens
```

---

# Benefits

| Benefit | Explanation |
|---|---|
| lower VRAM | fewer useless tokens |
| faster training | less attention compute |
| better GPU efficiency | smaller tensors |

---

# Drawbacks

| Drawback | Explanation |
|---|---|
| variable tensor shapes | slightly more complex |
| batching less predictable | dynamic memory patterns |

---

# 13. Device Mapping

---

# Purpose

Automatically place model across:
- GPU
- CPU

using:

```python
device_map="auto"
```

---

# Benefit

Avoid manual:
```python
model.to("cuda")
```

management.

---

# Drawbacks

| Drawback | Explanation |
|---|---|
| CPU offloading slower | PCIe transfer overhead |
| debugging harder | distributed tensor placement |
| unpredictable placement | automatic heuristics |

---

# 14. Current Stable Setup

Current successful strategy:

```text
4-bit quantization
+
QLoRA
+
small batch size
+
gradient accumulation
+
optional checkpointing
```

---

# 15. Current Situation

Training now:
- runs successfully
- uses ~70% VRAM
- slower than aggressive configs

Meaning:
> memory-safe but conservative.

---

# 16. Optimization Strategy Going Forward

Now possible to gradually optimize:

| Change | Effect |
|---|---|
| disable checkpointing | faster, more VRAM |
| increase batch size | faster throughput |
| reduce accumulation | faster updates |
| optimize padding | better efficiency |

---

# 17. Most Important Insight

Different techniques solve DIFFERENT bottlenecks:

| Technique | Main Reduction |
|---|---|
| Quantization | weight memory |
| LoRA | trainable parameter memory |
| Gradient checkpointing | activation memory |
| Smaller batches | simultaneous activations |
| Dynamic padding | padding waste |

Together they enable:
> local LLM fine-tuning on consumer GPUs.