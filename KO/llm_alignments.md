# llm_alignments.md

# LLM Alignment Variants

A practical overview of common alignment styles, training strategies, and deployment variants used in modern Large Language Models (LLMs).

---

# 1. Base Model

## What it is

A raw pretrained language model trained only to predict the next token.

Usually trained on:

* Web text
* Books
* Code
* Forums
* Academic content

Objective:

```text
Predict the next token correctly.
```

No instruction tuning.
No safety tuning.
No conversation behavior.

---

## Characteristics

### Pros

* Most "pure" intelligence/capability
* Flexible
* Often strongest for research/fine-tuning
* Better raw completion ability

### Cons

* Doesn't naturally follow instructions well
* Can ramble
* Unsafe outputs possible
* Weak conversational formatting

---

## Example Models

* Gemma Base
* Llama Base
* Mistral Base
* GPT foundation models (internal)
* DeepSeek Base

---

## Example Behavior

Prompt:

```text
How do I cook rice?
```

Base model may continue like:

```text
How do I cook rice? Rice is a staple food...
```

Instead of directly answering.

---

# 2. Instruct Model

## What it is

A base model additionally trained to follow human instructions.

Usually created with:

* Supervised Fine-Tuning (SFT)
* Instruction datasets
* Human-written examples

---

## Goal

Transform:

```text
Predict text
```

Into:

```text
Follow instructions helpfully
```

---

## Characteristics

### Pros

* Better at answering directly
* Cleaner formatting
* Easier to use
* Better chat behavior

### Cons

* Slight capability loss sometimes
* Can become overly cautious
* Less "creative raw completion"

---

## Example Models

* Gemma Instruct
* Llama Instruct
* Mistral Instruct
* GPT-3.5 Turbo
* GPT-4 Chat variants

---

## Example Behavior

Prompt:

```text
How do I cook rice?
```

Response:

```text
1. Wash the rice
2. Add water
3. Cook for 15 minutes
```

Direct and instruction-following.

---

# 3. Chat Model

## What it is

An instruct model additionally optimized for:

* Multi-turn conversation
* Personality consistency
* Dialogue memory formatting
* Assistant/user role handling

---

## Usually Includes

* Chat templates
* System prompts
* RLHF or preference tuning
* Safety alignment

---

## Characteristics

### Pros

* Natural conversation
* Good assistant behavior
* Better context handling

### Cons

* Sometimes verbose
* Can over-refuse
* Can hallucinate confidently

---

## Example Models

* ChatGPT
* Claude
* Gemini
* Grok
* Qwen Chat

---

# 4. RLHF-Aligned Model

## RLHF = Reinforcement Learning from Human Feedback

## What it is

Model trained using human preference comparisons.

Humans rank outputs:

```text
A is better than B
```

The model learns:

```text
Generate outputs humans prefer.
```

---

## Pipeline

### Step 1 — Pretraining

Train base model.

### Step 2 — SFT

Train on instruction examples.

### Step 3 — Reward Model

Humans rank outputs.

### Step 4 — RL Optimization

Use PPO/DPO/etc to optimize behavior.

---

## Goals

Improve:

* Helpfulness
* Harmlessness
* Honesty
* Tone
* Safety

---

## Characteristics

### Pros

* More usable
* Better social behavior
* Better refusals
* More aligned with product goals

### Cons

* Can become sycophantic
* Overly safe
* Sometimes less truthful
* May optimize for "sounding good"

---

## Common Methods

### PPO

Proximal Policy Optimization.

Older OpenAI-style RLHF.

### DPO

Direct Preference Optimization.

Simpler modern alternative.

### ORPO / KTO / IPO

Newer preference optimization variants.

---

# 5. Safety-Aligned Model

## What it is

A model specifically tuned to avoid:

* Harmful outputs
* Illegal instructions
* Toxicity
* Privacy violations
* Dangerous content

---

## Methods

* Safety SFT
* RLHF
* Constitutional AI
* Rule-based filters
* Output moderation

---

## Characteristics

### Pros

* Safer deployment
* Better public-facing product

### Cons

* Over-refusal
* False positives
* Reduced openness

---

## Example

User:

```text
How to make a bomb?
```

Safety-aligned model refuses.

---

# 6. Constitutional AI

## Popularized by

Anthropic

---

## What it is

Instead of relying only on humans ranking outputs,
the model follows a written "constitution."

Example principles:

* Avoid harm
* Be honest
* Respect autonomy
* Avoid manipulation

The AI critiques and revises its own outputs.

---

## Characteristics

### Pros

* More scalable than full human labeling
* Better consistency
* Strong safety behavior

### Cons

* Constitution quality matters
* Still subjective
* May encode company values

---

# 7. Tool-Use Aligned Model

## What it is

A model trained to:

* Call tools
* Use APIs
* Search the web
* Execute functions
* Use retrieval systems

---

## Training Includes

Special formatting like:

```json
{
  "tool": "web_search",
  "arguments": {
    "query": "weather today"
  }
}
```

---

## Characteristics

### Pros

* Real-time information
* Better reasoning workflows
* Reduced hallucination
* Agent capabilities

### Cons

* More complex systems
* Tool misuse possible
* Requires orchestration layer

---

## Example Models

* GPT-4 Tool Calling
* Claude Tool Use
* Gemini Function Calling
* Qwen Agent variants

---

# 8. Agentic-Aligned Model

## What it is

A model trained for:

* Planning
* Multi-step reasoning
* Self-correction
* Tool chaining
* Task execution loops

---

## Common Abilities

* Reflection
* ReAct
* Planning
* Memory usage
* Retry logic

---

## Example

Instead of:

```text
One response only
```

The model may:

```text
Think
→ search
→ analyze
→ retry
→ summarize
```

---

# 9. Reasoning Model

## What it is

A model optimized for:

* Chain-of-thought
* Step-by-step reasoning
* Math
* Coding
* Logic

---

## Techniques

* Reinforcement learning
* Process supervision
* Self-consistency
* Verifier models
* Long reasoning traces

---

## Characteristics

### Pros

* Strong math/coding
* Better planning
* Better logic

### Cons

* Slower
* More expensive inference
* Can overthink simple tasks

---

## Example Models

* OpenAI o-series
* DeepSeek-R1
* QwQ
* Gemini Thinking variants

---

# 10. Domain-Aligned Models

## What it is

Models specialized for a specific field.

---

## Examples

### Code Models

* CodeLlama
* DeepSeek-Coder

### Medical Models

* Med-PaLM

### Legal Models

* Harvey-based systems

### Finance Models

* BloombergGPT

---

## Characteristics

### Pros

* Better domain accuracy
* Specialized vocabulary

### Cons

* Narrower usefulness
* Domain bias

---

# 11. Personality-Aligned Models

## What it is

Models tuned for:

* Tone
* Style
* Character
* Emotional behavior

---

## Examples

* Friendly assistant
* Tutor
* Coding mentor
* RPG character
* Corporate support bot

---

## Usually Implemented With

* System prompts
* SFT
* RLHF
* Persona datasets

---

# 12. Open vs Closed Alignment

## Open Alignment

Public:

* Training methods
* Datasets
* Weights

Examples:

* Llama
* Mistral
* Gemma

---

## Closed Alignment

Internal/private:

* RLHF pipelines
* Safety systems
* Hidden prompts
* Proprietary reward models

Examples:

* ChatGPT
* Claude
* Gemini

---

# 13. Alignment Tradeoffs

## Common Tradeoff Table

| Property              | More Alignment    | Less Alignment |
| --------------------- | ----------------- | -------------- |
| Safety                | Higher            | Lower          |
| Raw freedom           | Lower             | Higher         |
| Instruction following | Higher            | Lower          |
| Creativity            | Sometimes lower   | Often higher   |
| Hallucination control | Better            | Worse          |
| Refusal rate          | Higher            | Lower          |
| Raw capability        | Sometimes reduced | Often stronger |

---

# 14. Real Production Stack

Modern assistants are usually:

```text
Base Model
→ Instruction Tuning
→ RLHF/DPO
→ Safety Alignment
→ Tool Training
→ System Prompt
→ External Moderation
→ Retrieval/Tools
```

Not just "one model."

---

# 15. Important Reality

Alignment is NOT:

```text
Making AI good
```

Alignment is:

```text
Making model behavior match desired objectives.
```

Those objectives depend on:

* Companies
* Researchers
* Governments
* Products
* Users

Different organizations optimize for different behavior.

---

# 16. Common Misconception

## Myth

```text
Instruct model = smarter
```

## Reality

Usually:

```text
Base model = raw capability
Instruct model = usability
```

The instruct version often feels smarter because:

* Better formatting
* Better obedience
* Better conversational behavior

But the underlying intelligence mostly comes from pretraining.

---

# 17. Practical Advice For Local LLM Users

## If You Want...

### Raw experimentation

Use:

```text
Base model
```

### Assistant/chatbot

Use:

```text
Instruct/chat model
```

### Agents/tools

Use:

```text
Tool-calling instruct model
```

### Coding

Use:

```text
Code-specialized instruct model
```

### Research

Use:

```text
Reasoning models
```

---

# 18. Example Family Variants

## Example: Gemma

### Gemma Base

Raw pretrained model.

### Gemma Instruct

Instruction tuned for assistant behavior.

---

## Example: Llama

### Llama Base

Raw completion model.

### Llama Instruct

Chat/instruction tuned version.

---

## Example: Qwen

### Qwen Base

General pretrained model.

### Qwen Chat

Conversational alignment.

### Qwen Agent

Tool/agent optimized.

---

# 19. Alignment Layers vs Capability Layers

A useful mental model:

```text
Pretraining = intelligence
Alignment = behavior shaping
```

Pretraining teaches:

* Knowledge
* Language
* Reasoning patterns

Alignment teaches:

* What to say
* What not to say
* How to respond
* Formatting/style
* Tool behavior

---

# 20. Final Summary

| Variant        | Main Goal                      |
| -------------- | ------------------------------ |
| Base           | Raw next-token prediction      |
| Instruct       | Follow instructions            |
| Chat           | Hold conversations             |
| RLHF           | Human-preferred behavior       |
| Safety         | Avoid harmful outputs          |
| Constitutional | Rule/principle following       |
| Tool-use       | API and tool calling           |
| Agentic        | Multi-step autonomous behavior |
| Reasoning      | Strong logical reasoning       |
| Domain         | Specialized expertise          |
| Personality    | Specific tone/style            |

---
