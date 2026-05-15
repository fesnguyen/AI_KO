# Bias–Variance Tradeoff

## 1. Overview

Bias and variance describe **two fundamental sources of error** in machine learning.

- **Bias** → error from wrong assumptions (underfitting)
- **Variance** → error from sensitivity to data (overfitting)

👉 Goal: **balance both to minimize generalization error**

---

## 2. Formal Definition

Assume true function \( f(x) \), model prediction \( \hat{f}(x) \), and noise \( \epsilon \):

$$
y = f(x) + \epsilon
$$

Expected squared error:

$$
\mathbb{E}[(y - \hat{f}(x))^2] =
\underbrace{\text{Bias}^2}_{\text{systematic error}} +
\underbrace{\text{Variance}}_{\text{sensitivity}} +
\underbrace{\text{Noise}}_{\text{irreducible}}
$$

---

## 3. Bias

### Definition
$$
\text{Bias}(x) = \mathbb{E}[\hat{f}(x)] - f(x)
$$

### Intuition
- Model is **too simple**
- Cannot capture true relationship

### Characteristics
- High training error
- High validation error
- Stable across datasets

### Examples
- Linear model on nonlinear data
- Under-parameterized neural network

---

## 4. Variance

### Definition
$$
\text{Variance}(x) = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]
$$

### Intuition
- Model is **too sensitive to training data**
- Learns noise instead of signal

### Characteristics
- Low training error
- High validation error
- Unstable across datasets

### Examples
- Deep decision trees
- Over-parameterized models

---

## 5. Noise (Irreducible Error)

- Comes from randomness in data
- Cannot be eliminated

$$
\epsilon \sim \text{random noise}
$$

👉 Even perfect model cannot remove this

---

## 6. Tradeoff (Core Idea)

- Increasing model complexity:
  - ↓ Bias
  - ↑ Variance

- Decreasing complexity:
  - ↑ Bias
  - ↓ Variance

👉 Optimal model sits at **balance point**

---

## 7. Visualization (Mental Model)

| Model | Bias | Variance |
|------|------|----------|
| Too simple | High | Low |
| Too complex | Low | High |
| Optimal | Balanced | Balanced |

---

## 8. Connection to Overfitting / Underfitting

| Situation | Bias | Variance |
|----------|------|----------|
| Underfitting | High | Low |
| Overfitting | Low | High |

---

## 9. Model Examples

### High Bias Models
- Linear regression (no features)
- Naive Bayes

### High Variance Models
- Decision trees (deep)
- kNN with small k

---

## 10. How Algorithms Handle It

### :contentReference[oaicite:0]{index=0}
- Reduces **variance** via averaging
- Bias stays similar

---

### :contentReference[oaicite:1]{index=1}
- Reduces **bias** via sequential learning
- Can increase variance if overfit

---

## 11. Techniques to Control Bias / Variance

### Reduce Bias
- Increase model complexity
- Add features
- Use boosting

---

### Reduce Variance
- More data
- Regularization (L1/L2)
- Pruning (trees)
- Bagging / Random Forest

---

## 12. Regularization Connection

- L2 → reduces variance (smooth weights)
- L1 → reduces variance + feature selection

👉 Both increase bias slightly to reduce variance

---

## 13. Cross-Validation Role

- Estimates **variance of model performance**
- Detects instability

High std in CV scores → high variance model

---

## 14. Key Quirks (Important)

### ❗ Low bias ≠ good model
- Can still massively overfit

---

### ❗ High variance is data-dependent
- Same model can behave differently on different datasets

---

### ❗ Bias is not always bad
- Slight bias improves generalization

---

### ❗ More data reduces variance, NOT bias
- Common misunderstanding

---

### ❗ Ensembles mainly reduce variance
- Not bias (except boosting)

---

## 15. Practical Diagnosis

### Case 1
- Train error high  
- Val error high  
→ High bias  

---

### Case 2
- Train error low  
- Val error high  
→ High variance  

---

### Case 3
- Both low  
→ Good model  

---

## 16. Mental Model (Fast Recall)

- **Bias = wrong assumptions**
- **Variance = too sensitive to data**

---

## 17. Key Takeaway

$$
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

👉 You are always trading:
- simplicity vs flexibility  
- stability vs adaptability  

---

## 18. One-line Intuition

- Bias = “model is too dumb”  
- Variance = “model is too reactive”