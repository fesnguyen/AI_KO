# Imbalanced Dataset Techniques

Handling imbalanced datasets is one of the most important skills in machine learning because many real-world problems naturally contain far fewer positive samples than negative ones.

Examples:
- Fraud detection
- Disease prediction
- Defect detection
- Rare event prediction

---

# 1. Why Accuracy Fails

Suppose:

- 990 negative samples
- 10 positive samples

A model predicting ALL negatives:

```python
accuracy = 990 / 1000
# 99%
```

But the model is completely useless.

---

# 2. Better Evaluation Metrics

## Precision

How many predicted positives are actually positive?

```python
Precision = TP / (TP + FP)
```

Useful when:
- False positives are expensive

Example:
- Spam detection

---

## Recall

How many actual positives are detected?

```python
Recall = TP / (TP + FN)
```

Useful when:
- Missing positives is dangerous

Example:
- Cancer detection

---

## F1 Score

Balance between Precision and Recall.

```python
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

---

## ROC-AUC

Measures ranking quality across thresholds.

Good for:
- General binary classification

---

## PR-AUC

Precision-Recall Area Under Curve.

Better than ROC-AUC for:
- Heavy imbalance

---

# 3. Data-Level Techniques

These methods directly modify the dataset.

---

# 3.1 Random Oversampling

Duplicate minority samples.

## Advantages
- Very simple
- Keeps all information

## Disadvantages
- Overfitting risk

## Example

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(X, y)
```

---

# 3.2 SMOTE

Synthetic Minority Over-sampling Technique.

Creates NEW synthetic minority samples.

Instead of copying:
```text
A -> A
```

It generates:
```text
A -> A'
```

using nearest neighbors.

## Advantages
- Reduces overfitting
- More generalized

## Disadvantages
- Can create noisy synthetic samples

## Example

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

# 3.3 ADASYN

Adaptive Synthetic Sampling.

Similar to SMOTE but focuses more on difficult minority samples.

## Advantages
- Focuses on hard cases

## Disadvantages
- Can amplify noise

## Example

```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)

X_resampled, y_resampled = adasyn.fit_resample(X, y)
```

---

# 3.4 Random Undersampling

Remove majority samples.

## Advantages
- Faster training
- Smaller dataset

## Disadvantages
- Information loss

## Example

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)

X_resampled, y_resampled = rus.fit_resample(X, y)
```

---

# 3.5 NearMiss

Undersampling technique using nearest neighbors.

Keeps majority samples closest to minority samples.

## Advantages
- Keeps informative majority samples

## Disadvantages
- Can remove useful global patterns

## Example

```python
from imblearn.under_sampling import NearMiss

nearmiss = NearMiss(version=1)

X_resampled, y_resampled = nearmiss.fit_resample(X, y)
```

---

# 3.6 Tomek Links

Removes overlapping noisy samples between classes.

Useful for cleaning class boundaries.

## Advantages
- Cleaner decision boundaries

## Disadvantages
- Mild balancing effect only

## Example

```python
from imblearn.under_sampling import TomekLinks

tl = TomekLinks()

X_resampled, y_resampled = tl.fit_resample(X, y)
```

---

# 3.7 Edited Nearest Neighbors (ENN)

Removes samples whose class differs from neighbors.

Useful for noise reduction.

## Advantages
- Cleans noisy data

## Disadvantages
- Can remove too many samples

## Example

```python
from imblearn.under_sampling import EditedNearestNeighbours

enn = EditedNearestNeighbours()

X_resampled, y_resampled = enn.fit_resample(X, y)
```

---

# 3.8 Hybrid Methods

Combines oversampling + cleaning.

Popular combinations:
- SMOTE + TomekLinks
- SMOTE + ENN

---

## SMOTE + Tomek Links

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)

X_resampled, y_resampled = smt.fit_resample(X, y)
```

---

## SMOTE + ENN

```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)

X_resampled, y_resampled = smote_enn.fit_resample(X, y)
```

---

# 4. Algorithm-Level Techniques

Instead of modifying data, modify model behavior.

---

# 4.1 Class Weights

Increase penalty for minority errors.

---

## Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight='balanced'
)
```

---

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    class_weight='balanced'
)
```

---

## XGBoost

```python
scale_pos_weight = negative_samples / positive_samples
```

Example:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    scale_pos_weight=10
)
```

---

## LightGBM

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    is_unbalance=True
)
```

---

# 5. Threshold Tuning

Default threshold:

```python
probability > 0.5
```

But for imbalance:

```python
probability > 0.3
```

may improve Recall.

---

## Example

```python
y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (y_prob > 0.3).astype(int)
```

---

# 6. Ensemble Methods

Specialized models for imbalance.

## Examples

- Balanced Random Forest
- EasyEnsemble
- RUSBoost

---

# 7. Anomaly Detection Approach

Useful when positive samples are EXTREMELY rare.

## Techniques

- Isolation Forest
- One-Class SVM
- Autoencoder anomaly detection

---

# 8. Typical Practical Workflow

## Step 1 — Stratified Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42
)
```

---

## Step 2 — Baseline Model

Train without balancing first.

---

## Step 3 — Evaluate Properly

Use:
- Confusion Matrix
- Recall
- F1
- PR-AUC

---

## Step 4 — Apply Class Weights

Usually first thing to try.

---

## Step 5 — Try SMOTE

Most common balancing method.

---

## Step 6 — Threshold Tuning

Optimize for business goal.

---

# 9. Common Interview Questions

## Why accuracy is bad for imbalanced datasets?

Because majority class dominates prediction.

---

## Why SMOTE is better than random oversampling?

Because SMOTE creates synthetic samples instead of duplicates.

---

## Why PR-AUC is important?

Because ROC-AUC can look artificially good under heavy imbalance.

---

## When should we use anomaly detection?

When minority samples are extremely rare.

---

# 10. Practical Recommendations

| Situation | Recommendation |
|---|---|
| Mild imbalance | Class weights |
| Medium imbalance | SMOTE |
| Extreme imbalance | Anomaly detection |
| Small dataset | Oversampling |
| Huge dataset | Undersampling |
| Tree boosting | XGBoost scale_pos_weight |

---

# 11. Important Warnings

## Never apply resampling BEFORE train-test split

WRONG:

```python
SMOTE -> train_test_split
```

CORRECT:

```python
train_test_split -> SMOTE only on training set
```

Otherwise:
- Data leakage
- Unrealistic performance

---

# 12. Libraries

## Install

```bash
pip install imbalanced-learn
```

---

# 13. Final Notes

The best imbalance technique depends on:
- Dataset size
- Imbalance severity
- Noise level
- Business objective

There is no universal best method.

Always compare:
- Baseline
- Class weights
- SMOTE
- Threshold tuning

using proper evaluation metrics.

---

# Transformer Refresher

Original Transformer paper:

https://arxiv.org/abs/1706.03762