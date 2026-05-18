# Handling Outliers in Machine Learning

Outliers are data points that are significantly different from the majority of the data.

Examples:
- Fraudulent transactions
- Sensor failures
- Human input mistakes
- Rare but valid events

Outliers can:
- Distort statistics
- Reduce model performance
- Cause unstable training
- Affect feature scaling

---

# 1. What is an Outlier?

Example dataset:

```python
values = [10, 12, 11, 13, 14, 15, 200]
```

Here:
```python
200
```

is likely an outlier.

---

# 2. Why Outliers Matter

Outliers heavily affect:
- Mean
- Standard deviation
- Distance-based models

Especially harmful for:
- Linear Regression
- Logistic Regression
- KNN
- SVM
- Neural Networks

Less harmful for:
- Random Forest
- XGBoost
- LightGBM

---

# 3. Common Causes of Outliers

## 3.1 Human Errors

Examples:
- Wrong decimal point
- Typing mistakes

```python
age = 999
```

---

## 3.2 Measurement Errors

Examples:
- Broken sensors
- Hardware failures

---

## 3.3 Natural Rare Events

Examples:
- Luxury house prices
- Fraud detection
- Rare diseases

These may actually contain useful information.

---

# 4. Detecting Outliers

---

# 4.1 Visualization Methods

## Boxplot

```python
import matplotlib.pyplot as plt

plt.boxplot(df["feature"])
plt.show()
```

---

## Histogram

```python
plt.hist(df["feature"], bins=30)
plt.show()
```

---

## Scatter Plot

```python
plt.scatter(df["x"], df["y"])
plt.show()
```

Useful for:
- Multivariate outliers

---

# 4.2 Statistical Methods

---

## Z-Score Method

Measures how far a point is from the mean.

Formula:

```python
z = (x - mean) / std
```

Common threshold:

```python
|z| > 3
```

---

## Example

```python
from scipy.stats import zscore

df["zscore"] = zscore(df["feature"])

outliers = df[df["zscore"].abs() > 3]
```

---

## Advantages
- Simple
- Fast

## Disadvantages
- Sensitive to skewed distributions

---

# 4.3 IQR Method (Most Common)

Interquartile Range.

Very popular because it is robust to skewed data.

---

## Formula

```python
IQR = Q3 - Q1
```

Outlier boundaries:

```python
Lower = Q1 - 1.5 * IQR
Upper = Q3 + 1.5 * IQR
```

---

## Example

```python
Q1 = df["feature"].quantile(0.25)
Q3 = df["feature"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["feature"] < lower) |
    (df["feature"] > upper)
]
```

---

## Advantages
- Robust
- Works well for skewed data

## Disadvantages
- Univariate only

---

# 4.4 Percentile Method

Cap extreme values using percentiles.

Example:
- 1st percentile
- 99th percentile

---

## Example

```python
lower = df["feature"].quantile(0.01)
upper = df["feature"].quantile(0.99)

df["feature"] = df["feature"].clip(lower, upper)
```

This technique is called:

# Winsorization

---

# 4.5 Isolation Forest

Machine learning method for anomaly detection.

Works by isolating unusual points.

---

## Example

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    contamination=0.01,
    random_state=42
)

preds = iso.fit_predict(df[["feature"]])

outliers = df[preds == -1]
```

---

## Advantages
- Handles multivariate outliers
- Works well for large datasets

## Disadvantages
- Harder to interpret

---

# 4.6 Local Outlier Factor (LOF)

Detects points with lower local density.

Useful for:
- Complex local anomalies

---

## Example

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.01
)

preds = lof.fit_predict(df[["feature"]])

outliers = df[preds == -1]
```

---

# 5. Handling Outliers

Detecting is only half the problem.

Now we decide:
- Remove?
- Transform?
- Cap?
- Keep?

---

# 5.1 Remove Outliers

Good when:
- Clear errors
- Very small number

---

## Example

```python
df = df[
    (df["feature"] >= lower) &
    (df["feature"] <= upper)
]
```

---

## Advantages
- Cleaner dataset

## Disadvantages
- Information loss

---

# 5.2 Winsorization (Capping)

Replace extreme values with boundaries.

---

## Example

```python
lower = df["feature"].quantile(0.01)
upper = df["feature"].quantile(0.99)

df["feature"] = df["feature"].clip(lower, upper)
```

---

## Advantages
- Keeps all rows
- Reduces extreme impact

## Disadvantages
- Artificial modification

---

# 5.3 Log Transformation

Useful for:
- Right-skewed distributions

---

## Example

```python
import numpy as np

df["log_feature"] = np.log1p(df["feature"])
```

---

## Advantages
- Compresses extreme values
- Stabilizes variance

## Disadvantages
- Only for positive values

---

# 5.4 Robust Scaling

Uses median and IQR instead of mean/std.

Very useful with outliers.

---

## Example

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)
```

---

# 5.5 Use Robust Models

Some models naturally resist outliers.

Examples:
- Random Forest
- XGBoost
- LightGBM

---

# 6. When NOT to Remove Outliers

Very important.

Sometimes outliers ARE the signal.

Examples:
- Fraud detection
- Rare diseases
- Cybersecurity attacks

Removing them destroys useful information.

---

# 7. Practical Workflow

---

## Step 1 — Understand Business Context

Ask:
- Is this error or real behavior?

---

## Step 2 — Visualize

Use:
- Boxplot
- Histogram
- Scatterplot

---

## Step 3 — Statistical Detection

Try:
- IQR
- Z-score

---

## Step 4 — Choose Strategy

Possible strategies:
- Remove
- Cap
- Transform
- Keep

---

## Step 5 — Compare Model Performance

Always compare:
- Before handling
- After handling

because some models do not care much about outliers.

---

# 8. Common Interview Questions

---

## Why is mean sensitive to outliers?

Because extreme values strongly shift the average.

---

## Why is median more robust?

Because it depends on position rather than magnitude.

---

## Difference between StandardScaler and RobustScaler?

| StandardScaler | RobustScaler |
|---|---|
| Uses mean/std | Uses median/IQR |
| Sensitive to outliers | Robust to outliers |

---

## Why is IQR popular?

Because it works well even with skewed distributions.

---

## Why does linear regression suffer from outliers?

Because regression minimizes squared error.

Large errors become heavily penalized.

---

# 9. Quick Recommendations

| Situation | Recommended Method |
|---|---|
| Normal distribution | Z-score |
| Skewed distribution | IQR |
| Heavy skew | Log transform |
| Multivariate anomalies | Isolation Forest |
| Keep all rows | Winsorization |
| Scaling with outliers | RobustScaler |

---

# 10. Important Warning

Never blindly remove outliers.

Always ask:
```text
Is this noise or valuable information?
```

This is one of the most important decisions in real-world ML.

---

# 11. Libraries

## Install

```bash
pip install scipy scikit-learn
```

---

# 12. Final Notes

Outlier handling is highly problem-dependent.

There is no universal rule.

The best approach depends on:
- Business domain
- Distribution shape
- Model type
- Data size

Always:
- Visualize first
- Understand context
- Compare model performance
- Avoid blindly deleting data

---

# Transformer Refresher

Original Transformer paper:

https://arxiv.org/abs/1706.03762