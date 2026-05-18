# Multicollinearity in Machine Learning

Multicollinearity happens when features are highly correlated with each other.

This means:
- One feature can be predicted using another feature
- Features contain overlapping information

---

# 1. Why Multicollinearity Matters

Multicollinearity can cause:
- Unstable coefficients
- Poor interpretability
- Inflated variance
- Unreliable statistical inference

Especially problematic for:
- Linear Regression
- Logistic Regression

Less problematic for:
- Random Forest
- XGBoost
- LightGBM

---

# 2. Simple Example

Suppose we have:

```python
house_size_m2
house_size_ft2
```

These are almost identical information.

Another example:

```python
salary
annual_income
```

Very high correlation.

---

# 3. What Happens Mathematically

Linear regression tries to estimate:

```python
y = b0 + b1x1 + b2x2
```

If:
```python
x1 ≈ x2
```

the model struggles to determine:
- Which feature deserves credit
- Which coefficient should increase/decrease

This causes unstable coefficients.

---

# 4. Symptoms of Multicollinearity

---

## Large Coefficient Changes

Small data changes produce huge coefficient changes.

---

## Unexpected Signs

Example:

```python
education_years -> negative coefficient
```

even though logically it should be positive.

---

## High R² but Low Statistical Significance

Model predicts well overall,
but individual features appear insignificant.

---

## Inflated Standard Errors

Confidence intervals become unstable.

---

# 5. Detecting Multicollinearity

---

# 5.1 Correlation Matrix

Most common method.

---

## Example

```python
corr = df.corr(numeric_only=True)

print(corr)
```

---

## Heatmap Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

---

## Rule of Thumb

Strong correlation:

```python
|correlation| > 0.8
```

may indicate multicollinearity.

---

# 5.2 Variance Inflation Factor (VIF)

Most important statistical method.

Measures how much variance is inflated due to correlation.

---

## Formula Intuition

```python
VIF = 1 / (1 - R²)
```

Where:
- R² comes from predicting one feature using other features

---

# VIF Interpretation

| VIF | Meaning |
|---|---|
| 1 | No multicollinearity |
| 1 - 5 | Moderate |
| > 5 | High |
| > 10 | Severe |

---

## Example

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[["x1", "x2", "x3"]]

vif = pd.DataFrame()

vif["feature"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(
        X.values,
        i
    )
    for i in range(X.shape[1])
]

print(vif)
```

---

# 5.3 Pairplot

Useful for visual inspection.

---

## Example

```python
sns.pairplot(df)
plt.show()
```

---

# 6. Why Tree Models Care Less

Tree models split features independently.

Examples:
- Random Forest
- XGBoost
- LightGBM

So correlated features usually:
- Do not destabilize training much

However:
- Feature importance can become misleading

---

# 7. Problems Caused by Multicollinearity

---

## Linear Models Become Unstable

Coefficients fluctuate heavily.

---

## Feature Importance Becomes Misleading

Correlated features "share importance."

---

## Harder Interpretation

Cannot clearly explain:
```text
Which feature truly matters?
```

---

## Increased Variance

Predictions become less reliable.

---

# 8. Handling Multicollinearity

---

# 8.1 Remove One of the Correlated Features

Most common solution.

Example:

Keep:
```python
house_size_m2
```

Remove:
```python
house_size_ft2
```

---

## Advantages
- Simple
- Effective

## Disadvantages
- Potential information loss

---

# 8.2 Domain Knowledge

Choose the more meaningful feature.

Example:

Prefer:
```python
annual_income
```

instead of:
```python
monthly_income
```

depending on business use.

---

# 8.3 Principal Component Analysis (PCA)

Transforms correlated features into uncorrelated components.

---

## Example

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

---

## Advantages
- Removes multicollinearity
- Reduces dimensions

## Disadvantages
- Loses interpretability

---

# 8.4 Regularization

Very important.

---

# Ridge Regression (L2)

Handles multicollinearity very well.

Shrinks coefficients smoothly.

---

## Example

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)
```

---

# Lasso Regression (L1)

Can shrink some coefficients to zero.

Performs feature selection.

---

## Example

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)

model.fit(X_train, y_train)
```

---

# ElasticNet

Combination of:
- L1
- L2

---

## Example

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

model.fit(X_train, y_train)
```

---

# 8.5 Feature Engineering

Combine correlated features.

Example:

```python
BMI = weight / height²
```

instead of:
- weight
- height

separately.

---

# 9. Dummy Variable Trap

Very common multicollinearity issue.

---

# Example

One-hot encoding:

```python
Gender_Male
Gender_Female
```

One can be perfectly predicted from the other.

---

# Solution

Drop one category.

---

## Example

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    drop="first"
)
```

---

# 10. Practical Workflow

---

## Step 1 — Correlation Matrix

Check highly correlated features.

---

## Step 2 — Calculate VIF

Identify problematic variables.

---

## Step 3 — Decide Strategy

Possible strategies:
- Remove features
- PCA
- Ridge/Lasso
- Feature engineering

---

## Step 4 — Compare Model Performance

Always compare:
- Before handling
- After handling

---

# 11. Important Notes

---

## Prediction vs Interpretation

For prediction:
- Multicollinearity may not hurt much

For interpretation:
- Very problematic

---

## Tree Models

Usually tolerate multicollinearity better.

---

## Linear Models

Very sensitive.

---

# 12. Interview Questions

---

## Why is multicollinearity dangerous?

Because coefficients become unstable and unreliable.

---

## Why does Ridge Regression help?

Because it penalizes large coefficients.

---

## Difference between Ridge and Lasso?

| Ridge | Lasso |
|---|---|
| Shrinks coefficients | Can remove coefficients |
| L2 regularization | L1 regularization |

---

## Why does PCA help?

Because PCA creates uncorrelated components.

---

## What is the dummy variable trap?

Perfect multicollinearity from one-hot encoding all categories.

---

# 13. Quick Recommendations

| Situation | Recommended Method |
|---|---|
| Simple linear model | Remove correlated features |
| Need interpretability | Ridge Regression |
| Feature selection needed | Lasso |
| High-dimensional data | PCA |
| Tree models | Usually ignore |

---

# 14. Important Warning

Do NOT blindly remove features only because:
```python
corr > 0.8
```

Sometimes correlated features still contain:
- Useful business meaning
- Different predictive signals

Always:
- Validate with performance
- Use domain knowledge

---

# 15. Libraries

## Install

```bash
pip install statsmodels seaborn
```

---

# 16. Final Notes

Multicollinearity is mostly:
- A statistical interpretation problem
- A linear model problem

Modern tree models usually handle it reasonably well.

However:
- Interpretability
- Stability
- Feature importance

can still become problematic.

Always:
- Check correlations
- Calculate VIF
- Compare model performance
- Use domain knowledge

---

# Transformer Refresher

Original Transformer paper:

https://arxiv.org/abs/1706.03762