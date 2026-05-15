# EDA (Exploratory Data Analysis)

Goal: understand data, detect issues, guide modeling.

---

## 0. Descriptions

* Overview structure:
  + n_count = 2: binary
  + ordinal encoder != label encoder
* Missing values
* Missing strategy
* Target analysis
* Feature types
* Distribution (features)
* Feature vs Target
* Correlation
* Leakage check
* Quick model check

---

## 1. Basic Overview

### ▶️ Shape + Head + null% + ...

[Load_Clean_DF + Show_Overview](utilities.md#️-load-and-clearn-df)

---

### ▶️ Info & types

```python
df.info()
```

---

### ▶️ Summary stats

```python
df.describe(include="all")
```

---

## 2. Missing Values

```python
pd.DataFrame({
    "non_null": df.notna().sum(),
    "null_%": df.isna().mean() * 100
}).sort_values("null_%", ascending=False)
```

👉 Watch:

* high missing columns
* patterns (not random)

---

## 3. Unique / Cardinality

```python
df.nunique().sort_values(ascending=False)
```

👉 Detect:

* IDs (very high cardinality)
* categorical vs numeric

---

## 4. Distribution (Univariate)

### ▶️ Numeric

```python
df.hist(figsize=(12, 8), bins=30)
plt.show()
```

---

### ▶️ Single feature

```python
sns.histplot(df["feature"], kde=True)
plt.show()
```

👉 Watch:

* skewness
* outliers
* zero-inflation

---

## 5. Categorical Analysis

```python
df["category"].value_counts().plot(kind="bar")
plt.show()
```

👉 Watch:

* imbalance
* rare categories

---

## 6. Correlation

```python
# ======================
# Feature Correlation Analysis
#
# Purpose: Examine linear relationships between numerical features and the target variable
# to identify potential predictors, multicollinearity, and feature engineering opportunities.
#
# Methodology:
# - Encode target as ordinal (Low=0, Medium=1, High=2) for correlation computation.
# - Include only numerical columns (categorical features excluded to avoid encoding complexity).
# - Compute Pearson correlation matrix.
# - Visualize with heatmap for intuitive interpretation.
#
# Interpretation Guidelines:
# - Correlation coefficients range from -1 to 1.
# - |r| > 0.7: Strong correlation (potential multicollinearity or key predictors).
# - |r| 0.3-0.7: Moderate correlation (useful for feature selection).
# - |r| < 0.3: Weak correlation (may not be predictive alone).
# - Focus on target correlations for feature importance; check feature-feature correlations for redundancy.
#
# Expected Insights: Real-world data often shows weak correlations; combine with domain knowledge and non-linear methods.
# ======================

df[TARGET] = df[TARGET].map({'Low': 0, 'Medium': 1, 'High': 2})

# Select numerical columns for correlation analysis
# - Excludes categorical features to focus on quantitative relationships
num_cols = df.select_dtypes(include=np.number).columns

# Include target in correlation matrix
# - Allows assessment of feature-target relationships
num_plus_target = num_cols.tolist()
num_plus_target.append(TARGET)

# Compute correlation matrix
# - Uses Pearson correlation by default (linear relationships)
corr = df[num_plus_target].corr()

# Visualize correlations with heatmap
# - Color scale: Red (positive), Blue (negative), White (neutral)
# - Annotations show exact coefficients for precision
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix (Including Target)')
plt.show()
```

👉 Watch:

* multicollinearity
* strong relationships

---

## 7. Target Relationship (if available)

### Target balance

```python
# ======================
# Target Variable Analysis: Irrigation Need Classification
#
# Objective: Understand the distribution of the target variable to assess class imbalance,
# which critically impacts model performance and evaluation strategies.
#
# Key Insights:
# - Class imbalance can lead to biased models favoring majority classes.
# - Strategies: Use balanced class weights, resampling (SMOTE/undersampling), or metrics like F1-score/AUC instead of accuracy.
# - Visualization: Count plot reveals frequency; normalized value_counts show proportions.
# ======================

TARGET = 'irrigation_need'

# Visualize class distribution with count plot
# - X-axis: Target classes (e.g., 'Low', 'Medium', 'High' irrigation needs)
# - Y-axis: Absolute counts per class
# - Interpretation: Identifies dominant classes and potential imbalance severity
sns.countplot(x=df[TARGET])
plt.title('Distribution of Irrigation Need Classes')
plt.xlabel('Irrigation Need Level')
plt.ylabel('Count')
plt.show()

# Quantitative analysis: Normalized value counts
# - Returns proportions (0-1) for each class
# - Helps quantify imbalance ratio (e.g., minority class < 10% may require special handling)
# - Use for baseline expectations and resampling decisions
df[TARGET].value_counts(normalize=True)
```

### ▶️ Numeric vs target

```python
sns.scatterplot(x=df["feature"], y=df["target"])
plt.show()
```

---

### ▶️ Categorical vs target

```python
sns.boxplot(x=df["category"], y=df["target"])
plt.show()
```

👉 Watch:

* signal strength
* separability

---

## 8. Outliers

```python
sns.boxplot(x=df["feature"])
plt.show()
```

---

## 9. Distribution Shift (Train vs Test)

```python
sns.histplot(train["feature"], color="blue", label="train", stat="density")
sns.histplot(test["feature"], color="red", label="test", stat="density")
plt.legend()
plt.show()
```

👉 Watch:

* mismatch → potential generalization issues

---

## 10. Temporal Trend (if time exists)

```python
df.groupby("date")["target"].mean().plot()
plt.show()
```

👉 Watch:

* drift over time
* seasonality

---

## 11. Quick Checklist

* [ ] Missing values checked
* [ ] Distribution reviewed
* [ ] Outliers detected
* [ ] Correlation checked
* [ ] Target relationship explored
* [ ] Train/test shift checked

---

## ⚠️ Common Pitfalls

* ❌ ignoring data leakage
* ❌ over-trusting correlation
* ❌ skipping visualization
* ❌ mixing train/test during EDA

---

## ⚡ TL;DR

EDA = understand data → find issues → guide feature engineering & modeling
