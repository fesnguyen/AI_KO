# Feature Engineering

Goal: transform raw data into features that improve model performance.

---

## 🧠 Principles

* Keep transformations **train → test consistent**
* Avoid **data leakage**
* Prefer **simple + interpretable** features first
* Validate impact with cross-validation

---

## 🔢 1. Binning

### ▶️ Unsupervised

```python
# Equal width (Easy but weak)
def equal_width_binning(x, bins=5):
    return pd.cut(x, bins=bins)

# Quantile binning (Better but still weak)
def quantile_binning(x, bins=5):
    return pd.qcut(x, q=bins, duplicates='drop')


# K-Mean binning (Moderate)
from sklearn.cluster import KMeans

def kmeans_binning(x, bins=5):
    x_reshaped = np.array(x).reshape(-1, 1)
    kmeans = KMeans(n_clusters=bins, random_state=42)
    return kmeans.fit_predict(x_reshaped)

# Domain binning (Very strong but rare)

# Logarithmic binning (Apply for skewded distribution)
def log_binning(x, bins=5):
    x_log = np.log1p(x)
    return pd.cut(x_log, bins=bins)
```

---

### ▶️ Supervised

```python
# Decision tree binning (Strong)
from sklearn.tree import DecisionTreeClassifier
def tree_binning(x, y, max_leaf_nodes=5):
    x_reshaped = np.array(x).reshape(-1, 1)
    tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    tree.fit(x_reshaped, y)
    return tree.apply(x_reshaped)

# WOE: weight of evidence binning
# Encodes bins based on event distribution
def woe_binning(df, feature, target, bins=5):
    df['bin'] = pd.qcut(df[feature], bins, duplicates='drop')
    grouped = df.groupby('bin')[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    
    grouped['woe'] = np.log((grouped['sum'] + 1e-6) / (grouped['non_event'] + 1e-6))
    return df['bin'].map(grouped['woe'])

# Chi-Square binning
# Merges bins based on statistical similarity
from sklearn.preprocessing import KBinsDiscretizer

def chi_square_like_binning(x, bins=5):
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    return est.fit_transform(np.array(x).reshape(-1, 1)).ravel()
```

---

## 🧩 2. Combine Categorical Columns

```python
# Simple concatenation
def combine_features(df, col1, col2): 
    return df[col1].astype(str) + "_" + df[col2].astype(str)

# Multiple feature crossing
def combine_multiple(df, cols): 
    return df[cols].astype(str).agg("_".join, axis=1)

# Frequency Encoding on Combined Feature
def frequency_encode(series): 
    freq = series.value_counts(normalize=True) 
    return series.map(freq)

# Target encoding
def target_encode(train, col, target): 
    mapping = train.groupby(col)[target].mean() 
    return train[col].map(mapping)

# One hot encoding after Combination
df["city_device"] = combine_features(df, "city", "device") 
df = pd.get_dummies(df, columns=["city_device"])

# Count encoding
def count_encode(series): 
    return series.map(series.value_counts())
```

👉 Captures interaction between categories

---

## 📊 3. Groupby Aggregations (NON target-aware)

### ▶️ Generic

```python
train["agg_mean"] = train.groupby("COL1")["COL2"].transform("mean")
test["agg_mean"]  = test["COL1"].map(train.groupby("COL1")["COL2"].mean())
```

---

### ▶️ Multiple stats

```python
agg = train.groupby("COL1")["COL2"].agg(["mean", "std", "min", "max"])

train = train.join(agg, on="COL1")
test  = test.join(agg, on="COL1")
```

---

👉 Use when:

* hierarchical structure exists
* feature interactions needed

---

## 🎯 4. Target Encoding (MOST POWERFUL ⚠️)

### ▶️ Basic (unsafe if done naively)

```python
target_mean = train.groupby("COL1")["target"].mean()

train["te"] = train["COL1"].map(target_mean)
test["te"]  = test["COL1"].map(target_mean)
```

---

⚠️ Risk: **leakage**

---

### ▶️ Proper (OOF target encoding — recommended)

```python
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train["te"] = np.nan

for tr_idx, val_idx in kf.split(train):
    tr, val = train.iloc[tr_idx], train.iloc[val_idx]
    
    mapping = tr.groupby("COL1")["target"].mean()
    train.loc[val_idx, "te"] = val["COL1"].map(mapping)

# test uses full train
mapping = train.groupby("COL1")["target"].mean()
test["te"] = test["COL1"].map(mapping)
```

---

## 📈 5. Target-based Groupby Features

### ▶️ Mean

```python
mean_price = train.groupby("COL1")["Price"].mean()

train["price_mean"] = train["COL1"].map(mean_price)
test["price_mean"]  = test["COL1"].map(mean_price)
```

---

### ▶️ Quantiles

```python
q = train.groupby("COL1")["Price"].quantile(0.75)

train["price_q75"] = train["COL1"].map(q)
test["price_q75"]  = test["COL1"].map(q)
```

---

### ▶️ Histogram bins (distribution encoding)

```python
bins = np.linspace(train["Price"].min(), train["Price"].max(), 5)

train["price_bin"] = np.digitize(train["Price"], bins)
test["price_bin"]  = np.digitize(test["Price"], bins)
```

---

👉 Captures distribution shape instead of single stat

---

## ⚖️ Target-aware vs Non-target-aware

| Type         | Example                     | Risk         |
| ------------ | --------------------------- | ------------ |
| Non-target   | binning, groupby(COL2)      | safe         |
| Target-aware | target encoding, price mean | leakage risk |

---

## ⚠️ Common Mistakes

* ❌ fitting on full data (leakage)
* ❌ not using OOF for target encoding
* ❌ applying different transforms to test
* ❌ over-engineering features

---

## 🧠 Workflow

1. Start simple (binning, interactions)
2. Add groupby stats
3. Add target encoding carefully
4. Validate with CV

---

## ⚡ TL;DR

* Binning → handle non-linearity
* Combine → interactions
* Groupby → structure
* Target encoding → strongest (but risky)
