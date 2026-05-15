# Ensemble Methods

## Overview

Combine multiple models to improve performance.

---

## 1. Bagging (Bootstrap Aggregating)

**Idea:** train models independently on resampled data → reduce variance

**Example: Random Forest**

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
```

---

## 2. Boosting

**Idea:** train sequentially → focus on previous errors → reduce bias

**Example: LightGBM**

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        log_evaluation(100),
        early_stopping(100)
    ]
)
```

---

## 3. Stacking (OOF – proper way)

**Idea:** combine models using a meta-model (no leakage)

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

base_models = [
    ("lgbm", LGBMClassifier(n_estimators=300)),
    ("lr", LogisticRegression(max_iter=1000))
]

model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)

model.fit(X_train, y_train)
```

---

## 4. Blending (simpler stacking)

**Idea:** use holdout set instead of OOF

```python
# split manually
X_tr, X_hold, y_tr, y_hold = train_test_split(X_train, y_train, test_size=0.2)

# train base models
m1 = LGBMClassifier().fit(X_tr, y_tr)
m2 = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

# create meta features
meta_X = np.column_stack([
    m1.predict_proba(X_hold)[:, 1],
    m2.predict_proba(X_hold)[:, 1]
])

# train meta model
meta_model = LogisticRegression().fit(meta_X, y_hold)
```

---

## Quick Comparison

| Method   | Goal            | When to use               |
| -------- | --------------- | ------------------------- |
| Bagging  | reduce variance | unstable models           |
| Boosting | reduce bias     | strong baseline (default) |
| Stacking | combine models  | when models are diverse   |
| Blending | quick stacking  | fast experiments          |

---

## Practical Rules

* Start with **Boosting (LightGBM/XGBoost)**
* Try **Stacking** only if models are diverse
* Avoid overcomplicating → validate with CV

---

## Common Mistakes

* ❌ stacking without OOF → leakage
* ❌ too many similar models
* ❌ no validation

---

## TL;DR

* Bagging → parallel
* Boosting → sequential
* Stacking → meta-learning
* Blending → shortcut stacking
