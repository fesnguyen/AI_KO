# Hyperparameter Tuning in Machine Learning

Hyperparameter tuning is the process of finding the best configuration for a machine learning model.

Hyperparameters are NOT learned during training.

Instead, they are manually chosen before training starts.

Examples:
- Learning rate
- Number of trees
- Max depth
- Batch size
- Number of layers

---

# 1. Parameters vs Hyperparameters

---

## Parameters

Learned automatically during training.

Examples:
- Weights
- Biases

---

## Hyperparameters

Chosen manually.

Examples:
```python
learning_rate = 0.01
max_depth = 6
n_estimators = 100
```

---

# 2. Why Hyperparameter Tuning Matters

Bad hyperparameters can cause:
- Overfitting
- Underfitting
- Slow training
- Poor generalization

Good hyperparameters can dramatically improve performance.

---

# 3. Common Hyperparameters

---

# 3.1 Linear Regression

Usually few hyperparameters.

Examples:
```python
alpha
```

for Ridge/Lasso regularization.

---

# 3.2 Decision Trees

Important hyperparameters:

| Hyperparameter | Meaning |
|---|---|
| max_depth | Maximum tree depth |
| min_samples_split | Minimum samples to split |
| min_samples_leaf | Minimum samples in leaf |

---

# 3.3 Random Forest

| Hyperparameter | Meaning |
|---|---|
| n_estimators | Number of trees |
| max_depth | Tree depth |
| max_features | Features per split |

---

# 3.4 XGBoost / LightGBM

Very important hyperparameters:

| Hyperparameter | Meaning |
|---|---|
| learning_rate | Step size |
| n_estimators | Number of trees |
| max_depth | Tree complexity |
| subsample | Row sampling |
| colsample_bytree | Feature sampling |
| reg_alpha | L1 regularization |
| reg_lambda | L2 regularization |

---

# 3.5 Neural Networks

| Hyperparameter | Meaning |
|---|---|
| learning_rate | Optimization speed |
| batch_size | Samples per update |
| epochs | Training iterations |
| hidden_layers | Model complexity |
| dropout | Regularization |

---

# 4. Underfitting vs Overfitting

---

## Underfitting

Model too simple.

Symptoms:
- Poor train performance
- Poor validation performance

---

## Overfitting

Model memorizes training data.

Symptoms:
- Excellent train performance
- Poor validation performance

---

# 5. Validation Strategy

Never tune on the test set.

Correct workflow:

```text
Train Set -> Validation Set -> Test Set
```

---

# 6. Cross Validation

More reliable evaluation.

---

## K-Fold Cross Validation

Example:
- Split into 5 folds
- Train 5 times
- Average scores

---

## Example

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(scores.mean())
```

---

# 7. Grid Search

Tries ALL parameter combinations.

---

## Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid=params,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
```

---

## Advantages
- Exhaustive
- Finds strong combinations

## Disadvantages
- Very slow

---

# 8. Random Search

Tests RANDOM combinations.

Usually faster than Grid Search.

---

## Example

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=params,
    n_iter=5,
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)

print(random_search.best_params_)
```

---

## Advantages
- Faster
- Efficient for large spaces

## Disadvantages
- May miss best combination

---

# 9. Bayesian Optimization

Smart hyperparameter search.

Instead of random guessing:
- Learns from previous trials
- Searches promising regions

Popular libraries:
- Optuna
- Hyperopt
- BayesianOptimization

---

# 10. Optuna (Very Popular)

Modern tuning library.

Very useful for:
- XGBoost
- LightGBM
- Neural networks

---

## Example

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):

    n_estimators = trial.suggest_int(
        "n_estimators",
        100,
        500
    )

    max_depth = trial.suggest_int(
        "max_depth",
        3,
        20
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    score = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring="f1"
    ).mean()

    return score

study = optuna.create_study(
    direction="maximize"
)

study.optimize(objective, n_trials=20)

print(study.best_params)
```

---

# 11. Important Hyperparameters for Gradient Boosting

---

# Learning Rate

Controls update step size.

Small learning rate:
- Slower learning
- Better generalization

Large learning rate:
- Faster learning
- Higher overfitting risk

---

# Number of Trees

More trees:
- Better learning
- Slower training
- More overfitting risk

---

# Max Depth

Controls tree complexity.

Small depth:
- Simpler model

Large depth:
- Complex model
- More overfitting

---

# Subsample

Fraction of rows sampled.

Helps:
- Reduce overfitting
- Improve generalization

---

# 12. Early Stopping

Stops training when validation score stops improving.

Very important for:
- Neural Networks
- XGBoost
- LightGBM

---

## Example (XGBoost)

```python
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=50,
    verbose=False
)
```

---

# 13. Hyperparameter Tuning Workflow

---

## Step 1 — Baseline Model

Always start simple.

---

## Step 2 — Understand Problem

Consider:
- Dataset size
- Overfitting risk
- Training time

---

## Step 3 — Tune Important Parameters First

Examples:
- learning_rate
- max_depth
- n_estimators

---

## Step 4 — Use Cross Validation

More reliable than single split.

---

## Step 5 — Compare Metrics

Choose appropriate metric:
- F1
- ROC-AUC
- RMSE
- MAE

depending on task.

---

# 14. Common Mistakes

---

## Tuning on Test Set

Wrong:
```text
Train -> Test -> Tune
```

Correct:
```text
Train -> Validation -> Test
```

---

## Searching Too Many Parameters

Huge search spaces become:
- Very slow
- Inefficient

---

## Ignoring Business Metrics

Best ML score may not equal best business outcome.

---

# 15. Practical Recommendations

| Situation | Recommendation |
|---|---|
| Small parameter space | Grid Search |
| Large parameter space | Random Search |
| Serious optimization | Optuna |
| Limited compute | Random Search |
| Deep learning | Bayesian optimization |

---

# 16. Which Models Need More Tuning?

| Model | Tuning Sensitivity |
|---|---|
| Linear Regression | Low |
| Random Forest | Medium |
| XGBoost | High |
| LightGBM | High |
| Neural Networks | Very High |

---

# 17. Typical Starting Values

---

## XGBoost / LightGBM

```python
learning_rate = 0.05
n_estimators = 500
max_depth = 6
subsample = 0.8
colsample_bytree = 0.8
```

---

## Neural Networks

```python
learning_rate = 1e-3
batch_size = 32
epochs = 20
dropout = 0.3
```

---

# 18. Interview Questions

---

## Why does smaller learning rate often improve performance?

Because updates become smoother and less unstable.

---

## Why use cross validation?

Because a single split may be misleading.

---

## Why is Random Search often better than Grid Search?

Because not all hyperparameters are equally important.

---

## Why use early stopping?

To prevent overfitting.

---

# 19. Final Notes

Hyperparameter tuning is:
- Part science
- Part engineering
- Part experimentation

There is no universal best configuration.

Good tuning requires:
- Strong validation strategy
- Proper metrics
- Efficient search methods
- Business understanding

---

# Libraries

## Install

```bash
pip install optuna xgboost lightgbm
```

---

# Transformer Refresher

Original Transformer paper:

https://arxiv.org/abs/1706.03762