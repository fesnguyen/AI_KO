# Evaluation

## Validate feature engineering step

```python
from sklearn.model_selection import cross_val_score

def evaluate(pipeline, X, y):
    scores = cross_val_score(pipeline, X, y, cv=5)
    return scores.mean(), scores.std()

eval_mean, eval_std = evaluate(with_binning, X, y)
```


---