# Utilities

## 🧱 General Helpers

### ▶️ General Config

```python
# ======================
# CONFIG
# ======================

pd.set_option("display.max_columns", None)

# mlflow configs
mlflow.set_tracking_uri("http://127.0.0.1:5000")
project_name = "Predict_Irrigation_Need"
mlflow.set_experiment(project_name)
mlflow.lightgbm.autolog()

%xmode Context

# Constants
TARGET = "Irrigation_Need"
```

---

## 📊 Data Inspection

### ▶️ Load and clearn df

```python
def prepare_data(url):
    """
    Load and correct data
    
    - Rename/Remove columns
    - Correct dtypes
    """

    df = pd.read_csv(url)

    # Rename columns
    df.columns = df.columns.str.strip().str.lower()

    # Remove id
    df.drop(['id'], axis=1, inplace=True, errors='ignore')

    # Correct data types
    cat_cols = df.select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    
    return df
```

### ▶️ Quick overview

```python
def show_overview(df, n_heads=5):
    
    """

    """

    # View head
    head = df.head(n_heads)

    # Summary
    summary = pd.DataFrame({
        col: [
            df[col].dtype,
            df[col].notna().sum(),
            df[col].isna().mean() * 100,   # null %
            df[col].nunique()
        ]
        for col in df.columns
    }, index=["dtype", "non_null", "null_%", "nunique"])
    
    combined = pd.concat([summary, head])
    
    # Simplified styling using pandas built-in methods
    styled_df = (combined.style
                 .format({'null_%': '{:.1f}%'})  # Format null percentage
                 .set_caption('Data Overview')   # Add caption
                 .apply(lambda x: ['font-weight: bold' if x.name in summary.index else '' for _ in x], axis=1)  # Highlight summary rows
                 .set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}])  # Bold headers
                )
    
    display(styled_df)
```

---

### ▶️ Non-null + null summary

```python
def null_summary(df):
    return pd.DataFrame({
        "non_null": df.notna().sum(),
        "null_%": df.isna().mean() * 100
    })
```

---

## 📈 Evaluation

### ▶️ Confusion matrix plot

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(model, X, y):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax)
    plt.show()
```

---

### ▶️ Feature importance (LightGBM)

```python
def plot_importance(model):
    import pandas as pd
    import matplotlib.pyplot as plt

    imp = pd.Series(model.feature_importances_, index=model.feature_name_)
    imp.sort_values().tail(20).plot(kind="barh")
    plt.title("Feature Importance")
    plt.show()
```

---

## 🔄 Feature Engineering

### ▶️ Quantile binning

```python
from sklearn.preprocessing import KBinsDiscretizer

def quantile_bin(train, test, col, n_bins=5):
    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    
    train[col + "_bin"] = kb.fit_transform(train[[col]]).astype(int).ravel()
    test[col + "_bin"] = kb.transform(test[[col]]).astype(int).ravel()
    
    return train, test
```

---

## 🧪 Validation

### ▶️ Train / validation split

```python
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed)
```

---

## ⚙️ MLflow Logging

### ▶️ Log confusion matrix

```python
import mlflow

def log_confusion_matrix(fig):
    mlflow.log_figure(fig, "confusion_matrix.png")
```

---

## 🧠 Tips

* Keep functions **small + reusable**
* Avoid hardcoding column names
* Prefer returning DataFrame, not numpy arrays

---

## ⚡ TL;DR

* organize by purpose (EDA / FE / Eval)
* keep functions copy-paste ready
* reuse across notebooks
