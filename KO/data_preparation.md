# Prepare the data
1. Handle missing values
2. Create meaningful features
3. Encoding categorical features

## 1. Handle missing values

## # 1.1 What are missing values?
Missing values are common in real-world datasets and can significantly affect model performance. Before training models, it's crucial to detect and handle missing data properly.

## # 1.2 Why should we handle missing values?
- Bias in model if missing values are not random:
  - **MCAR (Missing Completely At Random, low bias risk)**: Probability of missing data is the same for all observations (completely unrelated to any variable) - Ex: A sensor randomly fails 5% of the time regardless of conditions.
  - **MAR (Missing At Random, moderate bias risk)**: Missingness is related to other observed variables, but not the missing one itself - Ex: Younger people more likely to skip income field, but age is known.
  - **MNAR (Missing Not At Random, high bias risk)**: Missingness is related to the value that is missing - Ex: People with higher income are more likely to skip the income question.
    
- Certain algorithms (like sklearn’s RandomForest) handle NaNs poorly (Since it can't compare NaN with number).
- Degrades model performance or causes crashes.

## # 1.3 How to Handle Missing Values

```python
# Check for missing
missing_percent = df.isnull().mean() * 100
print(missing_percent)


# Visualize missing data
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull(), cbar=False, cmap='magma')
plt.title("Missing Data Heatmap")
plt.show()


# Option 1. If data set is small with very few missing entries
# Drop rows with any missing value
df_clean = df.dropna()
# Drop columns with any missing value
df_clean_cols = df.dropna(axis=1)


# Option 2. If Missing data is MCAR or MAR, and distribution is not skewed
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
# Drop missing values
data = df[feature].dropna()
# Compute skewness
skewness_value = skew(data)
# Plot
plt.figure(figsize=(8, 4))
sns.histplot(data, kde=True, bins=30, color='skyblue')
plt.title(f"Distribution of '{feature}' | Skewness: {skewness_value:.2f}")
plt.xlabel(feature)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# abs(skewness_value) < 0.5: Distribution is approximately symmetric.
# skewness_value > 0: Positively skewed (right tail).
# Negatively skewed (left tail).

# Fill numerical columns with mean/median/mode
# If data is skewed, mean can distort data -> use median instead
df['Age'].fillna(df['Age'].mean(), inplace=True)


# Option 3. Using SimpleImputer
# Preferred over manual .fillna(), especially when used in Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
# For numerical
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['Age', 'Salary']] = imp_mean.fit_transform(df[['Age', 'Salary']])
# For categorical
imp_mode = SimpleImputer(strategy='most_frequent')
df[['Department']] = imp_mode.fit_transform(df[['Department']])


# Option 4. Using KNN Imputation
# When there's a correlation between features (e.g., Age ↔ Salary) and dataset is not too large
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# Option 5. Traing a model to predict missing values based on other features
# For important features with many missing values and strong correlation with other columns
from sklearn.linear_model import LinearRegression
# Example: Predict missing Salary using Age
known = df[df['Salary'].notnull()]
unknown = df[df['Salary'].isnull()]
model = LinearRegression()
model.fit(known[['Age']], known['Salary'])
predicted_salary = model.predict(unknown[['Age']])
df.loc[df['Salary'].isnull(), 'Salary'] = predicted_salary

# Option 6. Mark misisng values with flags, not death flag =))
# To let the model learn if the fact a value is missing carries predictive information
df['Salary_missing'] = df['Salary'].isnull().astype(int)


# Note:
# Categorical Mode Imputation works well if there's a dominant category.
# KNN and Predictive Imputation are more accurate but slower and may overfit if not careful.
# Always validate the model after imputation to check if performance improved or degraded.
```

## 2. Feature Engineering: Creating Meaningful Variables

## 2.1 What are meaningful variables?
These are new features created from existing raw data that:
- Improve the model's ability to learn
- Capture domain knowledge or hidden relationships
- Simplify complex patterns

**General Process**
1. Understand the business/domain problem
2. Analyze raw data and relationships
3. Create new features using logic, math, or patterns
4. Validate usefulness (visualization, model feedback)
5. Avoid leakage (don’t use future or target info)

## # 2.3 Ways to create new meaningful features?

```python
# 1. Mathematical Transformations
# Reveal interactions or proportional relationships between features (e.g. normalize income by age).
df['income_per_age'] = df['income'] / df['age']
df['net_worth'] = df['assets'] - df['liabilities']


# 2. Aggregation Features
# Capture user or group-level behavior patterns, especially useful in transactional or time-series data.
df['customer_avg_amt'] = df.groupby('customer_id')['amount'].transform('mean')
# Other aggregation types: sum, min, max, std, count


# 3. Time-Based Features
# Extract behavioral trends or seasonality; time of day or day of week often influences outcomes.
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['days_since_signup'] = (df['timestamp'] - df['signup_date']).dt.days
# Also Lag/lead values, Rolling averages, Time since last event


# 4. Ratios & Proportions
# Normalize absolute values; helps the model generalize across different scales or sizes.
df['loan_to_income'] = df['loan_amount'] / df['income']
df['click_rate'] = df['clicks'] / df['impressions']


# 5. Target-Based Features (careful with leakage)
# Encodes the relationship between categories and the target; powerful but prone to leakage if not handled carefully.
df['category_target_mean'] = df.groupby('category')['target'].transform('mean')
# Use K-fold encoding or leave-one-out to avoid leakage.


# 6. Binning & Bucketing
# Simplifies continuous data, reduces model sensitivity to outliers, and adds interpretability.
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=["Teen", "Young", "Adult", "Senior"])


# 7. Text Feature Creation
# Quantifies language content for models to process; surface-level features often correlate with intent or sentiment.
df['text_length'] = df['review'].apply(len)
df['num_words'] = df['review'].apply(lambda x: len(x.split()))
df['has_free'] = df['review'].str.contains("free", case=False).astype(int)
# Also: n-grams, TF-IDF, embeddings (advanced)


# 8. Domain-Specific Features
# Bring in domain knowledge to create impactful variables.
df['risk_score'] = (df['claims_last_3yrs'] * 5) + df['accidents']


# 9. Polynomial or Interaction Features
# Helps linear models capture non-linear relationships and feature interactions.
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])
# Only useful for linear models or if strong interactions exist.


# 10. Frequency / Count Encoding - Replace categories with their frequency in the dataset
# onverts categorical popularity into a numeric form, often correlates with importance or exposure.
df['product_freq'] = df['product_id'].map(df['product_id'].value_counts())
```

## # 2.3 How to validate feature quality
- Correlation with target
- Feature importance from models
- Cross-validation performance change
- Permutation importance / SHAP values:
  - ```python
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_val, y_val, n_repeats=10)
    ```
  - ```python
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    shap.summary_plot(shap_values, X_val)`
    ```
**Tips & Best Practices**
- Use exploratory analysis (EDA) to inspire new features.
- Avoid overfitting with high-cardinality target encoding (Regularization (Smoothing), K-Fold Target Encoding, Leave-One-Out (LOO) Encoding).
- Use pipelines to automate feature creation reproducibly.
- Use cross-validation to validate performance gain.

## 3. Encoding Categorical Variables

## # 3.1 Label Encoding (Encode the label y)
- **Use when**:
  - The category has ordinal meaning (e.g., "Low" < "Medium" < "High").
  - Tree-based models (e.g., Decision Trees, XGBoost) can handle it well (Usually no need).
- **Avoid When**:
  - The model is linear or distance-based (e.g., KNN, Logistic Regression) — it may interpret the labels as having magnitude.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['education_level'] = le.fit_transform(df['education_level'])
```

## # 3.2 One-Hot Encoding
- **Use when**:
  - Categories are nominal (no inherent order).
  - The number of unique categories is low to medium (≤50 is manageable).
- **Avoid When**:
  - High-cardinality feature (hundreds or thousands of categories).

```python
pd.get_dummies(df, columns=['color'], drop_first=True)
```

## # 3.3 Ordinal Encoding
- **Use when**:
  - Categories have a known order, and you want to reflect that in the numbers.

```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df[['priority']] = encoder.fit_transform(df[['priority']])
```

## # 3.4 High-cardinality target encoding (When too many unique categorical)
1. Mean encoding:
   - Use when the categorical feature is strong correlated to the target
   - Risk: Data leakage or overfit
`df['category_encoded'] = df.groupby('category')['target'].transform('mean')`
2. Frequency Encoding:
   - Use When: Categories' popularity matters (e.g. more frequent = more trustworthy) or to minimal computational cost.
`df['category_freq'] = df['category'].map(df['category'].value_counts())`
3. Binary Encoding / Hashing (Advanced)
   - Use when: Very high cardinality (1000+ values).
```python
import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['category'])
df = encoder.fit_transform(df)
```

## # 3.4.1 How to avoid overfiting while using these encodings?
1. Regularization (Smoothing) for **rare categories**: Adds a prior to the mean to reduce overfitting for rare categories
```python
encoded_value = (count * mean_target + prior * global_mean) / (count + prior)
```
- Helps pull rare categories toward the overall mean
- Prevents extreme values for infrequent labels
  
2. K-Fold Target Encoding for **Medium-to-large datasets**:
   - Split your data into K folds
   - Encode each fold using the target mean of the other K−1 folds
   - Ensures the category's value does not use its own label during encoding
- -> Prevents leakage and maintains realistic performance during training.

3. Leave-One-Out (LOO) Encoding for **Small datasets, time-series**: For each row, compute the mean target excluding that row from the calculation.
- This technique balances between preserving info and avoiding label leakage — especially good when dataset is small.

## 4. Normalization / Standardization
**To scale numeric input features so that**:
- The model trains faster.
- Gradients converge more smoothly.
- Feature scales don't dominate others in distance-based or gradient-based algorithms.

**Scientific Rationale***
- Many ML algorithms assume data is centered and scaled:
- Gradient descent-based models (e.g., linear/logistic regression, neural networks) are sensitive to feature magnitude.
- Distance-based models (e.g., KNN, K-means, SVM with RBF kernel) rely on comparable scales.
- Tree-based models (e.g., Random Forest, XGBoost) don’t require scaling, but normalization may still help when features vary widely or if using PCA before trees.

## # 4.1 Normalization (Min-Max Scaling)
**Properties**:
- Sensitive to outliers.
- Maintains relationships and distribution shape.
- Good for image pixels, bounded input features.

**When to Use**:
- Required for neural networks (especially with sigmoid or tanh).
- Algorithms that assume bounded input.
- Features that have the same measurement unit.

```python
# Scale the data to a fixed range-usually [0,1]
# x_scaled = (x - x.min()) / (x.max() - x.min())

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
X_scaled = minmax.fit_transform(X)
```

## # 4.2 Standardization (Z-score Normalization)
**Properties**:
- Robust to scale differences.
- Assumes Gaussian distribution (but works in practice even when skewed).
- Preserves outliers more than normalization.

**When to Use**:
- Logistic regression, linear regression, SVMs, PCA.
- When you’re not sure about the distribution.

```python
# Rescales data to have zero mean and unit variance.
# x_standardized = (x - x.mean()) / x.std()

from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
X_standardized = standard.fit_transform(X)
```

## # 4.3 Advanced Options
- **RobustScaler**: Uses median and IQR; good for outliers.
- **QuantileTransformer**: Maps to uniform or normal distribution.
```python
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
X_trans = qt.fit_transform(X)
```
- **PowerTransformer**: Makes data more Gaussian-like (Yeo-Johnson, Box-Cox).
```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
X_trans = pt.fit_transform(X)
```

## # 4.4 Visual Comparision
| Method          | Mean | Std Dev | Range     |
| --------------- | ---- | ------- | --------- |
| Standardization | 0    | 1       | Unbounded |
| Min-Max         | ≠0   | ≠1      | \[0, 1]   |
```python
import matplotlib.pyplot as plt
plt.hist(X_standardized[:, 0], bins=50)
```

## # 4.5 Summary Table
| Use Case                       | Method           |
| ------------------------------ | ---------------- |
| Neural networks (sigmoid/tanh) | Normalization    |
| Linear/Logistic Regression     | Standardization  |
| KNN / SVM / KMeans             | Standardization  |
| Features have outliers         | RobustScaler     |
| Skewed distributions           | PowerTransformer |

## 5. Split into Train / Validation / Test Sets
To evaluate model performance accurately and prevent overfitting by simulating unseen data during training.

```python
# Basic
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Argu: stratify=y for classification problem

# Cross-Validation (for Small/Medium Datasets)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
#✔ Reduces variance from a single validation set
#✔ Useful when data is scarce

# Time Series Splitting
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
# Never shuffle time series data randomly.
```

## # 📌 5.1 Summary Table
| **Scenario** | **Best Practice** |
| -------------|-------------------|
| Standard tabular data | train_test_split + val split |
| Small dataset | Cross-validation |
| Classification | Use stratify=y |
| Time-series | Use TimeSeriesSplit |
| Final performance check | Only on test set |


---

# Best Practices Summary

## Missing Values
- Understand *why* data is missing before imputing.
- Use median for skewed numerical distributions.
- Use mode for categorical variables.
- Avoid dropping rows aggressively unless missingness is severe.

## Feature Engineering
- Create features that reflect meaningful relationships.
- Keep engineered features interpretable.
- Avoid creating too many noisy or redundant features.

## Encoding Categorical Features
- Use One-Hot Encoding for low-cardinality categories.
- Use Label Encoding mainly for ordinal categories.
- Be careful with high-cardinality features to avoid dimensional explosion.

## General Advice
- Always split train/validation data before preprocessing to avoid data leakage.
- Fit scalers/encoders only on training data.
- Track preprocessing steps carefully for reproducibility.
