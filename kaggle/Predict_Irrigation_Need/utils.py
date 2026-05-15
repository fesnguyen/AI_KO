from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display
import mlflow
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ======================
# LOAD DATA
# ======================
def load_and_clean_df(url):
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

    cat_cols = df.select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    
    return df


# ======================
# Prepare overview
# ======================
def show_overview(df, n_heads=5):
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


# ======================
# Quick lightGBM model check
#
# Purpose: Perform a rapid baseline evaluation of a LightGBM classifier on the dataset 
# to identify initial performance and potential issues (e.g., class imbalance, feature importance).
# ======================

def lightgbm_check(train_test_set):
    X_train, X_test, y_train, y_test = (train_test_set)
    
    # 1. Train with Balanced Weights (Critical for your weak 'High' class)
    model = LGBMClassifier(
        reg_alpha=0.1,
        importance_type='gain',
        class_weight='balanced',
        random_state=42,
        n_estimators=1000,
        verbosity=-1   # silence internal spam
    )

    cat_cols = X_train.select_dtypes(exclude=np.number).columns

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=['train', 'valid'],
        eval_metric='logloss',
        categorical_feature=cat_cols.tolist(),
        callbacks=[
            log_evaluation(50),        # clean periodic logs
            early_stopping(100)        # stop when no improvement
        ]
    )

    # 2. Visual Evaluation: Confusion Matrix
    # This shows exactly where the model confuses 'High' with 'Medium'
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', ax=ax[0])
    ax[0].set_title("Confusion Matrix")

    # 3. Feature Importance Graph
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
    feat_imp.plot(kind='barh', ax=ax[1], color='skyblue')
    ax[1].set_title("Feature Importance (Gain)")

    plt.tight_layout()
    plt.show()

    feat_imp.to_csv("feature_importance.csv", header=['importance'])

    # 4. Numeric Evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    print(report_text)

    with open("classification_report.txt", "w") as f:
        f.write(report_text)


# ======================
# Linear model comparison check
#
# Purpose: Compare baseline linear classifiers on the same train/test split
# to see whether a linear separator or logistic classifier can compete with tree-based models.
# ======================

def linear_models_check(train_test_set):
    model_specs = [
        ("LinearSVC", LinearSVC(random_state=42, class_weight='balanced', max_iter=5000)),
        # ("SVC_rbf", SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)),
        ("LogisticRegression", LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000, solver='lbfgs', multi_class='auto'))
    ]

    X_train, X_test, y_train, y_test = train_test_set
    cat_cols = X_train.select_dtypes(exclude=np.number).columns
    num_cols = X_train.select_dtypes(include=np.number).columns

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # fit on train
    X_train_cat = enc.fit_transform(X_train_scaled[cat_cols])
    X_test_cat  = enc.transform(X_test_scaled[cat_cols])

    cat_feature_names = enc.get_feature_names_out(cat_cols)

    # build DataFrames
    X_train_cat = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train_scaled.index)
    X_test_cat  = pd.DataFrame(X_test_cat,  columns=cat_feature_names, index=X_test_scaled.index)

    # combine cleanly
    X_train_final = pd.concat([X_train_scaled[num_cols], X_train_cat], axis=1)
    X_test_final  = pd.concat([X_test_scaled[num_cols],  X_test_cat], axis=1)

    results = []

    for model_name, model in model_specs:
        with mlflow.start_run(run_name=f"LinearModels_{model_name}", nested=True):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_rows", X_train.shape[0])
            mlflow.log_param("test_rows", X_test.shape[0])
            mlflow.log_param("num_features", X_train.shape[1])
            mlflow.log_param("num_categorical_features", len(cat_cols))
            mlflow.log_param("num_numerical_features", len(num_cols))

            model.fit(X_train_final, y_train)

            y_pred = model.predict(X_test_final)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_text = classification_report(y_test, y_pred)
            print(f"\n=== {model_name} ===")
            print(report_text)

            metrics = {
                "model": model_name,
                "accuracy": report["accuracy"],
                "precision_weighted": report["weighted avg"]["precision"],
                "recall_weighted": report["weighted avg"]["recall"],
                "f1_weighted": report["weighted avg"]["f1-score"]
            }

            for metric_name, value in metrics.items():
                if metric_name != "model":
                    mlflow.log_metric(metric_name, value)

            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_estimator(model, X_test_final, y_test, cmap='Blues', ax=ax)
            ax.set_title(f"{model_name} Confusion Matrix")
            plt.tight_layout()
            mlflow.log_figure(fig, f"{model_name}_confusion_matrix.png")
            plt.close(fig)

            with open(f"{model_name}_classification_report.txt", "w") as f:
                f.write(report_text)
            mlflow.log_artifact(f"{model_name}_classification_report.txt")

            results.append(metrics)

    results_df = pd.DataFrame(results)
    display(results_df)
    return results_df

# ======================
# Model Interpretability
# ======================

def feature_interpretability(train_test_set):
    # Rebuild the preprocessing from the linear model stage so we can inspect coefficients and tree behavior.
    X_train, X_test, y_train, y_test = train_test_set
    cat_cols = X_train.select_dtypes(exclude=np.number).columns
    num_cols = X_train.select_dtypes(include=np.number).columns

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = enc.fit_transform(X_train_scaled[cat_cols])
    X_test_cat = enc.transform(X_test_scaled[cat_cols])
    cat_feature_names = enc.get_feature_names_out(cat_cols)

    X_train_final = pd.concat([
        X_train_scaled[num_cols],
        pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train_scaled.index)
    ], axis=1)
    X_test_final = pd.concat([
        X_test_scaled[num_cols],
        pd.DataFrame(X_test_cat, columns=cat_feature_names, index=X_test_scaled.index)
    ], axis=1)

    # Fit interpretable models on the same data
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000, solver='lbfgs')
    tree_model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5)

    lr_model.fit(X_train_final, y_train)
    tree_model.fit(X_train_final, y_train)

    # Table of logistic regression coefficients with cluster numbers
    coef_df = pd.DataFrame({
        'feature': X_train_final.columns,
        'logistic_coef': np.mean(lr_model.coef_, axis=0)
    })
    coef_df['abs_coef'] = coef_df['logistic_coef'].abs()
    coef_df = coef_df.sort_values(by='abs_coef', ascending=False).head(20)

    important_features = coef_df['feature'].tolist()
    feature_corr = X_train_final[important_features].corr().abs()
    clusterer = AgglomerativeClustering(n_clusters=4)
    cluster_labels = clusterer.fit_predict(feature_corr)
    coef_df['cluster'] = cluster_labels
    coef_df = coef_df.sort_values(['cluster', 'abs_coef'], ascending=[True, False]).drop(columns='abs_coef')

    display(coef_df)

    # Decision tree structure (top levels)
    plt.figure(figsize=(18, 10))
    plot_tree(
        tree_model,
        feature_names=X_train_final.columns,
        class_names=[str(c) for c in tree_model.classes_],
        filled=True,
        max_depth=3,
        fontsize=10,
        impurity=False
    )
    plt.title('Decision Tree Structure (Top Levels)')
    plt.tight_layout()
    plt.show()