import pandas as pd
import numpy as np
import os
import time
from typing import Union, List, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class RegressionPipeline:
    def __init__(self, df: pd.DataFrame, label: str = None, version: int = 0):
        """
        Initialize the RegressionPipeline with the dataset, label column, and log version.
        """
        self.df = df.copy()
        self.label = label
        self.version = version
        self.model = None
        self.X = df.drop(columns=[label]) if label else df
        self.y = df[label] if label else None
        self.log_file = f"MLLog_{version}.csv"

        print("Regression Pipeline created successfully!")
        self._init_log()

    def _init_log(self):
        """Create the log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("time,message,runTime,accuracy,note\n")

    def _log(self, message: str, run_time: float = 0.0, accuracy: Optional[float] = None, note: Optional[str] = None):
        """Append a log entry to the CSV log file."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, 'a') as file:
            # file.write(f"{now},{message},{run_time:.4f},{'' if accuracy is None else accuracy:.4f},{note or ''}\n")
            file.write(f"{now},{message},{run_time:.4f},{'' if accuracy is None else accuracy},{note or ''}\n")
        print(f"[LOG] {message}")

    def handle_missing(self, cols: Union[str, List[str]], strategy: str = 'mean', fill_value=None, note: Optional[str] = None):
        """Impute missing values in specified columns."""
        print(self.X.isna().sum())
        start = time.time()
        cols = [cols] if isinstance(cols, str) else cols
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        self.X[cols] = imputer.fit_transform(self.X[cols])
        self._log("handle_missing", time.time() - start, note=note)
        return self

    def handle_outliers(self, method: str = 'zscore', threshold: float = 3.0,
                        cols: Union[str, List[str]] = None, note: Optional[str] = None):
        """Handle outliers in specified columns using z-score or IQR."""
        start = time.time()
        cols = [cols] if isinstance(cols, str) else cols
        if cols is None:
            cols = self.X.select_dtypes(include=[np.number]).columns.tolist()

        if method == 'zscore':
            for col in cols:
                z = (self.X[col] - self.X[col].mean()) / self.X[col].std()
                self.X[col] = self.X[col].mask(np.abs(z) > threshold)
        elif method == 'iqr':
            for col in cols:
                Q1 = self.X[col].quantile(0.25)
                Q3 = self.X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                self.X[col] = self.X[col].mask((self.X[col] < lower) | (self.X[col] > upper))
        else:
            raise ValueError("Invalid method. Choose 'zscore' or 'iqr'.")

        self._log("handle_outliers", time.time() - start, note=note)
        return self

    def scale_features(self, method: str = 'standard', cols: Union[str, List[str]] = None, note: Optional[str] = None):
        """Scale numerical features using specified scaling method."""
        start = time.time()
        cols = [cols] if isinstance(cols, str) else cols
        if cols is None:
            cols = self.X.select_dtypes(include=[np.number]).columns.tolist()

        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self.X[cols] = scaler.fit_transform(self.X[cols])
        self._log("scale_features", time.time() - start, note=note)
        return self

    def remove_duplicates(self, note: Optional[str] = None):
        """Remove duplicate rows from the dataset."""
        start = time.time()
        combined = pd.concat([self.X, self.y], axis=1)
        combined = combined.drop_duplicates()
        self.X = combined.drop(columns=[self.label])
        self.y = combined[self.label]
        self._log("remove_duplicates", time.time() - start, note=note)
        return self

    def drop_low_variance(self, threshold: float = 0.0, note: Optional[str] = None):
        """Remove features with variance below the specified threshold."""
        start = time.time()
        selector = VarianceThreshold(threshold=threshold)
        self.X = pd.DataFrame(selector.fit_transform(self.X), columns=self.X.columns[selector.get_support()])
        self._log("drop_low_variance", time.time() - start, note=note)
        return self

    def train_model(self, model=None, test_size: float = 0.2, random_state: int = 42, note: Optional[str] = None):
        """Train and evaluate the regression model."""
        start = time.time()
        if model is None:
            model = LinearRegression()
        self.model = model
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        print(X_train.isna().sum())

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        self._log("train_model", time.time() - start, accuracy=mse, note=note)
        print(f"Model trained. MSE: {mse:.4f}")
        return self
