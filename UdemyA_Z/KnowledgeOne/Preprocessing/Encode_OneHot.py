from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def handle_nan(df):
    none_counts = df.isnull().sum()

    import pandas as pd
    import numpy as np

    data = {
        'column1': [1, 2, None, 4, np.nan],
        'column2': [5, np.nan, 7, 8, None],
        'column3': ['a', 'b', 'c', None, 'd']
    }
    df = pd.DataFrame(data)

    # Assuming you have a DataFrame named 'df'
    column_with_none = [col for col in df.select_dtypes(include=['number']).columns if df[col].isnull().any()]

    print(column_with_none)

def encode_category(df, columns=None):
    """
    Encode categorical columns
    :param df:
    :param columns:
    :return:
    """
    if columns is None:
        columns = df.select_dtypes(include='category').columns.tolist()

    col_transformer = ColumnTransformer(transformers=[('encode', OneHotEncoder(), columns)], remainder='passthrough')
    df_en = col_transformer.fit_transform(df)

    return df_en, r"""
        def encode_category(df_X, df_y, columns) -> Any:
            col_transformer = ColumnTransformer(transformers=[('encode', OneHotEncoder(), columns)], remainder='passthrough')
            df_X_transformed = col_transformer.fit_transform(df_X)
            df_y_transformed = col_transformer.fit_transform(df_y)
            return df_X_transformed, df_y_transformed, col_transformer, 
    """

def scale(df_train, df_test=None, columns=None):
    standard_scaler = StandardScaler()
    return standard_scaler.fit_transform(df_train), standard_scaler.fit_transform(df_test) if df_test else None
