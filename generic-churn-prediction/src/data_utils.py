import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, target='churn'):

    # Separate the target variable
    X = data.drop(columns=[target])
    y = data[target]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Define preprocessing for numerical columns (impute missing values and scale)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical columns (impute missing and one-hot encode)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing for numerical and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Fit the preprocessor and transform the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    return X_preprocessed, y

def train_test_data(file_path, target='churn', test_size=0.2):

    data = load_data(file_path)
    X, y = preprocess_data(data, target=target)
    return train_test_split(X, y, test_size=test_size, random_state=42)

