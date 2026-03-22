import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

BASE_DIR = r'C:\Users\Pritam\Desktop\heart-disease'

def load_data(path=os.path.join(BASE_DIR, 'data', 'heart.csv')):
    df = pd.read_csv(path)
    df.rename(columns={'condition': 'target'}, inplace=True)
    return df

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def handle_outliers(df, columns=['trestbps', 'chol', 'thalach', 'oldpeak']):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df

def feature_engineering(df):
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 40, 50, 60, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    df['high_chol'] = (df['chol'] >= 240).astype(int)
    df['high_bp']   = (df['trestbps'] >= 140).astype(int)
    df['hr_ratio']  = (df['thalach'] / (220 - df['age'])).round(3)
    return df

def split_features_target(df, target_col='target'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_and_encode_features(X_train, X_test):
    # Automatically identify numeric and categorical features
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Use ColumnTransformer to scale numeric features and pass through others
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', 'passthrough', categorical_features)
        ]
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)
    
    feature_names = numeric_features + categorical_features
    return X_train_processed, X_test_processed, preprocessor, feature_names

def save_processed_data(X_train, X_test, y_train, y_test, preprocessor, feature_names):
    results_path = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_path, exist_ok=True)

    np.save(os.path.join(results_path, 'X_train.npy'), X_train)
    np.save(os.path.join(results_path, 'X_test.npy'), X_test)
    np.save(os.path.join(results_path, 'y_train.npy'), y_train.values)
    np.save(os.path.join(results_path, 'y_test.npy'), y_test.values)

    pd.Series(feature_names).to_csv(os.path.join(results_path, 'feature_names.csv'), index=False)

    with open(os.path.join(results_path, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)

def preprocess_pipeline(path=os.path.join(BASE_DIR, 'data', 'heart.csv')):
    df = load_data(path)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = feature_engineering(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_processed, X_test_processed, preprocessor, feature_names = scale_and_encode_features(X_train, X_test)
    save_processed_data(X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    preprocess_pipeline()