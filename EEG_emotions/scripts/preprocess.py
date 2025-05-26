# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """
    Load EEG emotion dataset from a CSV file.
    """
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """
    Preprocess the EEG data: separate features and labels, standardize features.
    """
    X = df.drop(columns=['label'])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
