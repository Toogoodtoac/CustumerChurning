"""
Preprocessing Module for Telco Customer Churn
Implements feature engineering, encoding, scaling, and SMOTE
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib


class ChurnPreprocessor:
    """
    Preprocessing pipeline for Telco Customer Churn
    Includes: cleaning, feature engineering, encoding, scaling, SMOTE
    """
    
    # Service columns for Number_of_Services feature
    SERVICE_COLUMNS = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Columns to drop (identifiers)
    DROP_COLUMNS = ['customerID']
    
    # Binary columns for LabelEncoding
    BINARY_COLUMNS = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'
    ]
    
    # Multi-value columns for OneHotEncoding
    MULTI_VALUE_COLUMNS = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    
    # Numeric columns for scaling
    NUMERIC_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    def __init__(self):
        self.label_encoders = {}
        self.onehot_encoder = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data:
        1. Convert TotalCharges to numeric (handle whitespace)
        2. Fill missing TotalCharges with 0 (new customers)
        """
        df = df.copy()
        
        # Convert TotalCharges to numeric, coercing errors to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill NaN with 0 (for new customers with tenure=0)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        return df
    
    def create_tenure_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tenure groups for better pattern recognition"""
        df = df.copy()
        
        def get_tenure_group(tenure):
            if tenure <= 12:
                return '0-12'
            elif tenure <= 24:
                return '12-24'
            elif tenure <= 48:
                return '24-48'
            elif tenure <= 72:
                return '48-72'
            else:
                return '72+'
        
        df['Tenure_Group'] = df['tenure'].apply(get_tenure_group)
        
        return df
    
    def create_number_of_services(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count total number of additional services used"""
        df = df.copy()
        
        def count_services(row):
            count = 0
            for col in self.SERVICE_COLUMNS:
                if row[col] == 'Yes':
                    count += 1
            return count
        
        df['Number_of_Services'] = df.apply(count_services, axis=1)
        
        return df
    
    def encode_binary_columns(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode binary columns using LabelEncoder"""
        df = df.copy()
        
        binary_cols = [c for c in self.BINARY_COLUMNS if c in df.columns and c != 'Churn']
        
        for col in binary_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def encode_target(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """Encode target variable (Churn)"""
        if fit:
            self.label_encoders['Churn'] = LabelEncoder()
            return self.label_encoders['Churn'].fit_transform(y)
        else:
            return self.label_encoders['Churn'].transform(y)
    
    def fit_transform(self, df: pd.DataFrame, apply_smote: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the data
        Returns: X (features), y (target)
        """
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.create_tenure_groups(df)
        df = self.create_number_of_services(df)
        
        # Separate target
        y = df['Churn'].copy()
        X = df.drop(columns=['Churn'] + self.DROP_COLUMNS, errors='ignore')
        
        # Encode target
        y_encoded = self.encode_target(y, fit=True)
        
        # Encode binary columns
        X = self.encode_binary_columns(X, fit=True)
        
        # Get columns for one-hot encoding (only those that exist)
        onehot_cols = [c for c in self.MULTI_VALUE_COLUMNS if c in X.columns]
        onehot_cols.append('Tenure_Group')  # Add engineered feature
        
        # Get remaining numeric columns
        numeric_cols = [c for c in X.columns if c not in onehot_cols]
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), [c for c in numeric_cols if c in X.columns]),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 [c for c in onehot_cols if c in X.columns])
            ],
            remainder='drop'
        )
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        try:
            num_features = [c for c in numeric_cols if c in X.columns]
            cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(
                [c for c in onehot_cols if c in X.columns]
            ).tolist()
            self.feature_names = num_features + cat_features
        except:
            self.feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        # Apply SMOTE if requested
        if apply_smote:
            smote = SMOTE(random_state=42)
            X_processed, y_encoded = smote.fit_resample(X_processed, y_encoded)
            print(f"After SMOTE: {np.bincount(y_encoded)}")
        
        self.is_fitted = True
        
        return X_processed, y_encoded
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.create_tenure_groups(df)
        df = self.create_number_of_services(df)
        
        # Remove target and ID if present
        X = df.drop(columns=['Churn'] + self.DROP_COLUMNS, errors='ignore')
        
        # Encode binary columns
        X = self.encode_binary_columns(X, fit=False)
        
        # Transform
        X_processed = self.preprocessor.transform(X)
        
        return X_processed
    
    def save(self, path: str):
        """Save the preprocessor to disk"""
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'ChurnPreprocessor':
        """Load the preprocessor from disk"""
        return joblib.load(path)


def preprocess_data(
    df: pd.DataFrame, 
    apply_smote: bool = True
) -> Tuple[np.ndarray, np.ndarray, ChurnPreprocessor]:
    """
    Convenience function to preprocess data
    Returns: X, y, preprocessor
    """
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.fit_transform(df, apply_smote=apply_smote)
    return X, y, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X, y, preprocessor = preprocess_data(df)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {len(preprocessor.feature_names)}")
