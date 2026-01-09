"""
Data Validation Module for Telco Customer Churn
Defines schema and validation functions for input data
"""

import pandas as pd
from typing import Dict, List, Any, Optional

# Schema definition for Telco Customer Churn dataset
SCHEMA = {
    'columns': [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ],
    'types': {
        'customerID': 'object',
        'gender': 'object',
        'SeniorCitizen': 'int64',
        'Partner': 'object',
        'Dependents': 'object',
        'tenure': 'int64',
        'PhoneService': 'object',
        'MultipleLines': 'object',
        'InternetService': 'object',
        'OnlineSecurity': 'object',
        'OnlineBackup': 'object',
        'DeviceProtection': 'object',
        'TechSupport': 'object',
        'StreamingTV': 'object',
        'StreamingMovies': 'object',
        'Contract': 'object',
        'PaperlessBilling': 'object',
        'PaymentMethod': 'object',
        'MonthlyCharges': 'float64',
        'TotalCharges': 'object',  # Initially object due to whitespace issues
        'Churn': 'object'
    },
    'valid_values': {
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['Yes', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': [
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ],
        'Churn': ['Yes', 'No']
    },
    'numeric_ranges': {
        'tenure': (0, 100),
        'MonthlyCharges': (0, 200),
        'TotalCharges': (0, 10000)
    }
}


class DataValidator:
    """Validates data against defined schema"""
    
    def __init__(self, schema: Dict = None):
        self.schema = schema or SCHEMA
        self.validation_errors = []
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Check if all required columns are present"""
        missing_cols = set(self.schema['columns']) - set(df.columns)
        if missing_cols:
            self.validation_errors.append(f"Missing columns: {missing_cols}")
            return False
        return True
    
    def validate_categorical_values(self, df: pd.DataFrame) -> bool:
        """Check if categorical columns have valid values"""
        is_valid = True
        for col, valid_values in self.schema['valid_values'].items():
            if col in df.columns:
                invalid_values = set(df[col].dropna().unique()) - set(valid_values)
                if invalid_values:
                    self.validation_errors.append(
                        f"Invalid values in '{col}': {invalid_values}"
                    )
                    is_valid = False
        return is_valid
    
    def validate_numeric_ranges(self, df: pd.DataFrame) -> bool:
        """Check if numeric columns are within expected ranges"""
        is_valid = True
        for col, (min_val, max_val) in self.schema['numeric_ranges'].items():
            if col in df.columns:
                # Convert to numeric, handling errors
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                out_of_range = (numeric_col < min_val) | (numeric_col > max_val)
                if out_of_range.any():
                    count = out_of_range.sum()
                    self.validation_errors.append(
                        f"Column '{col}' has {count} values outside range [{min_val}, {max_val}]"
                    )
                    is_valid = False
        return is_valid
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validations and return results"""
        self.validation_errors = []
        
        results = {
            'columns_valid': self.validate_columns(df),
            'categories_valid': self.validate_categorical_values(df),
            'ranges_valid': self.validate_numeric_ranges(df),
            'errors': self.validation_errors
        }
        
        results['is_valid'] = all([
            results['columns_valid'],
            results['categories_valid'],
            results['ranges_valid']
        ])
        
        return results


def validate_input_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function for validation"""
    validator = DataValidator()
    return validator.validate(df)


if __name__ == "__main__":
    # Test validation with sample data
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    results = validate_input_data(df)
    print(f"Validation Results: {results}")
