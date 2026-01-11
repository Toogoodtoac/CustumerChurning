"""
FastAPI Backend for Telco Customer Churn Prediction
Provides REST API endpoint for churn prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd
import numpy as np
import joblib
import os

# Initialize FastAPI app
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predict customer churn probability using ML models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

# Available models mapping
MODEL_FILES = {
    "Neural Network": "best_model.pkl",  # or neural_network if needed
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# Global variables for loaded models
models = {}
preprocessor = None


class CustomerProfile(BaseModel):
    """Pydantic model for customer data validation"""
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0, le=200)
    TotalCharges: float = Field(ge=0, le=10000)
    model_name: Optional[str] = "Neural Network"  # Model selection

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 79.85,
                "TotalCharges": 958.2,
                "model_name": "Neural Network"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    churn_prediction: str
    churn_probability: float
    risk_level: str
    recommendation: str
    model_used: str


def load_models():
    """Load all ML models and preprocessor"""
    global models, preprocessor
    
    try:
        # Load preprocessor
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("✅ Preprocessor loaded")
        
        # Load all available models
        for model_name, filename in MODEL_FILES.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                print(f"✅ {model_name} loaded")
            else:
                print(f"⚠️ {model_name} not found: {filename}")
        
        print(f"✅ Total models loaded: {len(models)}")
        
    except FileNotFoundError as e:
        print(f"⚠️ Model files not found: {e}")
        print("   Run the training notebook first to generate models.")


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Telco Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": len(models) > 0,
        "models_available": list(models.keys()),
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerProfile):
    """
    Predict customer churn probability
    
    - **customer**: Customer profile with all required fields
    - **returns**: Churn prediction, probability, risk level, and recommendation
    """
    if len(models) == 0 or preprocessor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please run training notebook first."
        )
    
    # Get selected model
    model_name = customer.model_name or "Neural Network"
    if model_name not in models:
        # Fallback to first available model
        model_name = list(models.keys())[0]
    
    selected_model = models[model_name]
    
    try:
        # Convert to DataFrame
        customer_dict = customer.model_dump()
        # Remove model_name from data (it's not a feature)
        customer_dict.pop('model_name', None)
        customer_dict['customerID'] = 'API_USER'  # Dummy ID
        customer_dict['Churn'] = 'No'  # Placeholder for preprocessing
        
        df = pd.DataFrame([customer_dict])
        
        # Feature Engineering: Add Tenure_Group
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
        
        # Feature Engineering: Add Number_of_Services
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['Number_of_Services'] = df[service_cols].apply(
            lambda row: sum(row == 'Yes'), axis=1
        )
        
        # Label Encoding for binary columns (same as training)
        binary_mappings = {
            'gender': {'Female': 0, 'Male': 1},
            'Partner': {'No': 0, 'Yes': 1},
            'Dependents': {'No': 0, 'Yes': 1},
            'PhoneService': {'No': 0, 'Yes': 1},
            'PaperlessBilling': {'No': 0, 'Yes': 1}
        }
        
        for col, mapping in binary_mappings.items():
            df[col] = df[col].map(mapping)
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict using selected model
        prediction = selected_model.predict(X)[0]
        probability = selected_model.predict_proba(X)[0][1]
        
        # Determine risk level and recommendation
        if probability < 0.3:
            risk_level = "Low"
            recommendation = "Customer is stable. Continue current service."
        elif probability < 0.6:
            risk_level = "Medium"
            recommendation = "Consider proactive engagement. Offer loyalty discount."
        else:
            risk_level = "High"
            recommendation = "⚠️ Urgent attention needed! Offer 20% discount and free service upgrade."
        
        return PredictionResponse(
            churn_prediction="Yes" if prediction == 1 else "No",
            churn_probability=round(float(probability), 4),
            risk_level=risk_level,
            recommendation=recommendation,
            model_used=model_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get information about the loaded models"""
    if len(models) == 0:
        return {"message": "No models loaded"}
    
    return {
        "models_loaded": list(models.keys()),
        "preprocessor_loaded": preprocessor is not None,
        "total_models": len(models)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
