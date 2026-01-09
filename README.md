# Telco Customer Churn Prediction

A complete Machine Learning pipeline for predicting customer churn in telecom industry.

## 🚀 Quick Start

### 1. Activate Virtual Environment
```powershell
cd d:\CustomerChurning
.\venv\Scripts\activate
```

### 2. Run Training Notebook
```powershell
jupyter notebook notebooks/EDA_and_Training.ipynb
```
Run all cells to:
- Perform EDA
- Train models (Logistic Regression, Random Forest, XGBoost, Neural Network)
- Save trained models to `models/` directory

### 3. Start API Server
```powershell
cd d:\CustomerChurning
.\venv\Scripts\activate
uvicorn app.api:app --reload --port 8000
```
API available at: http://localhost:8000
- Docs: http://localhost:8000/docs

### 4. Start Streamlit App (in new terminal)
```powershell
cd d:\CustomerChurning
.\venv\Scripts\activate
streamlit run app/streamlit_app.py
```
App available at: http://localhost:8501

## 📁 Project Structure

```
d:\CustomerChurning\
├── venv/                          # Virtual environment
├── notebooks/
│   └── EDA_and_Training.ipynb     # Jupyter notebook - EDA & Training
├── src/
│   ├── __init__.py
│   ├── data_validation.py         # Schema validation
│   └── preprocessing.py           # Feature engineering pipeline
├── app/
│   ├── api.py                     # FastAPI backend
│   └── streamlit_app.py           # Streamlit frontend
├── models/                        # Saved models (after training)
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── neural_network.h5
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── requirements.txt
```

## 🎯 Features

- **EDA**: Comprehensive data exploration with visualizations
- **Feature Engineering**: Tenure groups, service count
- **SMOTE**: Class imbalance handling
- **4 Models**: Logistic Regression, Random Forest, XGBoost, Neural Network
- **REST API**: FastAPI with Pydantic validation
- **Web UI**: Interactive Streamlit interface

## 📊 Dataset

Telco Customer Churn dataset from IBM Sample Data Sets
- 7043 customers
- 21 features
- Binary classification: Churn (Yes/No)

## 🔧 Dependencies

See `requirements.txt` for full list.
Key packages: pandas, scikit-learn, xgboost, tensorflow, fastapi, streamlit
