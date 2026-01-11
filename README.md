# 🔮 Telco Customer Churn Prediction

A complete Machine Learning pipeline for predicting customer churn in the telecommunications industry.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Toogoodtoac/CustumerChurning.git
cd CustumerChurning
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell)
.\venv\Scripts\activate

# Windows (CMD)
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Training (Optional - only if models not present)
```bash
python src/train_models.py
```

### 5. Start the Application

**Terminal 1 - Start API Server:**
```bash
uvicorn app.api:app --reload --port 8000
```

**Terminal 2 - Start Web Interface:**
```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
CustumerChurning/
├── app/
│   ├── api.py                  # FastAPI backend with model selection
│   └── streamlit_app.py        # Streamlit web interface
│
├── src/
│   ├── __init__.py             # Package init
│   ├── preprocessing.py        # Data preprocessing & feature engineering
│   ├── train_models.py         # Model training script
│   └── data_validation.py      # Data validation with schema
│
├── models/
│   ├── best_model.pkl          # Best performing model (Neural Network)
│   ├── logistic_regression.pkl # Logistic Regression model
│   ├── random_forest.pkl       # Random Forest model
│   ├── xgboost.pkl             # XGBoost model
│   ├── neural_network.h5       # Neural Network (Keras)
│   ├── preprocessor.pkl        # Preprocessing pipeline
│   └── *.png                   # Visualization charts
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
│
├── notebooks/                  # Jupyter notebooks for EDA
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 🎯 Features

### Machine Learning
- **4 ML Models:** Logistic Regression, Random Forest, XGBoost, Neural Network
- **Feature Engineering:** Tenure Groups, Number of Services
- **Class Balancing:** SMOTE (applied correctly after train/test split)

### Web Application
- **Model Selection:** Choose between 4 trained models
- **Interactive UI:** Sliders, radio buttons, dropdowns for all customer fields
- **Color-coded Results:** Green (Low risk), Yellow (Medium risk), Red (High risk)
- **Business Recommendations:** Actionable retention strategies

### API
- **REST API:** FastAPI with automatic Swagger documentation
- **Endpoints:** `/predict`, `/health`, `/model-info`
- **Model Selection:** Pass `model_name` parameter to choose model

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Neural Network | 77.9% | 56.8% | 69.0% | **62.3%** | 84.1% |
| Logistic Regression | 74.3% | 51.0% | **80.0%** | 62.3% | 84.1% |
| Random Forest | 76.8% | 55.2% | 66.8% | 60.5% | 84.0% |
| XGBoost | 78.1% | 58.5% | 60.7% | 59.6% | 83.4% |

---

## 📈 Dataset

- **Source:** IBM Telco Customer Churn
- **Samples:** 7,043 customers
- **Features:** 21 columns
- **Target:** Churn (Yes/No)
- **Class Imbalance:** 26.5% churners

---

## 🛠️ Technologies

| Category | Technologies |
|----------|-------------|
| **ML/DL** | Scikit-learn, XGBoost, TensorFlow/Keras |
| **Data** | Pandas, NumPy, imbalanced-learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |

---

## 📋 Requirements

See `requirements.txt` for full list. Main dependencies:
- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, tensorflow
- fastapi, uvicorn
- streamlit
- imbalanced-learn

---

## 👤 Author

**Toogoodtoac**
- GitHub: [@Toogoodtoac](https://github.com/Toogoodtoac)
