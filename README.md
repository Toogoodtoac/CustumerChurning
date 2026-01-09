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
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Training (if models not present)
```bash
python src/train_models.py
```

### 5. Start the Application

**Terminal 1 - API Server:**
```bash
uvicorn app.api:app --reload --port 8000
```

**Terminal 2 - Web Interface:**
```bash
streamlit run app/streamlit_app.py
```

### 6. Access the App
- **Web App:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

---

## 📁 Project Structure

```
CustumerChurning/
├── app/
│   ├── api.py              # FastAPI backend
│   └── streamlit_app.py    # Streamlit frontend
├── src/
│   ├── preprocessing.py    # Data preprocessing
│   ├── train_models.py     # Model training script
│   └── data_validation.py  # Data validation
├── models/                 # Trained models & visualizations
├── data/                   # Dataset
├── notebooks/              # Jupyter notebooks
└── requirements.txt        # Dependencies
```

---

## 🎯 Features

- **4 ML Models:** Logistic Regression, Random Forest, XGBoost, Neural Network
- **Model Selection:** Choose which model to use in the web app
- **Interactive UI:** Streamlit interface with sliders, dropdowns
- **REST API:** FastAPI with automatic documentation
- **Risk Assessment:** Low/Medium/High risk classification
- **Business Recommendations:** Actionable retention strategies

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

- Python 3.8+
- FastAPI
- Streamlit
- Scikit-learn
- XGBoost
- TensorFlow/Keras
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 📝 License

MIT License

---

## 👤 Author

**Toogoodtoac**
- GitHub: [@Toogoodtoac](https://github.com/Toogoodtoac)
