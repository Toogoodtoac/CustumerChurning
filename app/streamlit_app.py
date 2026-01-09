"""
Streamlit Frontend for Telco Customer Churn Prediction
Interactive web interface for churn prediction
"""

import streamlit as st
import requests
import json
import pandas as pd

# Page config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .low-risk {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    .medium-risk {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border: 2px solid #ffc107;
    }
    .high-risk {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #495057;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# API URL
API_URL = "http://localhost:8000"

# Header
st.markdown('<div class="main-header">📊 Telco Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    if health.get('model_loaded'):
        st.success("✅ API Connected & Model Loaded")
    else:
        st.warning("⚠️ API Connected but model not loaded. Run training notebook first.")
except:
    st.error("❌ Cannot connect to API. Start the API with: `uvicorn app.api:app --reload --port 8000`")
    st.info("📝 You can still fill in the form below. The prediction will work once API is running.")

# Model selection in sidebar
with st.sidebar:
    st.markdown("## 🎯 Model Selection")
    selected_model = st.selectbox(
        "Choose Model:",
        ["Neural Network", "Logistic Regression", "Random Forest", "XGBoost"],
        index=0,
        help="Select which trained model to use for prediction"
    )
    st.info(f"Currently using: **{selected_model}**")

# Create columns for input
col1, col2, col3 = st.columns(3)

# Column 1: Demographics
with col1:
    st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
    
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    senior_citizen = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
    senior_citizen_val = 1 if senior_citizen == "Yes" else 0
    partner = st.radio("Partner", ["Yes", "No"], horizontal=True)
    dependents = st.radio("Dependents", ["Yes", "No"], horizontal=True)

# Column 2: Services
with col2:
    st.markdown('<div class="section-header">📱 Services</div>', unsafe_allow_html=True)
    
    phone_service = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
    
    if phone_service == "Yes":
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    else:
        multiple_lines = "No phone service"
    
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    
    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

# Column 3: Account
with col3:
    st.markdown('<div class="section-header">💼 Account Info</div>', unsafe_allow_html=True)
    
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0, 0.5)
    total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges), 10.0)

st.markdown("---")

# Prediction Button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

if predict_button:
    # Prepare data
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen_val,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "model_name": selected_model
    }
    
    try:
        # Call API
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display result
            st.markdown("---")
            st.markdown("## 📋 Prediction Result")
            
            # Determine style based on risk
            risk = result['risk_level']
            if risk == "Low":
                box_class = "low-risk"
                emoji = "✅"
            elif risk == "Medium":
                box_class = "medium-risk"
                emoji = "⚠️"
            else:
                box_class = "high-risk"
                emoji = "🚨"
            
            # Create result display
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric(
                    "Churn Prediction",
                    f"{emoji} {result['churn_prediction']}",
                    delta=None
                )
            
            with col_r2:
                prob_pct = result['churn_probability'] * 100
                st.metric(
                    "Churn Probability",
                    f"{prob_pct:.1f}%",
                    delta=None
                )
            
            with col_r3:
                st.metric(
                    "Risk Level",
                    f"{result['risk_level']}",
                    delta=None
                )
            
            # Recommendation box
            st.markdown("### 💡 Recommendation")
            if risk == "High":
                st.error(result['recommendation'])
            elif risk == "Medium":
                st.warning(result['recommendation'])
            else:
                st.success(result['recommendation'])
            
            # Show customer profile summary
            with st.expander("📊 Customer Profile Summary"):
                profile_df = pd.DataFrame([customer_data]).T
                profile_df.columns = ['Value']
                st.dataframe(profile_df, use_container_width=True)
                
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please ensure the API server is running.")
        st.code("cd d:\\CustomerChurning && .\\venv\\Scripts\\activate && uvicorn app.api:app --reload --port 8000", language="powershell")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    📊 Telco Customer Churn Prediction System | Built with Streamlit & FastAPI<br>
    🎯 Using Machine Learning to retain customers
</div>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    This application predicts whether a telecom customer is likely to churn (leave the service).
    
    **Features used:**
    - Demographics (gender, senior citizen, partner, dependents)
    - Services (phone, internet, streaming, security)
    - Account info (tenure, contract, billing, charges)
    
    **Models available:**
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Neural Network
    
    **API Endpoints:**
    - `GET /` - API info
    - `GET /health` - Health check
    - `POST /predict` - Make prediction
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.code("""
# Start API
uvicorn app.api:app --port 8000

# Start Streamlit (in another terminal)
streamlit run app/streamlit_app.py
    """, language="bash")
