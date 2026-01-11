"""
Streamlit Frontend for Telco Customer Churn Prediction
Standalone version for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #495057;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Load models (cached for performance)
@st.cache_resource
def load_models():
    """Load all models and preprocessor"""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    
    models = {}
    preprocessor = None
    
    try:
        # Load preprocessor
        preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        
        # Load models
        model_files = {
            "Neural Network": "best_model.pkl",
            "Logistic Regression": "logistic_regression.pkl",
            "Random Forest": "random_forest.pkl",
            "XGBoost": "xgboost.pkl"
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models, preprocessor

# Feature engineering functions
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

def preprocess_customer(customer_data, preprocessor):
    """Preprocess customer data for prediction"""
    df = pd.DataFrame([customer_data])
    
    # Add dummy columns for preprocessing
    df['customerID'] = 'CLOUD_USER'
    df['Churn'] = 'No'
    
    # Convert TotalCharges to ensure numeric type
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Feature Engineering: Add Tenure_Group
    df['Tenure_Group'] = df['tenure'].apply(get_tenure_group)
    
    # Feature Engineering: Add Number_of_Services
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Number_of_Services'] = df[service_cols].apply(
        lambda row: sum(row == 'Yes'), axis=1
    )
    
    # Label Encoding for binary columns
    binary_mappings = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # Debug: Print processed data
    print(f"DEBUG - tenure: {df['tenure'].values[0]}, Tenure_Group: {df['Tenure_Group'].values[0]}")
    print(f"DEBUG - Number_of_Services: {df['Number_of_Services'].values[0]}")
    print(f"DEBUG - Contract: {df['Contract'].values[0]}, PaymentMethod: {df['PaymentMethod'].values[0]}")
    
    # Apply preprocessor
    X = preprocessor.transform(df)
    
    return X

# Load models
models, preprocessor = load_models()

# Header
st.markdown('<div class="main-header">📊 Telco Customer Churn Predictor</div>', unsafe_allow_html=True)

# Show loaded models
if models:
    st.success(f"✅ Models Loaded: {', '.join(models.keys())}")
else:
    st.error("❌ No models found. Please ensure model files are in the models/ directory.")

st.markdown("---")

# Model selection in sidebar
with st.sidebar:
    st.markdown("## 🎯 Model Selection")
    available_models = list(models.keys()) if models else ["No models available"]
    selected_model = st.selectbox(
        "Choose Model:",
        available_models,
        index=0,
        help="Select which trained model to use for prediction"
    )
    st.info(f"Currently using: **{selected_model}**")
    
    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Telco Customer Churn Prediction**
    
    Predict whether a telecom customer is likely to churn.
    
    **Author:** Ngo Anh Hieu  
    **GitHub:** [Toogoodtoac](https://github.com/Toogoodtoac/CustumerChurning)
    """)

# Use st.form to prevent reloading on every input change
with st.form("prediction_form"):
    st.markdown("### 📝 Enter Customer Information")
    
    # Create columns for input
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Demographics
    with col1:
        st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
        
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        senior_citizen = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Partner", ["Yes", "No"], horizontal=True)
        dependents = st.radio("Dependents", ["Yes", "No"], horizontal=True)
    
    # Column 2: Services
    with col2:
        st.markdown('<div class="section-header">📱 Services</div>', unsafe_allow_html=True)
        
        phone_service = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
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
        total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 840.0, 10.0)
    
    st.markdown("---")
    
    # Submit button inside form
    submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

# Process form submission
if submitted and models and preprocessor:
    # Convert senior citizen
    senior_citizen_val = 1 if senior_citizen == "Yes" else 0
    
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
        "TotalCharges": total_charges
    }
    
    try:
        with st.spinner("🔄 Making prediction..."):
            # Preprocess
            X = preprocess_customer(customer_data, preprocessor)
            
            # Get selected model
            model = models[selected_model]
            
            # Predict
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk = "Low"
            recommendation = "Customer is stable. Continue current service."
        elif probability < 0.6:
            risk = "Medium"
            recommendation = "Consider proactive engagement. Offer loyalty discount."
        else:
            risk = "High"
            recommendation = "⚠️ Urgent attention needed! Offer 20% discount and free service upgrade."
        
        # Display result
        st.markdown("---")
        st.markdown("## 📋 Prediction Result")
        
        # Create result display
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        
        with col_r1:
            if prediction == 1:
                st.metric("Churn Prediction", "🚨 YES")
            else:
                st.metric("Churn Prediction", "✅ NO")
        
        with col_r2:
            prob_pct = probability * 100
            st.metric("Churn Probability", f"{prob_pct:.1f}%")
        
        with col_r3:
            if risk == "Low":
                st.metric("Risk Level", "🟢 Low")
            elif risk == "Medium":
                st.metric("Risk Level", "🟡 Medium")
            else:
                st.metric("Risk Level", "🔴 High")
        
        with col_r4:
            st.metric("Model Used", selected_model)
        
        # Recommendation box
        st.markdown("### 💡 Recommendation")
        if risk == "High":
            st.error(recommendation)
        elif risk == "Medium":
            st.warning(recommendation)
        else:
            st.success(recommendation)
        
        # Show customer profile summary
        with st.expander("📊 Customer Profile Summary"):
            profile_df = pd.DataFrame([customer_data]).T
            profile_df.columns = ['Value']
            st.dataframe(profile_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    📊 Telco Customer Churn Prediction | Author: Ngo Anh Hieu | 
    <a href="https://github.com/Toogoodtoac/CustumerChurning">GitHub</a>
</div>
""", unsafe_allow_html=True)
