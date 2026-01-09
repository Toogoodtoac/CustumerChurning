# 📊 BÁO CÁO DỰ ÁN: DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ DỊCH VỤ VIỄN THÔNG
## Telco Customer Churn Prediction - Machine Learning Pipeline

**Tác giả:** AI-Powered ML Pipeline  
**Ngày:** Tháng 1, 2026

---

## 1. GIỚI THIỆU BÀI TOÁN

### 1.1 Bối cảnh
Trong ngành viễn thông, việc giữ chân khách hàng là yếu tố sống còn. Chi phí để có được một khách hàng mới cao gấp **5-7 lần** so với việc giữ chân khách hàng hiện tại. Do đó, việc dự đoán sớm những khách hàng có khả năng rời bỏ (churn) là vô cùng quan trọng.

### 1.2 Mục tiêu
- Xây dựng mô hình Machine Learning dự đoán khách hàng rời bỏ
- So sánh hiệu năng các mô hình ML và Deep Learning
- Triển khai API và giao diện web cho người dùng

### 1.3 Bộ dữ liệu
**Nguồn:** IBM Sample Data Sets - Telco Customer Churn

| Thông tin | Giá trị |
|-----------|---------|
| Số lượng mẫu | 7,043 khách hàng |
| Số lượng đặc trưng | 21 cột |
| Biến mục tiêu | Churn (Yes/No) |

---

## 2. PHÂN TÍCH VÀ KHÁM PHÁ DỮ LIỆU (EDA)

### 2.1 Phân bố biến mục tiêu

| Churn | Số lượng | Tỷ lệ |
|-------|----------|-------|
| No (Ở lại) | 5,174 | 73.5% |
| Yes (Rời bỏ) | 1,869 | 26.5% |

**⚠️ Vấn đề:** Dữ liệu mất cân bằng (Imbalanced) với tỷ lệ ~2.8:1

![Churn Distribution](file:///d:/CustomerChurning/models/churn_distribution.png)

### 2.2 Xử lý dữ liệu bẩn

**Vấn đề TotalCharges:**
- Cột TotalCharges có kiểu `object` thay vì `numeric`
- 11 dòng chứa khoảng trắng (khách hàng mới với tenure=0)

**Giải pháp:**
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
```

### 2.3 Phân tích đa biến - Key Insights

![Churn by Categories](file:///d:/CustomerChurning/models/churn_by_categories.png)

| Yếu tố | Insight |
|--------|---------|
| **Contract** | Month-to-month: ~43% churn (CAO NHẤT!) |
| **Internet Service** | Fiber optic: ~42% churn |
| **Payment Method** | Electronic check: ~45% churn |
| **Gender** | Không có sự khác biệt đáng kể |

![Numeric by Churn](file:///d:/CustomerChurning/models/numeric_by_churn.png)

---

## 3. TIỀN XỬ LÝ VÀ KỸ THUẬT ĐẶC TRƯNG

### 3.1 Data Validation Schema

```python
SCHEMA = {
    'valid_values': {
        'gender': ['Male', 'Female'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        ...
    },
    'numeric_ranges': {
        'tenure': (0, 100),
        'MonthlyCharges': (0, 200),
        'TotalCharges': (0, 10000)
    }
}
```

### 3.2 Feature Engineering

**1. Tenure_Group** - Nhóm thời gian sử dụng:
| Nhóm | Tenure | Churn Rate |
|------|--------|------------|
| 0-12 | Khách mới | ~47% (CAO NHẤT) |
| 12-24 | 1-2 năm | ~28% |
| 24-48 | 2-4 năm | ~19% |
| 48-72 | 4-6 năm | ~13% |
| 72+ | >6 năm | ~7% (THẤP NHẤT) |

**2. Number_of_Services** - Số lượng dịch vụ:
- Đếm tổng các dịch vụ: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Insight:** Càng nhiều dịch vụ → Càng ít churn (hiệu ứng lock-in)

![Engineered Features](file:///d:/CustomerChurning/models/engineered_features.png)

### 3.3 Xử lý mất cân bằng - SMOTE

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

| Trước SMOTE | Sau SMOTE |
|-------------|-----------|
| [5174, 1869] | [5174, 5174] |
| 73.5% vs 26.5% | 50% vs 50% |

### 3.4 Encoding & Scaling

| Loại biến | Phương pháp |
|-----------|-------------|
| Binary (Yes/No, Male/Female) | LabelEncoder |
| Multi-value (Contract, PaymentMethod) | OneHotEncoder (drop='first') |
| Numeric (tenure, charges) | StandardScaler |

---

## 4. HUẤN LUYỆN MÔ HÌNH

### 4.1 Các mô hình sử dụng

| # | Mô hình | Loại | Mục đích |
|---|---------|------|----------|
| 1 | Logistic Regression | ML | Baseline, giải thích hệ số |
| 2 | Random Forest | ML | Feature importance, ensemble |
| 3 | XGBoost | ML | Hiệu năng cao trên tabular data |
| 4 | Neural Network ⭐ | DL | Bonus - Deep Learning |

### 4.2 Neural Network Architecture

```python
model = Sequential([
    Dense(16, activation='relu', input_dim=n_features),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## 5. ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH

### 5.1 Kết quả

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **0.8541** | **0.8534** | **0.8551** | **0.8542** | **0.9351** |
| Random Forest | 0.8430 | 0.8158 | 0.8860 | 0.8495 | 0.9178 |
| Neural Network | 0.7884 | 0.7808 | 0.8019 | 0.7912 | 0.8707 |
| Logistic Regression | 0.7802 | 0.7544 | 0.8309 | 0.7908 | 0.8632 |

**🏆 Mô hình tốt nhất:** XGBoost với F1-Score = 0.8542

![Model Comparison](file:///d:/CustomerChurning/models/model_comparison.png)

### 5.2 Confusion Matrices

![Confusion Matrices](file:///d:/CustomerChurning/models/confusion_matrices.png)

### 5.3 Feature Importance

![Feature Importance](file:///d:/CustomerChurning/models/feature_importance.png)

**Top 5 yếu tố quan trọng nhất:**
1. TotalCharges
2. MonthlyCharges
3. tenure
4. Contract (Month-to-month)
5. Number_of_Services

---

## 6. TRIỂN KHAI SẢN PHẨM

### 6.1 FastAPI Backend

**Endpoint:** `POST /predict`

```python
class CustomerProfile(BaseModel):
    gender: Literal["Male", "Female"]
    tenure: int = Field(ge=0, le=100)
    Contract: Literal["Month-to-month", "One year", "Two year"]
    MonthlyCharges: float = Field(ge=0, le=200)
    # ... các trường khác
```

**Response:**
```json
{
    "churn_prediction": "Yes",
    "churn_probability": 0.7542,
    "risk_level": "High",
    "recommendation": "⚠️ Urgent! Offer 20% discount"
}
```

### 6.2 Streamlit Web Interface

- Giao diện tương tác với slider và selectbox
- Hiển thị kết quả với màu theo mức độ rủi ro
- Đề xuất hành động dựa trên dự đoán

---

## 7. HƯỚNG DẪN SỬ DỤNG

### 7.1 Cài đặt

```powershell
cd d:\CustomerChurning
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Huấn luyện mô hình

```powershell
python src/train_models.py
```

### 7.3 Chạy API

```powershell
uvicorn app.api:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 7.4 Chạy Web App

```powershell
streamlit run app/streamlit_app.py
# Web: http://localhost:8501
```

---

## 8. KẾT LUẬN VÀ KIẾN NGHỊ

### 8.1 Kết luận
- XGBoost đạt hiệu năng tốt nhất với F1-Score 0.8542
- SMOTE giúp cải thiện đáng kể khả năng phát hiện churn
- Feature Engineering (Tenure_Group, Number_of_Services) tăng hiệu quả mô hình

### 8.2 Kiến nghị cho doanh nghiệp

| Vấn đề | Giải pháp |
|--------|-----------|
| Month-to-month churn cao | Khuyến mãi chuyển sang 1-2 năm |
| Khách mới dễ rời bỏ | Chương trình onboarding đặc biệt |
| Electronic check churn cao | Khuyến khích thanh toán tự động |
| Fiber optic churn cao | Kiểm tra chất lượng dịch vụ |
| Ít dịch vụ = dễ rời bỏ | Gói bundle combo giảm giá |

### 8.3 Hướng phát triển

1. **Hyperparameter Tuning:** GridSearchCV hoặc Bayesian Optimization
2. **Model Ensembling:** Kết hợp nhiều mô hình
3. **Real-time Prediction:** Tích hợp với CRM
4. **A/B Testing:** So sánh hiệu quả các chiến lược retention
5. **MLOps:** CI/CD pipeline, model monitoring

---

## 📁 CẤU TRÚC THƯ MỤC

```
d:\CustomerChurning\
├── notebooks/
│   └── EDA_and_Training.ipynb
├── src/
│   ├── data_validation.py
│   ├── preprocessing.py
│   └── train_models.py
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── models/
│   ├── best_model.pkl (XGBoost)
│   ├── preprocessor.pkl
│   ├── model_results.csv
│   └── *.png (visualizations)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── requirements.txt
```

---

**📊 Báo cáo được tạo tự động bởi ML Pipeline**
