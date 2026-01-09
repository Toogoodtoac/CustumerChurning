"""
Generate PDF Report - UPDATED VERSION with Fixed Results
"""
from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(31, 119, 180)
        self.cell(0, 10, 'Telco Customer Churn - ML Report (FIXED)', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)
    
    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(52, 73, 94)
        self.cell(0, 8, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
    
    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(51, 51, 51)
        self.multi_cell(0, 5, text)
        self.ln(2)
    
    def warning_box(self, text):
        self.set_fill_color(255, 243, 205)
        self.set_draw_color(255, 193, 7)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(133, 100, 4)
        self.multi_cell(0, 6, text, border=1, fill=True)
        self.ln(3)
    
    def add_table(self, headers, data, highlight_row=None):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(52, 152, 219)
        self.set_text_color(255, 255, 255)
        
        col_width = (self.w - 20) / len(headers)
        for header in headers:
            self.cell(col_width, 8, header, border=1, align='C', fill=True)
        self.ln()
        
        self.set_font('Helvetica', '', 9)
        self.set_text_color(51, 51, 51)
        for i, row in enumerate(data):
            if highlight_row is not None and i == highlight_row:
                self.set_fill_color(212, 237, 218)
                self.set_font('Helvetica', 'B', 9)
            else:
                self.set_fill_color(255, 255, 255) if i % 2 == 0 else self.set_fill_color(249, 249, 249)
                self.set_font('Helvetica', '', 9)
            for item in row:
                self.cell(col_width, 7, str(item), border=1, align='C', fill=True)
            self.ln()
        self.ln(5)
    
    def add_image_if_exists(self, path, w=180):
        if os.path.exists(path):
            self.image(path, x=15, w=w)
            self.ln(5)

# Create PDF
pdf = PDFReport()
pdf.set_auto_page_break(auto=True, margin=15)

# Title Page
pdf.add_page()
pdf.set_font('Helvetica', 'B', 24)
pdf.set_text_color(31, 119, 180)
pdf.ln(30)
pdf.cell(0, 15, 'DU DOAN KHACH HANG ROI BO', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 15, 'DICH VU VIEN THONG', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(10)
pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 10, 'Telco Customer Churn Prediction', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 10, 'Machine Learning Pipeline', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(15)
pdf.set_font('Helvetica', 'B', 12)
pdf.set_text_color(40, 167, 69)
pdf.cell(0, 10, 'PHIEN BAN DA SUA LOI DATA LEAKAGE', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(20)
pdf.set_font('Helvetica', 'I', 11)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 10, 'Thang 1, 2026', align='C', new_x='LMARGIN', new_y='NEXT')

# Data Leakage Fix Explanation
pdf.add_page()
pdf.chapter_title('LOI DATA LEAKAGE VA CACH SUA')

pdf.warning_box('CANH BAO: Phien ban truoc da mac loi Data Leakage khi ap dung SMOTE truoc khi chia train/test, dan den ket qua ao (F1=0.85). Phien ban nay da sua loi.')

pdf.section_title('Van de')
pdf.body_text('- SMOTE da duoc ap dung TRUOC khi chia train/test\n- Tap Test co ty le 50:50 thay vi tu nhien ~27%\n- Mo hinh duoc test tren du lieu nhan tao\n- Ket qua F1=0.85 la AO, khong phan anh thuc te')

pdf.section_title('Cach sua (Best Practice)')
pdf.body_text('1. Chia Train/Test TRUOC\n2. Chi ap dung SMOTE len tap TRAIN\n3. Giu nguyen tap TEST (ty le mat can bang tu nhien ~27%)\n4. Danh gia lai')

pdf.section_title('Ket qua')
pdf.body_text('- Tap Test giu nguyen ty le tu nhien: 73% No, 27% Yes\n- Ket qua thap hon (F1~0.62) nhung TRUNG THUC\n- Phan anh dung hieu nang thuc te cua mo hinh')

# Section 1: Introduction
pdf.add_page()
pdf.chapter_title('1. GIOI THIEU BAI TOAN')

pdf.section_title('1.1 Boi canh')
pdf.body_text('Trong nganh vien thong, viec giu chan khach hang la yeu to song con. Chi phi de co duoc mot khach hang moi cao gap 5-7 lan so voi viec giu chan khach hang hien tai.')

pdf.section_title('1.2 Muc tieu')
pdf.body_text('- Xay dung mo hinh ML du doan khach hang roi bo\n- So sanh hieu nang cac mo hinh ML va Deep Learning\n- Trien khai API va giao dien web')

pdf.section_title('1.3 Bo du lieu')
pdf.body_text('Nguon: IBM Sample Data Sets - Telco Customer Churn\n- So luong mau: 7,043 khach hang\n- So luong dac trung: 21 cot\n- Bien muc tieu: Churn (Yes/No) - ty le 73:27')

# Section 2: EDA
pdf.add_page()
pdf.chapter_title('2. PHAN TICH DU LIEU (EDA)')

pdf.section_title('2.1 Phan bo bien muc tieu')
pdf.add_table(
    ['Churn', 'So luong', 'Ty le'],
    [
        ['No (O lai)', '5,174', '73.5%'],
        ['Yes (Roi bo)', '1,869', '26.5%']
    ]
)
pdf.body_text('Van de: Du lieu mat can bang (Imbalanced) - can xu ly dung cach!')

pdf.add_image_if_exists('models/churn_distribution.png')

pdf.section_title('2.2 Key Insights')
pdf.add_table(
    ['Yeu to', 'Insight'],
    [
        ['Contract', 'Month-to-month: ~43% churn (CAO NHAT!)'],
        ['Internet Service', 'Fiber optic: ~42% churn'],
        ['Payment Method', 'Electronic check: ~45% churn'],
        ['Tenure', 'Khach 0-12 thang: ~47% churn']
    ]
)

# Section 3: Correct SMOTE Application
pdf.add_page()
pdf.chapter_title('3. XU LY MAT CAN BANG - SMOTE (DUNG CACH)')

pdf.section_title('3.1 Quy trinh dung')
pdf.body_text('BUOC 1: Chia train/test (80/20, stratified)\n   Train: 5,634 mau, Test: 1,409 mau\n   Ca hai giu ty le tu nhien 73:27\n\nBUOC 2: Ap dung SMOTE CHI tren tap TRAIN\n   Train sau SMOTE: ~8,270 mau (50:50)\n   Test GIU NGUYEN: 1,409 mau (73:27)\n\nBUOC 3: Train model tren TRAIN (SMOTE)\n\nBUOC 4: Danh gia tren TEST (NGUYEN BAN)')

pdf.section_title('3.2 So sanh')
pdf.add_table(
    ['', 'Cach sai (Leakage)', 'Cach dung (Fixed)'],
    [
        ['Thu tu', 'SMOTE -> Split', 'Split -> SMOTE'],
        ['Ty le Test', '50:50 (ao)', '73:27 (thuc te)'],
        ['F1-Score', '0.85 (ao)', '~0.62 (thuc te)'],
        ['Tin cay', 'KHONG', 'CO']
    ]
)

# Section 4: Results
pdf.add_page()
pdf.chapter_title('4. KET QUA MO HINH (TRUNG THUC)')

pdf.section_title('4.1 So sanh cac mo hinh')
pdf.add_table(
    ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    [
        ['Neural Network', '0.7786', '0.5683', '0.6898', '0.6232', '0.8406'],
        ['Logistic Reg.', '0.7431', '0.5102', '0.7995', '0.6229', '0.8411'],
        ['Random Forest', '0.7679', '0.5519', '0.6684', '0.6046', '0.8403'],
        ['XGBoost', '0.7814', '0.5851', '0.6070', '0.5958', '0.8336']
    ],
    highlight_row=0
)

pdf.body_text('Mo hinh tot nhat: Neural Network voi F1-Score = 0.6232\nLogistic Regression co Recall cao nhat (0.7995) - tot de bat het churners')

pdf.add_image_if_exists('models/model_comparison.png')

pdf.add_page()
pdf.section_title('4.2 Confusion Matrices')
pdf.add_image_if_exists('models/confusion_matrices.png')

pdf.section_title('4.3 Nhan xet ve Deep Learning')
pdf.body_text('Neural Network dat hieu nang tuong duong hoac tot hon cac mo hinh tree-based trong truong hop nay. Tuy nhien, voi du lieu tabular nho (~7000 dong), su khac biet khong dang ke.\n\nVoi du lieu lon hon, tree-based models (XGBoost, RF) thuong hoat dong tot hon vi:\n- Xu ly bien phan loai tot hon\n- Khong can nhieu du lieu de hoi tu\n- It bi overfitting hon')

# Section 5: Error Analysis
pdf.add_page()
pdf.chapter_title('5. PHAN TICH LOI (ERROR ANALYSIS)')

pdf.section_title('5.1 False Negatives - Khach hang bo sot')
pdf.body_text('False Negative (FN) la truong hop nghiem trong nhat:\n- Mo hinh du doan khach hang O LAI\n- Nhung thuc te ho DA ROI BO\n- Doanh nghiep mat co hoi giu chan khach hang')

pdf.section_title('5.2 Chien luoc cai thien')
pdf.body_text('1. Tang Recall: Chap nhan nhieu False Positive hon de giam False Negative\n2. Dieu chinh threshold: Giam nguong tu 0.5 xuong 0.3-0.4\n3. Cost-sensitive learning: Phat nang FN hon FP\n4. Ensemble: Ket hop nhieu mo hinh')

# Section 6: Business Recommendations
pdf.add_page()
pdf.chapter_title('6. KIEN NGHI DOANH NGHIEP')

pdf.add_table(
    ['Nhom rui ro', 'Dac diem', 'Hanh dong'],
    [
        ['Month-to-month', '43% churn', 'Giam 20% khi len 1-2 nam'],
        ['Khach moi (0-12m)', '47% churn', 'Chuong trinh onboarding'],
        ['Electronic check', '45% churn', 'Thuong auto-pay'],
        ['It dich vu', 'De roi bo', 'Goi bundle giam gia'],
        ['Fiber optic', '42% churn', 'Kiem tra chat luong']
    ]
)

pdf.section_title('6.2 Quy trinh ap dung')
pdf.body_text('1. Du doan hang thang: Chay model tren tat ca khach hang\n2. Xep hang rui ro: Phan loai Low/Medium/High risk\n3. Hanh dong: Lien he khach hang High risk truoc\n4. Theo doi: Do luong hieu qua chien luoc retention')

# Section 7: Conclusion
pdf.add_page()
pdf.chapter_title('7. KET LUAN')

pdf.section_title('7.1 Bai hoc ve Data Leakage')
pdf.body_text('- Data Leakage la loi nghiem trong co the khien ket qua ao\n- Luon chia train/test TRUOC khi xu ly\n- Ket qua thap hon nhung trung thuc quan trong hon')

pdf.section_title('7.2 Ket qua thuc te')
pdf.body_text('- F1-Score thuc te: ~0.62 (khong phai 0.85)\n- AUC ~0.84: Mo hinh van co kha nang phan biet tot\n- Can ket hop voi domain knowledge va chien luoc business')

pdf.section_title('7.3 Huong phat trien')
pdf.body_text('1. Thu nghiem threshold thap hon (0.3-0.4) de tang Recall\n2. Feature engineering them: Interaction features\n3. Hyperparameter tuning: GridSearchCV\n4. Model monitoring: Theo doi model drift')

# Save PDF
pdf.output('Report.pdf')
print("✅ PDF Report (FIXED VERSION) created: Report.pdf")
print(f"   Size: {os.path.getsize('Report.pdf') / 1024:.1f} KB")
