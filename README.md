# 🎯 JIB Marketing Campaign Budget Estimator

ระบบประมาณการงบประมาณแคมเปญการตลาดล่วงหน้า (Campaign Budget Estimation Model) สำหรับร้านค้าจำหน่ายสินค้าไอที **J.I.B. Computer Group** โดยใช้เทคนิค Machine Learning เพื่อช่วยให้ฝ่ายการตลาดและทีมวางแผนกลยุทธ์สามารถจัดสรรงบประมาณได้อย่างแม่นยำ สมเหตุสมผล และลดความเสี่ยงจากการใช้งบเกินหรือขาด

---

## 📌 ภาพรวมโปรเจกต์ (Project Overview)

การจัดสรรงบประมาณแคมเปญการตลาดมักขึ้นอยู่กับปัจจัยหลายด้าน เช่น ประเภทแคมเปญ, ช่องทางการตลาด, กลุ่มเป้าหมาย และช่วงเวลาที่เริ่มแคมเปญ โปรเจกต์นี้จึงพัฒนาขึ้นเพื่อ:
* แปลงข้อมูลแคมเปญและคุณลักษณะเชิงเวลา (Temporal Features) ให้เป็นตัวแปรที่โมเดลเข้าใจ
* พยากรณ์งบประมาณที่เหมาะสมด้วยโมเดลแบบ Ensemble Learning
* ให้บริการผ่านเว็บแอปพลิเคชันด้วย **Streamlit** ที่ใช้งานง่ายและโต้ตอบได้ทันที

---

## ⚙️ สถาปัตยกรรมโมเดลและฟีเจอร์ (Model Architecture & Features)

### 1. ฟีเจอร์ที่ใช้ในการพยากรณ์ (Features Used)
* **Categorical Features:**
  * `Campaign_Type`: ประเภทแคมเปญ (เช่น Seasonal Sale, Product Launch, Flash Sale)
  * `Channel`: ช่องทางการโฆษณา (เช่น Social Media, Online Store, In-Store Branch, Shopee / Lazada, Line OA)
  * `Target_Audience`: กลุ่มเป้าหมาย (เช่น Gamers & Creators, Students & Education, General Public)
* **Numerical & Engineered Features:**
  * `duration_days`: จำนวนวันที่จัดแคมเปญ (คำนวณจาก `End_Date - Start_Date`)
  * `start_month`: เดือนที่เริ่มจัดแคมเปญ (1 - 12)
  * `start_quarter`: ไตรมาสที่จัดแคมเปญ (Q1 - Q4)
  * `start_dayofweek`: วันในสัปดาห์ที่เริ่มแคมเปญ (0 = จันทร์, 6 = อาทิตย์)

### 2. โมเดลการเรียนรู้ (GradientBoostingRegressor Model)
* **Pipeline:** รวมขั้นตอน `OneHotEncoder(drop='first')` และตัวโมเดลเข้าด้วยกันเพื่อป้องกัน Data Leakage
* **Algorithm:** **HistGradientBoostingRegressor**
  * `max_iter`: 180
  * `learning_rate`: 0.04
  * `max_depth`: 4
  * `l2_regularization`: 1.0
  * `random_state`: 42

---

## 📊 ประสิทธิภาพและการประเมินโมเดล (5-Fold Cross-Validation)

ประเมินผลผ่าน **5-Fold Cross-Validation** บนชุดข้อมูล 1,000 ตัวอย่าง:

| ตัวชี้วัด (Metric) | ค่าเฉลี่ย (Mean) | ค่าเบี่ยงเบนมาตรฐาน (Std) |
| :--- | :---: | :---: |
| **$R^2$ Score** | **0.7378** | ± 0.0270 |
| **Mean Absolute Error (MAE)** | **60,155.65 บาท** | ± 2,230.12 บาท |
| **Root Mean Squared Error (RMSE)** | **76,820.40 บาท** | ± 2,985.40 บาท |

> **หมายเหตุ:** โมเดลสามารถอธิบายความแปรปรวนของงบประมาณ ($R^2$) ได้ถึง ~73.8% โดยมีค่าเฉลี่ยความคลาดเคลื่อนอยู่ที่ประมาณ 60,000 บาท ซึ่งสอดคล้องกับ Safe Margin ที่แสดงบนหน้าเว็บ

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
JIB_Model_Analysis/
├── JIB_Marketing_Campaign.csv   # ชุดข้อมูลแคมเปญการตลาด
├── train_model.py              # สคริปต์ทำ Feature Engineering, 5-Fold CV และบันทึกโมเดล
├── app.py                      # โค้ดหน้าเว็บแอปพลิเคชัน Streamlit
├── campaign_budget_model.pkl   # ไฟล์โมเดล Pipeline ที่ผ่านการเทรน
├── requirements.txt            # รายการไลบรารีที่จำเป็น
└── README.md                   # เอกสารอธิบายรายละเอียดโปรเจกต์
