# 📢 J.I.B. Smart Channel & Strategy Recommender (Model 2)

ระบบปัญญาประดิษฐ์เพื่อการวิเคราะห์และแนะนำช่องทางการตลาดที่เหมาะสมที่สุด (Smart Marketing Channel Recommender) พัฒนาขึ้นสำหรับกลุ่มธุรกิจค้าปลีกสินค้าไอที **J.I.B. Computer Group** เพื่อแก้ปัญหาในการจัดสรรงบประมาณและการเลือกช่องทางสื่อโฆษณาให้เกิดความคุ้มค่าและตรงกลุ่มเป้าหมายมากที่สุด

---

## 🎯 วัตถุประสงค์ (Objectives)

* **Business Problem:** นักการตลาดมักมีไอเดียแคมเปญ กลุ่มเป้าหมาย และงบประมาณในใจ แต่ขาดข้อมูลเชิงประจักษ์ว่าควรยิงแคมเปญผ่านช่องทางใดเพื่อให้เกิดประสิทธิภาพสูงสุด
* **Solution:** นำข้อมูลประวัติแคมเปญของ J.I.B. มาสร้างโมเดล Machine Learning แบบ Multi-Class Classification เพื่อทำนายและจัดอันดับช่องทางประชาสัมพันธ์ที่มีความเหมาะสมสูงสุด พร้อมเปอร์เซ็นต์ความน่าจะเป็นและคำแนะนำเชิงกลยุทธ์

---

## 🧠 สถาปัตยกรรมโมเดล (Model Architecture)

* **ประเภทโมเดล:** Supervised Learning — Multi-Class Classification
* **อัลกอริทึม (Algorithm):** `RandomForestClassifier` (Ensemble Learning)
  * `n_estimators`: 150
  * `max_depth`: 8
  * `min_samples_split`: 4
  * `random_state`: 42
* **Target Variable ($Y$):** `Channel`
  * คลาสเป้าหมาย 5 ช่องทาง: `Social Media`, `Online Store`, `In-Store Branch`, `Shopee / Lazada`, `Line OA`
* **Features ที่ใช้ ($X$):**
  1. `Campaign_Type` (Categorical): ประเภทของแคมเปญ
  2. `Target_Audience` (Categorical): กลุ่มลูกค้าเป้าหมาย
  3. `Budget` (Numerical): งบประมาณของแคมเปญ (บาท)
  4. `duration_days` (Numerical): ระยะเวลาจัดแคมเปญ (คำนวณจาก `End_Date - Start_Date`)

---

## ⚙️ Data Preprocessing & Pipeline

สร้าง `Pipeline` ผ่าน `ColumnTransformer` เพื่อป้องกัน Data Leakage:
* **Categorical Features:** จัดการด้วย `OneHotEncoder(drop='first', handle_unknown='ignore')`
* **Numerical Features:** ส่งต่อค่าตรงผ่าน Pipeline พร้อมสเกลร่วมกับ Tree-based Model

---

## 📊 ประสิทธิภาพของโมเดล (Model Performance)

ทดสอบและประเมินผลผ่าน **5-Fold Stratified Cross-Validation**:

| Metric | Score | ความหมายเชิงสถิติ |
| :--- | :---: | :--- |
| **Mean Accuracy** | **~95.20%** | โมเดลทำนายช่องทางได้ถูกต้องแม่นยำสูงบนชุดทดสอบ |
| **Mean F1-Macro** | **~95.03%** | รักษาความสมดุลของความแม่นยำในทุกคลาสช่องทาง ไม่เอนเอียงตามคลาสที่มีขนาดใหญ่ |

---

## 🖥️ คุณสมบัติของเว็บแอปพลิเคชัน (`app2.py`)

1. **User Interface ใช้งานง่าย:** เลือกเงื่อนไขแคมเปญผ่าน Dropdown, ตัวเลขงบประมาณ และ Slider กำหนดจำนวนวัน
2. **Top Recommended Channel:** สรุปช่องทางที่แนะนำอันดับ 1 พร้อมระบุความเหมาะสมเป็นเปอร์เซ็นต์
3. **Probability Distribution:** แสดงแถบระดับความน่าจะเป็นของช่องทางทั้งหมด (Top 1 ถึง 5) ผ่าน Progress Bar
4. **Actionable Strategic Tips:** นำเสนอคำแนะนำเชิงกลยุทธ์เฉพาะกลุ่ม เช่น เทคนิคการขายหน้าร้านสำหรับกลุ่ม DIY PC Builders หรือการใช้คูปอง Flash Sale บน Marketplace

---

## 📂 โครงสร้างโฟลเดอร์ (Directory Structure)

```text
model2/
├── app2.py                      # สคริปต์หน้าเว็บ Streamlit Dashboard
├── train_channel_model.py       # สคริปต์เทรนและประเมินผลโมเดล Classification
├── channel_recommender_model.pkl # ไฟล์โมเดล Pipeline ที่เทรนเสร็จสมบูรณ์
└── README.md                    # เอกสารอธิบายรายละเอียดโมเดล 2