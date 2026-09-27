# 📊 J.I.B. Campaign Portfolio Tier & Segmentation (Model 3)

ระบบปัญญาประดิษฐ์เพื่อการวิเคราะห์และจัดกลุ่มระดับแคมเปญการตลาด (Campaign Portfolio Segmentation) สำหรับกลุ่มธุรกิจค้าปลีกสินค้าไอที **J.I.B. Computer Group** โดยใช้วิธีการเรียนรู้แบบไม่มีผู้สอน (Unsupervised Learning) เพื่อช่วยฝ่ายบริหารและทีมการตลาดในการจัดกลุ่ม Tier แคมเปญ และวางแผนจัดสรรทรัพยากร/สต็อกสินค้าได้อย่างเป็นระบบ

---

## 🎯 วัตถุประสงค์ (Objectives)

* **Business Problem:** ในแต่ละปี J.I.B. มีการจัดแคมเปญการตลาดจำนวนมากที่มีความหลากหลายด้านงบประมาณและระยะเวลา ทำให้การจัดสรรทรัพยากรบุคคล การเตรียมสต็อกสินค้า และการตั้งเป้าหมาย KPI ขาดเกณฑ์มาตรฐานที่ชัดเจน
* **Solution:** ใช้อัลกอริทึม **K-Means Clustering** จัดกลุ่มแคมเปญตามคุณลักษณะเชิงปริมาณ (Quantitative Dimensions) เพื่อแบ่งระดับพอร์ตโฟลิโอออกเป็น 3 Tiers เชิงกลยุทธ์ พร้อมคำแนะนำแนวทางการบริหารจัดการ

---

## 🧠 สถาปัตยกรรมและการพัฒนาโมเดล (Model Architecture)

* **ประเภทการเรียนรู้:** Unsupervised Learning (Clustering Analysis)
* **อัลกอริทึมหลัก:** `K-Means Clustering`
  * กำหนดค่าจำนวนกลุ่ม: $k = 3$
  * `random_state`: 42
  * `n_init`: 'auto'
* **ตัวแปรเชิงปริมาณที่ใช้ (Quantitative Features):**
  1. `Budget` (Numerical): งบประมาณของแคมเปญ (บาท)
  2. `duration_days` (Numerical): ระยะเวลาจัดแคมเปญ (จำนวนวัน)
* **การแปลงสเกลข้อมูล (Feature Scaling):**
  * ปรับสเกลข้อมูลด้วย `StandardScaler` เพื่อปรับค่าเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1 ป้องกันไม่ให้ตัวแปร Budget ที่มีสเกลหลักแสนกลืนตัวแปร duration_days ที่มีสเกลหลักสิบ

---

## 📈 การประเมินประสิทธิภาพและการเลือกจำนวนกลุ่ม (Evaluation & Optimal k)

### 1. การหาจำนวนกลุ่มที่เหมาะสม (Optimal k)
* **Elbow Method:** คำนวณค่า WCSS (Within-Cluster Sum of Squares) สำหรับ $k = 1$ ถึง $10$ และหาจุดหักศอกด้วย `KneeLocator` พบจุดศอกที่ชัดเจนที่ **$k = 3$**
* **Silhouette Analysis:** ตรวจสอบความกระชับและการแยกตัวของคลัสเตอร์ โดยค่า $k = 3$ ได้คะแนนสูงสุด:
  * $k = 2$ : 0.379
  * **$k = 3$ : 0.433 (Optimal k - สูงที่สุด)**
  * $k = 4$ : 0.404
  * $k = 5$ ถึง $10$ : มีค่าลดหลั่นลงมาตามลำดับ (0.385 - 0.365)

### 2. เปรียบเทียบผลกับ Hierarchical Clustering (ที่ $k = 3$)
* **K-Means Clustering:** Silhouette Score = **0.433**
* **Hierarchical Clustering (Ward's Linkage):** Silhouette Score = **0.395**
* **สรุป:** K-Means ให้โครงสร้างการเกาะกลุ่มที่มีความแน่น (Compactness) และแยกออกจากกันได้ชัดเจนกว่าอย่างมีนัยสำคัญ

---

## 🏷️ การนิยามกลุ่มเชิงธุรกิจ (Business Strategic Tiers)

แคมเปญทั้งหมดถูกจัดกลุ่มออกเป็น 3 Tiers ดังนี้:

| Tier | ชื่อกลุ่มเชิงกลยุทธ์ | คุณลักษณะสำคัญ | กลยุทธ์การบริหารจัดการ |
| :---: | :--- | :--- | :--- |
| **Tier 1** | High Budget & Long Duration | งบประมาณสูงมาก จัดต่อเนื่องระยะยาว (Flagship / Expo) | ประสานงานซัพพลายเออร์สำรองสต็อกการ์ดจอ/ซีพียูล่วงหน้า 2-3 สัปดาห์ และเน้นสื่อ Omni-channel |
| **Tier 2** | Mid-Tier Standard Campaigns | งบประมาณและระยะเวลาปานกลาง (Seasonal / Back to School) | เน้นจัดเซต Bundle Promotion เพื่อเพิ่ม Basket Size ต่อยอดบิล |
| **Tier 3** | Quick Tactical & Flash Sales | งบประมาณต่ำ-กลาง ระยะเวลาสั้น 1-3 วัน (Payday / Double Day) | เน้นความไว ลดราคากระแทกใจบน Marketplace เพื่อเร่งปิดการขาย |

---

## 🖥️ คุณสมบัติของเว็บแอปพลิเคชัน (`app3.py`)

1. **Parameter Inputs:** ปรับแต่งงบประมาณ (Number Input) และระยะเวลา (Slider)
2. **Instant Cluster Assignment:** คำนวณและแสดงผล Tier ที่แคมเปญถูกจัดกลุ่มในทันที
3. **Actionable Strategic Insights:** แสดงคำแนะนำเฉพาะสำหรับแต่ละ Tier ครอบคลุมทั้งด้านสต็อกสินค้าและช่องทางสื่อ
4. **Academic Model Inspection:** แถบ Expander แสดงสถิติและผลการประเมินโมเดลตามเกณฑ์วิชาการ

---

## 📂 โครงสร้างโฟลเดอร์ (Directory Structure)

```text
model3/
├── app3.py                      # สคริปต์หน้าเว็บ Streamlit Dashboard
├── train_cluster_model.py       # สคริปต์เทรน K-Means และประเมิน Silhouette
├── campaign_cluster_model.pkl   # ไฟล์โมเดลและ Scaler สำหรับนำไป Predict
└── README.md                    # เอกสารอธิบายรายละเอียดโมเดล 3
