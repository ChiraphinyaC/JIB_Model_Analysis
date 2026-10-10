import os
import sqlite3
import pandas as pd
import numpy as np
import joblib

# นำเข้าโมดูลสำหรับประเมินผลและสร้างโมเดล Machine Learning
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# ขั้นตอนที่ 1: การโหลดและเตรียมข้อมูลดิบ (Data Ingestion & Feature Engineering)
# ==============================================================================

# ตรวจสอบและค้นหาตำแหน่งไฟล์ CSV แคมเปญการตลาดของ J.I.B. (รองรับกรณีรันจากโฟลเดอร์หลักหรือโฟลเดอร์ย่อย)
csv_path = 'JIB_Marketing_Campaign .csv'
if not os.path.exists(csv_path):
    csv_path = '../JIB_Marketing_Campaign .csv'
if not os.path.exists(csv_path):
    csv_path = 'JIB_Marketing_Campaign.csv'
if not os.path.exists(csv_path):
    csv_path = '../JIB_Marketing_Campaign.csv'

# อ่านไฟล์ CSV เข้ามาเป็น DataFrame
raw_df = pd.read_csv(csv_path)

# แปลงประเภทข้อมูลคอลัมน์วันที่ให้อยู่ในรูปแบบ datetime เพื่อนำไปคำนวณ
raw_df['Start_Date'] = pd.to_datetime(raw_df['Start_Date'])
raw_df['End_Date'] = pd.to_datetime(raw_df['End_Date'])

# สร้าง Feature ใหม่: คำนวณระยะเวลาจัดแคมเปญ (duration_days) จากผลต่างของวันสิ้นสุดและวันเริ่มต้น
raw_df['duration_days'] = (raw_df['End_Date'] - raw_df['Start_Date']).dt.days


# ==============================================================================
# ขั้นตอนที่ 2: จำลองสถาปัตยกรรมคลังข้อมูล (Data Warehouse - Star Schema) ใน SQLite
# ==============================================================================

# สร้างฐานข้อมูลจำลองในหน่วยความจำ RAM (In-Memory Database) ไม่ต้องเปิด Server แยก
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# ค้นหาและเปิดอ่านไฟล์ schema.sql ที่เก็บคำสั่ง DDL สร้างตาราง และคำสั่ง SQL JOIN
sql_file_path = 'schema.sql'
if not os.path.exists(sql_file_path):
    sql_file_path = 'model2/schema.sql'
if not os.path.exists(sql_file_path):
    sql_file_path = '../model2/schema.sql'

with open(sql_file_path, 'r', encoding='utf-8') as f:
    sql_content = f.read()

# แยกสคริปต์ออกเป็น 2 ส่วน: ส่วนสร้างตาราง (DDL) และส่วนดึงข้อมูล (SELECT)
parts = sql_content.split('-- 2.')
ddl_script = parts[0]
select_query = '-- 2.' + parts[1] if len(parts) > 1 else ""

# ตรวจสอบคำสั่ง SELECT เพื่อนำไปรัน JOIN ดึงข้อมูลเข้าโมเดล
if 'SELECT' in select_query:
    select_sql = select_query[select_query.find('SELECT'):]
else:
    # คำสั่งสำรองกรณีไม่พบคำสั่ง SELECT ในไฟล์ schema.sql
    select_sql = """
    SELECT 
        f.Campaign_ID,
        ct.Campaign_Type,
        ta.Target_Audience,
        f.Budget,
        f.duration_days,
        ch.Channel_Name AS Channel
    FROM Fact_Marketing_Campaign f
    JOIN Dim_Campaign_Type ct    ON f.Type_ID = ct.Type_ID
    JOIN Dim_Target_Audience ta  ON f.Audience_ID = ta.Audience_ID
    JOIN Dim_Channel ch          ON f.Channel_ID = ch.Channel_ID
    JOIN Dim_Date d              ON f.Date_Key = d.Date_Key;
    """

# รันคำสั่ง DDL สร้างตารางตามโครงสร้าง Star Schema (4 Dimension Tables + 1 Fact Table)
cursor.executescript(ddl_script)


# ==============================================================================
# ขั้นตอนที่ 3: กระบวนการ ETL (นำเข้าข้อมูลจาก CSV กระจายลง Dimension และ Fact Tables)
# ==============================================================================

# --- สร้างมิติที่ 1: ตารางประเภทแคมเปญ (Dim_Campaign_Type) ---
dim_type = pd.DataFrame({'Campaign_Type': raw_df['Campaign_Type'].unique()}).reset_index()
dim_type.columns = ['Type_ID', 'Campaign_Type']
dim_type['Type_ID'] += 1  # กำหนด Primary Key เริ่มต้นที่ 1

# --- สร้างมิติที่ 2: ตารางกลุ่มเป้าหมาย (Dim_Target_Audience) ---
dim_aud = pd.DataFrame({'Target_Audience': raw_df['Target_Audience'].unique()}).reset_index()
dim_aud.columns = ['Audience_ID', 'Target_Audience']
dim_aud['Audience_ID'] += 1  # กำหนด Primary Key เริ่มต้นที่ 1

# --- สร้างมิติที่ 3: ตารางช่องทางการสื่อสาร (Dim_Channel) ---
dim_chan = pd.DataFrame({'Channel_Name': raw_df['Channel'].unique()}).reset_index()
dim_chan.columns = ['Channel_ID', 'Channel_Name']
dim_chan['Channel_ID'] += 1  # กำหนด Primary Key เริ่มต้นที่ 1

# --- เตรียมตารางข้อเท็จจริง (Fact Table): แมป Foreign Key กลับเข้าสู่ข้อมูลหลัก ---
fact_df = raw_df.copy()
fact_df = fact_df.merge(dim_type, on='Campaign_Type')
fact_df = fact_df.merge(dim_aud, on='Target_Audience')
fact_df = fact_df.merge(dim_chan, left_on='Channel', right_on='Channel_Name')

# ตรวจสอบและสร้าง Campaign_ID สำหรับเป็น Primary Key ให้ Fact Table หากยังไม่มี
if 'Campaign_ID' not in fact_df.columns:
    fact_df['Campaign_ID'] = [f'CP-{i+1:04d}' for i in range(len(fact_df))]

# กำหนด Date_Key ในรูปแบบตัวเลข YYYYMMDD (เช่น 20250101) เพื่อใช้เชื่อมโยงกับ Dim_Date
fact_df['Date_Key'] = fact_df['Start_Date'].dt.strftime('%Y%m%d').astype(int)

# --- สร้างมิติที่ 4: ตารางมิติด้านเวลา (Dim_Date) ---
# ใช้ drop_duplicates(subset=['Date_Key']) เพื่อป้องกันค่า Primary Key ซ้ำในฐานข้อมูล
dim_date = fact_df[['Date_Key', 'Start_Date']].drop_duplicates(subset=['Date_Key']).copy()
dim_date['Start_Month'] = dim_date['Start_Date'].dt.month
dim_date['Start_Quarter'] = dim_date['Start_Date'].dt.quarter
dim_date['End_Date'] = dim_date['Start_Date'].dt.strftime('%Y-%m-%d')
dim_date['Start_Date'] = dim_date['Start_Date'].dt.strftime('%Y-%m-%d')

# จัดเรียงลำดับคอลัมน์ให้ตรงตามโครงสร้างตาราง Dim_Date ใน SQL
dim_date = dim_date[['Date_Key', 'Start_Date', 'End_Date', 'Start_Month', 'Start_Quarter']]

# บันทึกข้อมูลทั้งหมดลงในตารางฐานข้อมูล SQLite
dim_type.to_sql('Dim_Campaign_Type', conn, if_exists='append', index=False)
dim_aud.to_sql('Dim_Target_Audience', conn, if_exists='append', index=False)
dim_chan.to_sql('Dim_Channel', conn, if_exists='append', index=False)
dim_date.to_sql('Dim_Date', conn, if_exists='append', index=False)

# คัดเลือกเฉพาะ Foreign Keys และ Measures เพื่อบันทึกลง Fact_Marketing_Campaign
fact_cols = ['Campaign_ID', 'Type_ID', 'Audience_ID', 'Channel_ID', 'Date_Key', 'Budget', 'duration_days']
fact_df[fact_cols].to_sql('Fact_Marketing_Campaign', conn, if_exists='append', index=False)


# ==============================================================================
# ขั้นตอนที่ 4: สกัดข้อมูลผ่าน SQL JOIN (หัวใจสำคัญของ Data Warehouse Pipeline)
# ==============================================================================

# รันคำสั่ง SQL JOIN ดึงเฉพาะข้อมูลที่จำเป็นเข้าสู่ตารางแบน (Flat Table) สำหรับเทรนโมเดล
df = pd.read_sql_query(select_sql, conn)
conn.close()  # ปิดการเชื่อมต่อฐานข้อมูลเมื่อดึงข้อมูลเสร็จสิ้น

# แสดงตัวอย่างข้อมูลผลลัพธ์ที่ได้จากการรัน SQL JOIN
print("--- ตัวอย่างข้อมูลที่สกัดจาก SQL JOIN (5 แถวแรก) ---")
print(df.head())


# ==============================================================================
# ขั้นตอนที่ 5: การสร้าง Machine Learning Pipeline (Random Forest Classifier)
# ==============================================================================

# กำหนดตัวแปรต้น (Features: X) และตัวแปรเป้าหมาย (Target: y)
feature_cols = ['Campaign_Type', 'Target_Audience', 'Budget', 'duration_days']
X = df[feature_cols]
y = df['Channel']

# แบ่งกลุ่มประเภทของคอลัมน์เพื่อทำการ Preprocessing
categorical_cols = ['Campaign_Type', 'Target_Audience']
numerical_cols = ['Budget', 'duration_days']

# ใช้ ColumnTransformer ทำ One-Hot Encoding เฉพาะคอลัมน์ที่เป็นข้อความ
# drop='first' ช่วยลด Multicollinearity และ handle_unknown='ignore' เพื่อป้องกัน error จากข้อมูลใหม่
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # ให้คอลัมน์ตัวเลขผ่านไปได้โดยตรง
)

# กำหนดโมเดล Random Forest Classifier พร้อมตั้งค่าไฮเปอร์พารามิเตอร์เพื่อป้องกัน Overfitting
channel_model = RandomForestClassifier(
    n_estimators=150,       # จำนวนต้นไม้ตัดสินใจ 150 ต้น
    max_depth=8,            # ควบคุมความลึกสูงสุดของแต่ละต้นไม้ไม่เกิน 8 ชั้น
    min_samples_split=4,    # จำนวนตัวอย่างขั้นต่ำที่ต้องมีเพื่อใช้แตกกิ่ง
    random_state=42         # ล็อกค่าสุ่มเพื่อให้ได้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน
)

# ประกอบ Preprocessor และ Model เข้าด้วยกันผ่าน Pipeline ป้องกันปัญหา Data Leakage
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', channel_model)
])


# ==============================================================================
# ขั้นตอนที่ 6: การวัดประสิทธิภาพโมเดลด้วย 5-Fold Stratified Cross-Validation
# ==============================================================================

# ใช้ StratifiedKFold เพื่อรักษาอัตราส่วนการกระจายตัวของคลาสเป้าหมายในแต่ละ Fold ให้เท่ากัน
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ประเมินผลทั้งค่าความแม่นยำ (Accuracy) และ Macro F1-Score
cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=['accuracy', 'f1_macro'])

print("\n=== ผลลัพธ์ 5-Fold Cross-Validation (Channel Model) ===")
print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
print(f"Mean F1 Macro: {cv_results['test_f1_macro'].mean():.4f} (+/- {cv_results['test_f1_macro'].std():.4f})")


# ==============================================================================
# ขั้นตอนที่ 7: ฝึกสอนโมเดลบนชุดข้อมูลทั้งหมดและบันทึกไฟล์ (Model Deployment)
# ==============================================================================

# ฝึกสอน Pipeline กับข้อมูลทั้งหมด
pipeline.fit(X, y)

# ส่งออกโมเดลและพรีโพรเซสเซอร์ทั้งหมดเป็นไฟล์ .pkl สำหรับนำไปใช้งานต่อบนเว็บแอปพลิเคชัน Streamlit
joblib.dump(pipeline, 'channel_recommender_model.pkl')
print("\n[สำเร็จ] บันทึกโมเดลลงใน 'channel_recommender_model.pkl' เรียบร้อยแล้ว!")