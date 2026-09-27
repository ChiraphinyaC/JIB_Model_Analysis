import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. โหลดข้อมูล
try:
    df = pd.read_csv('JIB_Marketing_Campaign.csv')
except FileNotFoundError:
    df = pd.read_csv('JIB_Marketing_Campaign .csv')

# 2. Feature Engineering
df['Start_Date'] = pd.to_datetime(df['Start_Date'])
df['End_Date'] = pd.to_datetime(df['End_Date'])
df['duration_days'] = (df['End_Date'] - df['Start_Date']).dt.days

# กำหนด Features และ Target
feature_cols = ['Campaign_Type', 'Target_Audience', 'Budget', 'duration_days']
X = df[feature_cols]
y = df['Channel']

categorical_cols = ['Campaign_Type', 'Target_Audience']
numerical_cols = ['Budget', 'duration_days']

# 3. สร้าง Pipeline ป้องกัน Data Leakage
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

channel_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=4,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', channel_model)
])

# 4. ทดสอบความแม่นยำด้วย 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=['accuracy', 'f1_macro'])

print("=== ผลลัพธ์ 5-Fold Cross-Validation (Channel Model) ===")
print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
print(f"Mean F1 Macro: {cv_results['test_f1_macro'].mean():.4f} (+/- {cv_results['test_f1_macro'].std():.4f})")

# 5. Fit ทั้งหมดและบันทึกโมเดล
pipeline.fit(X, y)
joblib.dump(pipeline, 'channel_recommender_model.pkl')
print("\n[สำเร็จ] บันทึกโมเดลลงใน 'channel_recommender_model.pkl' เรียบร้อยแล้ว!")