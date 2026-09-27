import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. โหลดข้อมูล
df = pd.read_csv('JIB_Marketing_Campaign .csv')

# 2. Feature Engineering
df['Start_Date'] = pd.to_datetime(df['Start_Date'])
df['End_Date'] = pd.to_datetime(df['End_Date'])
df['duration_days'] = (df['End_Date'] - df['Start_Date']).dt.days
df['start_month'] = df['Start_Date'].dt.month
df['start_quarter'] = df['Start_Date'].dt.quarter
df['start_dayofweek'] = df['Start_Date'].dt.dayofweek

feature_cols = ['Campaign_Type', 'Channel', 'Target_Audience', 'duration_days', 'start_month', 'start_quarter', 'start_dayofweek']
X = df[feature_cols]
y = df['Budget']

categorical_cols = ['Campaign_Type', 'Channel', 'Target_Audience']
numerical_cols = ['duration_days', 'start_month', 'start_quarter', 'start_dayofweek']

# 3. Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

model = HistGradientBoostingRegressor(
    max_iter=180,
    learning_rate=0.04,
    max_depth=4,
    l2_regularization=1.0,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# 4. 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'r2': 'r2', 'neg_mae': 'neg_mean_absolute_error', 'neg_rmse': 'neg_root_mean_squared_error'}
cv_results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring)

print("=== 5-Fold Cross-Validation Results ===")
print(f"Mean R2:   {cv_results['test_r2'].mean():.4f}")
print(f"Mean MAE:  {-cv_results['test_neg_mae'].mean():,.2f} THB")

# 5. Fit กับข้อมูลทั้งหมดและบันทึกโมเดล
pipeline.fit(X, y)
joblib.dump(pipeline, 'campaign_budget_model.pkl')
print("\nบันทึกโมเดลลงใน campaign_budget_model.pkl เรียบร้อยแล้ว!")