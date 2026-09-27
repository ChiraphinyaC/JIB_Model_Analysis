import joblib
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

base_dir = Path(__file__).resolve().parent
root_dir = base_dir.parent

csv_candidates = [
    root_dir / "JIB_Marketing_Campaign.csv",
    root_dir / "JIB_Marketing_Campaign .csv",
    base_dir / "JIB_Marketing_Campaign.csv",
    base_dir / "JIB_Marketing_Campaign .csv",
]

csv_path = next((p for p in csv_candidates if p.exists()), None)
if csv_path is None:
    raise FileNotFoundError(
        "ไม่พบไฟล์ CSV ของแคมเปญ กรุณาตรวจสอบว่ามีไฟล์ "
        "'JIB_Marketing_Campaign.csv' หรือ 'JIB_Marketing_Campaign .csv' "
        "อยู่ในโฟลเดอร์หลักหรือ model3"
    )

print(f"กำลังโหลดข้อมูลจาก: {csv_path}")
df = pd.read_csv(csv_path)
df.columns = [str(c).strip() for c in df.columns]

budget_col = next(
    (c for c in ["Budget", "budget", "Campaign_Budget", "Campaign Budget"] if c in df.columns),
    None
)
if budget_col is None:
    raise KeyError("ไม่พบคอลัมน์งบประมาณ เช่น 'Budget' หรือ 'Campaign_Budget'")

if {"Start_Date", "End_Date"}.issubset(df.columns):
    df["Start_Date"] = pd.to_datetime(df["Start_Date"], errors="coerce")
    df["End_Date"] = pd.to_datetime(df["End_Date"], errors="coerce")
    df["duration_days"] = (df["End_Date"] - df["Start_Date"]).dt.days
elif "Duration_Days" in df.columns:
    df["duration_days"] = pd.to_numeric(df["Duration_Days"], errors="coerce")
else:
    raise KeyError("ไม่พบคอลัมน์วันเริ่ม/สิ้นสุดแคมเปญ (Start_Date/End_Date) หรือ Duration_Days")

df = df[[budget_col, "duration_days"]].copy()
df.columns = ["Budget", "duration_days"]

df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce")
df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")
df = df.dropna().reset_index(drop=True)

if len(df) < 3:
    raise ValueError("ข้อมูลมีแถวไม่เพียงพอสำหรับการ clustering ต้องมีอย่างน้อย 3 แถว")

X = df[["Budget", "duration_days"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score (k={k}) = {sil_score:.4f}")

tier_names = {
    0: "Tier 1: High Budget & Long Duration (แคมเปญระดับ Flagship/ระยะยาว)",
    1: "Tier 2: Mid-Tier Standard Campaigns (แคมเปญตามรอบปกติ/ระดับกลาง)",
    2: "Tier 3: Quick Tactical & Flash Sales (แคมเปญกระตุ้นยอดด่วน/ระยะสั้น)"
}

cluster_package = {
    "scaler": scaler,
    "kmeans": kmeans,
    "tier_names": tier_names,
    "feature_cols": ["Budget", "duration_days"]
}

output_model_path = base_dir / "campaign_cluster_model.pkl"
joblib.dump(cluster_package, output_model_path)

print(f"\n[สำเร็จ] บันทึกโมเดลลงใน '{output_model_path}' เรียบร้อยแล้ว!")