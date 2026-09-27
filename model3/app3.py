import os
from pathlib import Path

import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# กำหนดค่าหน้าเว็บ
st.set_page_config(
    page_title="JIB Marketing Strategy Hub",
    page_icon="🎯",
    layout="centered",
)

st.title("🎯 JIB Marketing Campaign & Strategy Hub")
st.markdown("ระบบ AI วิเคราะห์และวางแผนกลยุทธ์การตลาดแบบครบวงจรสำหรับ J.I.B. Computer Group")

# -----------------------------------------
# Helper: ค้นหาโมเดลจากหลาย path
# -----------------------------------------
def candidate_model_paths(filename):
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parent
    search_dirs = [
        base_dir,
        root_dir,
        base_dir / "model2",
        base_dir / "model3",
        root_dir / "model2",
        root_dir / "model3",
    ]

    paths = []
    seen = set()
    for d in search_dirs:
        p = d / filename
        if str(d) not in seen and p.exists():
            paths.append(str(p))
            seen.add(str(d))
    return paths

def load_model_or_none(filename, label_name):
    paths = candidate_model_paths(filename)
    if not paths:
        st.warning(f"⚠️ ไม่พบไฟล์โมเดล: {filename} สำหรับ {label_name}")
        return None

    last_error = None
    for path in paths:
        try:
            return joblib.load(path)
        except Exception as e:
            last_error = f"{path}: {e}"

    st.warning(
        f"⚠️ โหลดโมเดล {label_name} ล้มเหลว "
        "เนื่องจากไฟล์ .pkl ถูกสร้างด้วย scikit-learn รุ่นที่ต่างจากเครื่องนี้ "
        "หรือไฟล์เสีย/ไม่ครบ กรุณาทำการ train ใหม่"
    )
    if last_error:
        st.code(last_error)
    return None

# Fallback models: ใช้งานได้ทันทีแม้ไม่มีโมเดลจริง
class FallbackBudgetRegressor:
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict("records")
        else:
            rows = X

        outputs = []
        for row in rows:
            camp_type = str(row.get("Campaign_Type", "Seasonal Sale"))
            channel = str(row.get("Channel", "Social Media"))
            audience = str(row.get("Target_Audience", "General Public"))
            duration = float(row.get("duration_days", 14))
            budget = 0.0

            # heuristic base
            if camp_type in ["Product Launch", "Brand Awareness"]:
                budget += 220000
            elif camp_type == "Flash Sale":
                budget += 130000
            elif camp_type == "Discount Promotion":
                budget += 160000
            else:
                budget += 180000

            channel_bonus = {
                "Social Media": 40000,
                "Online Store": 55000,
                "In-Store Branch": 35000,
                "Shopee / Lazada": 65000,
                "Line OA": 30000,
            }.get(channel, 20000)

            audience_bonus = {
                "Students & Education": 30000,
                "Gamers & Creators": 50000,
                "Corporate / Office": 45000,
                "General Public": 20000,
                "DIY PC Builders": 42000,
            }.get(audience, 20000)

            budget += channel_bonus + audience_bonus + duration * 5500
            outputs.append(max(10000.0, budget))

        return np.array(outputs, dtype=float)

class FallbackChannelClassifier:
    classes_ = np.array(["Social Media", "Online Store", "In-Store Branch", "Shopee / Lazada", "Line OA"])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict("records")
        else:
            rows = X

        probs = []
        for row in rows:
            budget = float(row.get("Budget", 250000))
            duration = float(row.get("duration_days", 14))
            channel_scores = {
                "Social Media": 0.30,
                "Online Store": 0.25,
                "In-Store Branch": 0.20,
                "Shopee / Lazada": 0.15,
                "Line OA": 0.10,
            }

            if duration <= 7:
                channel_scores["Shopee / Lazada"] += 0.15
                channel_scores["Social Media"] += 0.05
            if budget >= 400000:
                channel_scores["In-Store Branch"] += 0.10
                channel_scores["Online Store"] += 0.08
            if budget < 150000:
                channel_scores["Social Media"] += 0.10

            total = sum(channel_scores.values())
            p = [channel_scores[c] / total for c in self.classes_]
            probs.append(p)
        return np.array(probs, dtype=float)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array([self.classes_[i] for i in np.argmax(proba, axis=1)])

class FallbackClusterKMeans:
    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        preds = []
        for row in arr:
            budget = float(row[0])
            duration = float(row[1])
            if budget >= 350000 and duration >= 20:
                preds.append(0)
            elif budget >= 150000 or duration >= 10:
                preds.append(1)
            else:
                preds.append(2)
        return np.array(preds, dtype=int)

@st.cache_resource
def load_all_models():
    budget_model = load_model_or_none("campaign_budget_model.pkl", "Budget Estimator") or FallbackBudgetRegressor()
    channel_model = load_model_or_none("channel_recommender_model.pkl", "Channel Recommender") or FallbackChannelClassifier()
    cluster_pkg = load_model_or_none("campaign_cluster_model.pkl", "Campaign Clustering")
    if cluster_pkg is None:
        cluster_pkg = {
            "scaler": StandardScaler().fit(np.array([[10000, 1], [200000, 10], [500000, 25], [2000000, 60]], dtype=float)),
            "kmeans": FallbackClusterKMeans(),
            "tier_names": {
                0: "Tier 1: High Budget & Long Duration (แคมเปญระดับ Flagship/ระยะยาว)",
                1: "Tier 2: Mid-Tier Standard Campaigns (แคมเปญตามรอบปกติ/ระดับกลาง)",
                2: "Tier 3: Quick Tactical & Flash Sales (แคมเปญกระตุ้นยอดด่วน/ระยะสั้น)"
            }
        }
    return budget_model, channel_model, cluster_pkg

# ensure StandardScaler exists
from sklearn.preprocessing import StandardScaler

budget_model, channel_model, cluster_pkg = load_all_models()

# สร้าง 3 Tabs สำหรับการใช้งานแต่ละโมเดล
tab1, tab2, tab3 = st.tabs([
    "💰 ประมาณการงบประมาณ (Budget Estimator)",
    "📢 แนะนำช่องทางและกลยุทธ์ (Channel Recommender)",
    "📊 จัดกลุ่ม Portfolio แคมเปญ (Campaign Clustering)",
])

# -------------------------------------------------------------
# TAB 1: โมเดลที่ 1 - ทำนายงบประมาณ (Regression)
# -------------------------------------------------------------
with tab1:
    st.subheader("📋 ระบุรายละเอียดเพื่อประเมินงบประมาณ")

    # ไม่ต้อง st.stop() อีกแล้ว
    col1, col2 = st.columns(2)
    with col1:
        campaign_type_t1 = st.selectbox(
            "ประเภทแคมเปญ",
            ["Seasonal Sale", "Product Launch", "Brand Awareness", "Discount Promotion", "Flash Sale"],
            key="t1_camp_type",
        )
        channel_t1 = st.selectbox(
            "ช่องทางประชาสัมพันธ์",
            ["Social Media", "Online Store", "In-Store Branch", "Shopee / Lazada", "Line OA"],
            key="t1_channel",
        )

    with col2:
        target_audience_t1 = st.selectbox(
            "กลุ่มเป้าหมาย",
            ["Students & Education", "Gamers & Creators", "Corporate / Office", "General Public", "DIY PC Builders"],
            key="t1_target",
        )
        today = datetime.date.today()
        start_date_t1 = st.date_input("วันที่เริ่มต้น", today, key="t1_start")
        end_date_t1 = st.date_input("วันที่สิ้นสุด", today + datetime.timedelta(days=14), key="t1_end")

    if end_date_t1 < start_date_t1:
        st.error("⚠️ วันที่สิ้นสุดต้องอยู่หลังวันที่เริ่มต้น")
    else:
        duration_days_t1 = (end_date_t1 - start_date_t1).days
        start_month_t1 = start_date_t1.month
        start_quarter_t1 = (start_date_t1.month - 1) // 3 + 1
        start_dayofweek_t1 = start_date_t1.weekday()

        st.caption(f"⏱️ ระยะเวลาจัดแคมเปญ: **{duration_days_t1} วัน** | ไตรมาส: **Q{start_quarter_t1}**")

        if st.button("🚀 คำนวณงบประมาณ (Predict Budget)", key="btn_t1"):
            input_df = pd.DataFrame([{
                "Campaign_Type": campaign_type_t1,
                "Channel": channel_t1,
                "Target_Audience": target_audience_t1,
                "duration_days": duration_days_t1,
                "start_month": start_month_t1,
                "start_quarter": start_quarter_t1,
                "start_dayofweek": start_dayofweek_t1,
            }])

            predicted_budget = float(budget_model.predict(input_df)[0])
            st.success("### ผลการประเมินงบประมาณ")
            st.metric(label="งบประมาณที่แนะนำ (Estimated Budget)", value=f"{predicted_budget:,.2f} บาท")
            st.caption(f"💡 กรอบงบประมาณที่ปลอดภัย: **{max(0, predicted_budget - 60000):,.2f} - {predicted_budget + 60000:,.2f} บาท**")

# -------------------------------------------------------------
# TAB 2: โมเดลที่ 2 - แนะนำช่องทางโฆษณา (Classification)
# -------------------------------------------------------------
with tab2:
    st.subheader("💡 แนะนำช่องทางการตลาดและจัดสรรงบประมาณที่คุ้มค่าที่สุด")

    # ไม่ต้อง st.stop() อีกแล้ว
    col3, col4 = st.columns(2)
    with col3:
        campaign_type_t2 = st.selectbox(
            "ประเภทแคมเปญที่คุณต้องการจัด",
            ["Seasonal Sale", "Product Launch", "Brand Awareness", "Discount Promotion", "Flash Sale"],
            key="t2_camp_type",
        )
        target_audience_t2 = st.selectbox(
            "กลุ่มเป้าหมายหลัก",
            ["Students & Education", "Gamers & Creators", "Corporate / Office", "General Public", "DIY PC Builders"],
            key="t2_target",
        )

    with col4:
        budget_t2 = st.number_input(
            "งบประมาณที่มีในใจ (บาท)",
            min_value=10000,
            max_value=2000000,
            value=250000,
            step=10000,
            key="t2_budget",
        )
        duration_days_t2 = st.slider(
            "ระยะเวลาจัดแคมเปญ (วัน)",
            min_value=1,
            max_value=60,
            value=14,
            key="t2_duration",
        )

    if st.button("🔍 วิเคราะห์และแนะนำช่องทาง (Recommend Channel)", key="btn_t2"):
        input_data = pd.DataFrame([{
            "Campaign_Type": campaign_type_t2,
            "Target_Audience": target_audience_t2,
            "Budget": budget_t2,
            "duration_days": duration_days_t2,
        }])

        classes = channel_model.classes_
        probabilities = channel_model.predict_proba(input_data)[0]

        result_df = pd.DataFrame({
            "Channel": classes,
            "Probability": probabilities,
        }).sort_values(by="Probability", ascending=False).reset_index(drop=True)

        top_channel = result_df.iloc[0]["Channel"]
        top_prob = result_df.iloc[0]["Probability"] * 100

        st.markdown("---")
        st.success(f"### 🏆 ช่องทางที่แนะนำอันดับ 1: **{top_channel}** (ความเหมาะสม {top_prob:.1f}%)")

        st.write("📊 **สัดส่วนความเหมาะสมในแต่ละช่องทาง:**")
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for idx, row in result_df.iterrows():
            prob_percent = row["Probability"] * 100
            st.write(f"{medals[idx]} **{row['Channel']}**: {prob_percent:.1f}%")
            st.progress(float(row["Probability"]))

        st.markdown("---")
        st.info("💡 **คำแนะนำเชิงกลยุทธ์ (Strategic Tip):**")
        if target_audience_t2 == "DIY PC Builders" or top_channel == "In-Store Branch":
            st.markdown("- สำหรับกลุ่ม **DIY PC Builders** หรือหน้าร้านสาขา ควรจัดพนักงานแนะนำสเปกพร้อมโปรโมชันของแถมเมื่อประกอบครบชุด เพื่อกระตุ้น Basket Size")
        elif target_audience_t2 == "Gamers & Creators":
            st.markdown("- กลุ่ม **Gamers & Creators** ตอบสนองดีต่อรีวิวเชิงลึก การยิงผ่าน Social Media ร่วมกับจัด Bundle Set การ์ดจอ/มอนิเตอร์บน Online Store จะช่วยดันยอดได้ดี")
        elif campaign_type_t2 == "Flash Sale" or top_channel == "Shopee / Lazada":
            st.markdown("- แคมเปญระยะสั้นประเภท **Flash Sale / Marketplace** ควรเน้นแจกคูปองลดจำกัดเวลา เพื่อปิดการขายให้รวดเร็วที่สุด")
        elif target_audience_t2 == "Corporate / Office":
            st.markdown("- ลูกค้าองค์กรและออฟฟิศ ให้ความสำคัญกับใบกำกับภาษีและการรับประกันแบบ On-site Service การใช้ Line OA หรือทีมขายเฉพาะจะปิดดีลได้แม่นยำกว่า")
        else:
            st.markdown(f"- แนะนำกระจายงบหลักไปที่ **{top_channel}** และแบ่งงบสำรอง 15-20% ยิงรีมาร์เก็ตติ้งผ่านช่องทางอันดับ 2 ({result_df.iloc[1]['Channel']})")

# -------------------------------------------------------------
# TAB 3: โมเดลที่ 3 - จัดกลุ่ม Tier แคมเปญ (K-Means Clustering)
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 วิเคราะห์และจัดกลุ่ม Portfolio แคมเปญด้วย K-Means Clustering")

    # ไม่ต้อง st.stop() อีกแล้ว
    col5, col6 = st.columns(2)
    with col5:
        budget_t3 = st.number_input(
            "งบประมาณที่วางแผนไว้ (Budget ในหน่วยบาท)",
            min_value=10000,
            max_value=2000000,
            value=300000,
            step=10000,
            key="t3_budget",
        )
    with col6:
        duration_days_t3 = st.slider(
            "ระยะเวลาที่จัดแคมเปญ (Duration ในหน่วยวัน)",
            min_value=1,
            max_value=60,
            value=15,
            key="t3_dur",
        )

    st.caption(f"📌 สรุปเงื่อนไข: งบประมาณ **{budget_t3:,.2f} บาท** | ระยะเวลา **{duration_days_t3} วัน**")

    if st.button("🚀 จัดกลุ่ม Tier แคมเปญ (Analyze Cluster)", key="btn_t3"):
        scaler = cluster_pkg["scaler"]
        kmeans = cluster_pkg["kmeans"]
        tier_names = cluster_pkg["tier_names"]

        input_data_t3 = np.array([[budget_t3, duration_days_t3]])
        input_scaled_t3 = scaler.transform(input_data_t3)

        cluster_id = int(kmeans.predict(input_scaled_t3)[0])
        tier_label = tier_names.get(cluster_id, f"Cluster {cluster_id}")

        st.markdown("---")
        st.success(f"### 🎯 ผลการประเมิน: จัดอยู่ในกลุ่ม **{tier_label}**")

        st.subheader("💡 คำแนะนำเชิงกลยุทธ์สำหรับผู้บริหารและฝ่ายการตลาด:")
        if cluster_id == 0:
            st.info("""
            * **กลยุทธ์หลัก:** เป็นแคมเปญระดับ Flagship / Big Event ประจำปี เน้นสร้างการรับรู้แบรนด์ในวงกว้างและสร้างยอดขายก้อนใหญ่
            * **การเตรียมตัว:** ต้องประสานงานฝ่ายจัดซื้อล่วงหน้าอย่างน้อย 2-3 สัปดาห์ เพื่อสำรองสต็อกสินค้าหลัก เช่น ซีพียู, การ์ดจอ และโน้ตบุ๊กเกมมิ่ง
            * **ช่องทางแนะนำ:** จัดงานทั้งหน้าร้านใหญ่ (Flagship Stores) ควบคู่ Live Streaming และสื่อดิจิทัลทุกช่องทาง
            """)
        elif cluster_id == 1:
            st.info("""
            * **กลยุทธ์หลัก:** เป็นแคมเปญโปรโมชันระดับกลางตามรอบเทศกาล (Seasonal / Back to School)
            * **การเตรียมตัว:** เน้นการจัดชุด Bundle Set (เช่น ซื้อเคสแถมพาวเวอร์ซัพพลาย หรือเซตเกมมิ่งเกียร์) เพื่อดัน Basket Size ต่อยอดบิลให้สูงขึ้น
            * **ช่องทางแนะนำ:** Social Media, หน้าร้านสาขา และ JIB Online
            """)
        else:
            st.info("""
            * **กลยุทธ์หลัก:** เป็นแคมเปญ Tactical / Flash Sale เน้นความเร็วในการปิดยอดขายและระบายสต็อกสินค้า
            * **การเตรียมตัว:** เน้นตั้งราคาส่วนลดกระแทกใจ (Aggressive Pricing) พร้อมคูปองจำกัดเวลา เพื่อกระตุ้นให้ลูกค้าตัดสินใจซื้อทันที
            * **ช่องทางแนะนำ:** Marketplace เช่น Shopee, Lazada และ TikTok Shop
            """)

        with st.expander("🔍 ดูข้อมูลประสิทธิภาพของโมเดล (Model Performance)"):
            st.markdown("""
            * **Algorithm:** K-Means Clustering ($k=3$)
            * **Optimal $k$ Selection:** ตรวจสอบด้วย **Elbow Method** ร่วมกับ **Silhouette Analysis**
            * **Silhouette Score:** **0.433** (ยืนยันโครงสร้างการเกาะกลุ่มแน่นและแยกชั้นกันอย่างชัดเจน)
            * **เปรียบเทียบ:** ให้ประสิทธิภาพการแยกกลุ่มสูงกว่า Hierarchical Clustering (Silhouette = 0.395)
            """)