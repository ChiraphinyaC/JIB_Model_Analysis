import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib

st.set_page_config(page_title="JIB Marketing Strategy Hub", layout="centered")

st.title("🎯 JIB Marketing Campaign & Strategy Hub")
st.markdown("ระบบ AI วิเคราะห์และวางแผนกลยุทธ์การตลาดสำหรับ J.I.B. Computer Group")

# โหลดทั้ง 2 โมเดลเก็บไว้ใน Cache
@st.cache_resource
def load_all_models():
    budget_model = joblib.load('campaign_budget_model.pkl')
    channel_model = joblib.load('channel_recommender_model.pkl')
    return budget_model, channel_model

try:
    budget_model, channel_model = load_all_models()
except Exception as e:
    st.error("⚠️ ไม่พบไฟล์โมเดล กรุณารัน train_model.py และ train_channel_model.py ก่อน")
    st.stop()

# สร้าง 2 Tabs สำหรับหน้าเว็บ
tab1, tab2 = st.tabs([
    "💰 ประมาณการงบประมาณ (Budget Estimator)", 
    "📢 แนะนำช่องทางและกลยุทธ์ (Channel Recommender)"
])

# -------------------------------------------------------------
# TAB 1: โมเดลเดิม - ทำนายงบประมาณ (Regression)
# -------------------------------------------------------------
with tab1:
    st.subheader("📋 ระบุรายละเอียดเพื่อประเมินงบประมาณ")
    
    col1, col2 = st.columns(2)
    with col1:
        campaign_type_t1 = st.selectbox(
            "ประเภทแคมเปญ",
            ['Seasonal Sale', 'Product Launch', 'Brand Awareness', 'Discount Promotion', 'Flash Sale'],
            key="t1_camp_type"
        )
        channel_t1 = st.selectbox(
            "ช่องทางประชาสัมพันธ์",
            ['Social Media', 'Online Store', 'In-Store Branch', 'Shopee / Lazada', 'Line OA'],
            key="t1_channel"
        )

    with col2:
        target_audience_t1 = st.selectbox(
            "กลุ่มเป้าหมาย",
            ['Students & Education', 'Gamers & Creators', 'Corporate / Office', 'General Public', 'DIY PC Builders'],
            key="t1_target"
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
                'Campaign_Type': campaign_type_t1,
                'Channel': channel_t1,
                'Target_Audience': target_audience_t1,
                'duration_days': duration_days_t1,
                'start_month': start_month_t1,
                'start_quarter': start_quarter_t1,
                'start_dayofweek': start_dayofweek_t1
            }])

            predicted_budget = budget_model.predict(input_df)[0]
            st.success("### ผลการประเมินงบประมาณ")
            st.metric(label="งบประมาณที่แนะนำ (Estimated Budget)", value=f"{predicted_budget:,.2f} บาท")
            st.caption(f"💡 กรอบงบประมาณที่ปลอดภัย: **{max(0, predicted_budget - 60000):,.2f} - {predicted_budget + 60000:,.2f} บาท**")

# -------------------------------------------------------------
# TAB 2: โมเดลใหม่ - แนะนำช่องทางโฆษณา (Classification)
# -------------------------------------------------------------
with tab2:
    st.subheader("💡 แนะนำช่องทางการตลาดและจัดสรรงบประมาณที่คุ้มค่าที่สุด")
    
    col3, col4 = st.columns(2)
    with col3:
        campaign_type_t2 = st.selectbox(
            "ประเภทแคมเปญที่คุณต้องการจัด",
            ['Seasonal Sale', 'Product Launch', 'Brand Awareness', 'Discount Promotion', 'Flash Sale'],
            key="t2_camp_type"
        )
        target_audience_t2 = st.selectbox(
            "กลุ่มเป้าหมายหลัก",
            ['Students & Education', 'Gamers & Creators', 'Corporate / Office', 'General Public', 'DIY PC Builders'],
            key="t2_target"
        )

    with col4:
        budget_t2 = st.number_input(
            "งบประมาณที่มีในใจ (บาท)",
            min_value=10000,
            max_value=2000000,
            value=250000,
            step=10000,
            key="t2_budget"
        )
        duration_days_t2 = st.slider(
            "ระยะเวลาจัดแคมเปญ (วัน)",
            min_value=1,
            max_value=60,
            value=14,
            key="t2_duration"
        )

    if st.button("🔍 วิเคราะห์และแนะนำช่องทาง (Recommend Channel)", key="btn_t2"):
        input_data = pd.DataFrame([{
            'Campaign_Type': campaign_type_t2,
            'Target_Audience': target_audience_t2,
            'Budget': budget_t2,
            'duration_days': duration_days_t2
        }])

        # ดึง Classes และค่า Probability ออกมา
        classes = channel_model.classes_
        probabilities = channel_model.predict_proba(input_data)[0]

        # รวมผลลัพธ์และเรียงจากมากไปน้อย
        result_df = pd.DataFrame({
            'Channel': classes,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=False).reset_index(drop=True)

        top_channel = result_df.iloc[0]['Channel']
        top_prob = result_df.iloc[0]['Probability'] * 100

        st.markdown("---")
        st.success(f"### 🏆 ช่องทางที่แนะนำอันดับ 1: **{top_channel}** (ความเหมาะสม {top_prob:.1f}%)")

        st.write("📊 **สัดส่วนความเหมาะสมในแต่ละช่องทาง:**")
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for idx, row in result_df.iterrows():
            prob_percent = row['Probability'] * 100
            st.write(f"{medals[idx]} **{row['Channel']}**: {prob_percent:.1f}%")
            st.progress(float(row['Probability']))

        # Strategic Tip แนะนำตามกลุ่มเป้าหมายและช่องทาง
        st.markdown("---")
        st.info("💡 **คำแนะนำเชิงกลยุทธ์ (Strategic Tip):**")
        if target_audience_t2 == 'DIY PC Builders' or top_channel == 'In-Store Branch':
            st.markdown("- สำหรับกลุ่ม **DIY PC Builders** หรือหน้าร้านสาขา ควรจัดพนักงานแนะนำสเปกพร้อมโปรโมชันของแถมเมื่อประกอบครบชุด เพื่อกระตุ้น Basket Size")
        elif target_audience_t2 == 'Gamers & Creators':
            st.markdown("- กลุ่ม **Gamers & Creators** ตอบสนองดีต่อรีวิวเชิงลึก การยิงผ่าน Social Media ร่วมกับจัด Bundle Set การ์ดจอ/มอนิเตอร์บน Online Store จะช่วยดันยอดได้ดี")
        elif campaign_type_t2 == 'Flash Sale' or top_channel == 'Shopee / Lazada':
            st.markdown("- แคมเปญระยะสั้นประเภท **Flash Sale / Marketplace** ควรเน้นแจกคูปองลดจำกัดเวลา เพื่อปิดการขายให้รวดเร็วที่สุด")
        elif target_audience_t2 == 'Corporate / Office':
            st.markdown("- ลูกค้าองค์กรและออฟฟิศ ให้ความสำคัญกับใบกำกับภาษีและการรับประกันแบบ On-site Service การใช้ Line OA หรือทีมขายเฉพาะจะปิดดีลได้แม่นยำกว่า")
        else:
            st.markdown(f"- แนะนำกระจายงบหลักไปที่ **{top_channel}** และแบ่งงบสำรอง 15-20% ยิงรีมาร์เก็ตติ้งผ่านช่องทางอันดับ 2 ({result_df.iloc[1]['Channel']})")