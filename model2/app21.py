import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="JIB Marketing Channel Recommender", 
    page_icon="🎯", 
    layout="centered"
)

st.title("🎯 JIB Marketing Channel Recommender")
st.markdown("ระบบ AI วิเคราะห์และแนะนำช่องทางการตลาดที่เหมาะสมที่สุดสำหรับ J.I.B. Computer Group")

# โหลดเฉพาะโมเดล Channel Recommender
@st.cache_resource
def load_channel_model():
    model_paths = [
        'channel_recommender_model.pkl',
        'model2/channel_recommender_model.pkl',
        '../channel_recommender_model.pkl'
    ]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

channel_model = load_channel_model()

if channel_model is None:
    st.error("⚠️ ไม่พบไฟล์โมเดล `channel_recommender_model.pkl` กรุณารันไฟล์เทรนโมเดลก่อน")
    st.stop()

# -------------------------------------------------------------
# ส่วนฟอร์มรับค่าข้อมูลแคมเปญ
# -------------------------------------------------------------
st.subheader("📋 ระบุรายละเอียดแคมเปญการตลาด")

col1, col2 = st.columns(2)

with col1:
    campaign_type = st.selectbox(
        "ประเภทแคมเปญที่คุณต้องการจัด",
        ['Seasonal Sale', 'Product Launch', 'Brand Awareness', 'Discount Promotion', 'Flash Sale'],
        key="camp_type"
    )
    target_audience = st.selectbox(
        "กลุ่มเป้าหมายหลัก",
        ['Students & Education', 'Gamers & Creators', 'Corporate / Office', 'General Public', 'DIY PC Builders'],
        key="target_audience"
    )

with col2:
    budget = st.number_input(
        "งบประมาณแคมเปญ (บาท)",
        min_value=10000,
        max_value=2000000,
        value=250000,
        step=10000,
        key="budget"
    )
    duration_days = st.slider(
        "ระยะเวลาจัดแคมเปญ (วัน)",
        min_value=1,
        max_value=60,
        value=14,
        key="duration_days"
    )

st.markdown("---")

# -------------------------------------------------------------
# ส่วนการพยากรณ์และแสดงผลลัพธ์
# -------------------------------------------------------------
if st.button("🚀 วิเคราะห์และแนะนำช่องทาง (Recommend Channel)", use_container_width=True):
    input_data = pd.DataFrame([{
        'Campaign_Type': campaign_type,
        'Target_Audience': target_audience,
        'Budget': budget,
        'duration_days': duration_days
    }])

    # ดึงค่า Classes และเปอร์เซ็นต์ความน่าจะเป็นจาก Random Forest
    classes = channel_model.classes_
    probabilities = channel_model.predict_proba(input_data)[0]

    # จัดอันดับช่องทางจากความเหมาะสมมากไปน้อย
    result_df = pd.DataFrame({
        'Channel': classes,
        'Probability': probabilities
    }).sort_values(by='Probability', ascending=False).reset_index(drop=True)

    top_channel = result_df.iloc[0]['Channel']
    top_prob = result_df.iloc[0]['Probability'] * 100

    st.success(f"### 🏆 ช่องทางที่แนะนำอันดับ 1: **{top_channel}** (คะแนนความเหมาะสม {top_prob:.1f}%)")

    st.write("📊 **สัดส่วนคะแนนความเหมาะสมในแต่ละช่องทาง:**")
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for idx, row in result_df.iterrows():
        prob_percent = row['Probability'] * 100
        st.write(f"{medals[idx]} **{row['Channel']}**: {prob_percent:.1f}%")
        st.progress(float(row['Probability']))

    # ข้อเสนอแนะเชิงกลยุทธ์ (Strategic Tip)
    st.markdown("---")
    st.info("💡 **คำแนะนำเชิงกลยุทธ์ (Strategic Tip):**")
    if target_audience == 'DIY PC Builders' or top_channel == 'In-Store Branch':
        st.markdown("- สำหรับกลุ่ม **DIY PC Builders** หรือหน้าร้านสาขา ควรจัดพนักงานเชี่ยวชาญสเปกคอยให้คำปรึกษา พร้อมโปรโมชันของแถมเมื่อประกอบครบชุด เพื่อกระตุ้น Basket Size")
    elif target_audience == 'Gamers & Creators':
        st.markdown("- กลุ่ม **Gamers & Creators** ตอบสนองดีต่อรีวิวและการทดสอบสตรีม การเน้นโฆษณาผ่าน Social Media ร่วมกับจัด Bundle Set มอนิเตอร์/การ์ดจอ จะช่วยปิดยอดได้เร็ว")
    elif campaign_type == 'Flash Sale' or top_channel == 'Shopee / Lazada':
        st.markdown("- แคมเปญระยะสั้นประเภท **Flash Sale / Marketplace** ควรเน้นแจกคูปองลดจำกัดเวลา เพื่อกระตุ้นการตัดสินใจซื้อในทันที")
    elif target_audience == 'Corporate / Office':
        st.markdown("- ลูกค้าองค์กรและออฟฟิศ ให้ความสำคัญกับใบกำกับภาษีเต็มรูปแบบและการรับประกันแบบ On-site การสื่อสารผ่าน Line OA หรือทีมขายตรงจะได้ผลลัพธ์ดีที่สุด")
    else:
        st.markdown(f"- แนะนำเน้นสัดส่วนงบประมาณหลักไปที่ **{top_channel}** และสำรองงบประมาณ 15–20% สำหรับทำ Retargeting ผ่านช่องทางอันดับ 2 (**{result_df.iloc[1]['Channel']}**)")