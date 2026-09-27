import streamlit as st
import pandas as pd
import datetime
import joblib

st.set_page_config(page_title="JIB Budget Estimator", layout="centered")

st.title("🎯 JIB Campaign Budget Estimator")
st.markdown("ระบบ AI ช่วยประเมินงบประมาณแคมเปญการตลาดล่วงหน้า")

@st.cache_resource
def load_model():
    return joblib.load('campaign_budget_model.pkl')

model = load_model()

st.subheader("📋 ระบุรายละเอียดแคมเปญ")

col1, col2 = st.columns(2)
with col1:
    campaign_type = st.selectbox(
        "ประเภทแคมเปญ (Campaign Type)",
        ['Seasonal Sale', 'Product Launch', 'Brand Awareness', 'Discount Promotion', 'Flash Sale']
    )
    channel = st.selectbox(
        "ช่องทางโฆษณา (Channel)",
        ['Social Media', 'Online Store', 'In-Store Branch', 'Shopee / Lazada', 'Line OA']
    )

with col2:
    target_audience = st.selectbox(
        "กลุ่มเป้าหมาย (Target Audience)",
        ['Students & Education', 'Gamers & Creators', 'Corporate / Office', 'General Public', 'DIY PC Builders']
    )
    today = datetime.date.today()
    start_date = st.date_input("วันที่เริ่มต้น", today)
    end_date = st.date_input("วันที่สิ้นสุด", today + datetime.timedelta(days=14))

if end_date < start_date:
    st.error("⚠️ วันที่สิ้นสุดต้องอยู่หลังวันที่เริ่มต้น")
else:
    duration_days = (end_date - start_date).days
    start_month = start_date.month
    start_quarter = (start_date.month - 1) // 3 + 1
    start_dayofweek = start_date.weekday()

    st.write(f"⏱️ **ระยะเวลา:** {duration_days} วัน | **ไตรมาส:** Q{start_quarter}")

    if st.button("🚀 คำนวณงบประมาณ (Predict Budget)"):
        input_data = pd.DataFrame([{
            'Campaign_Type': campaign_type,
            'Channel': channel,
            'Target_Audience': target_audience,
            'duration_days': duration_days,
            'start_month': start_month,
            'start_quarter': start_quarter,
            'start_dayofweek': start_dayofweek
        }])

        predicted = model.predict(input_data)[0]

        st.success("### ผลการประเมินงบประมาณ")
        st.metric(label="งบประมาณแนะนำ (Estimated Budget)", value=f"{predicted:,.2f} บาท")
        st.caption(f"💡 กรอบงบประมาณที่ปลอดภัย: **{max(0, predicted - 60000):,.2f} - {predicted + 60000:,.2f} บาท**")