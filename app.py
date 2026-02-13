import streamlit as st
import joblib
import numpy as np

model = joblib.load('regression_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="House Price Predictor", page_icon="🏡")

st.title("🏡 House Price Prediction App")
st.write("กรอกข้อมูลรายละเอียดบ้านเพื่อทำนายราคาประเมิน")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("พื้นที่ (ตารางฟุต)", min_value=100, value=5000)
    bedrooms = st.number_input("จำนวนห้องนอน", min_value=1, value=3)
    bathrooms = st.number_input("จำนวนห้องน้ำ", min_value=1, value=2)

with col2:
    parking = st.number_input("ที่จอดรถ (คัน)",min_value=1,value=1)
    status = st.selectbox("สถานะการตกแต่ง", 
                          options=[0, 1, 2], 
                          format_func=lambda x: ['Unfurnished', 'Semi-Furnished', 'Furnished'][x])


if st.button("Predict Price", use_container_width=True):
    input_data = np.array([[area, bedrooms, bathrooms, parking, status]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.divider()
    st.subheader(f"ราคาประเมินคือ: ${prediction[0]:,.2f}")