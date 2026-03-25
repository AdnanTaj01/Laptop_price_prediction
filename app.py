import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# Load files
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# ================= HEADER ================= #
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict laptop prices using Machine Learning 🚀</p>", unsafe_allow_html=True)

st.write("---")

# ================= LAYOUT ================= #

col1, col2 = st.columns(2)

# -------- LEFT COLUMN -------- #
with col1:
    st.subheader("🧾 Basic Information")

    company = st.selectbox('Brand', df['Company'].unique())
    type_name = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)

    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Display', ['No', 'Yes'])

# -------- RIGHT COLUMN -------- #
with col2:
    st.subheader("⚙️ Hardware Details")

    screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)

    resolution = st.selectbox(
        'Screen Resolution',
        ['1920x1080','1366x768','1600x900','3840x2160',
         '3200x1800','2880x1800','2560x1600',
         '2560x1440','2304x1440']
    )

    cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
    gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())

    hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
    ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])

    os = st.selectbox('Operating System', df['os'].unique())

st.write("---")

# ================= BUTTON ================= #
center_col = st.columns([1,2,1])

with center_col[1]:
    predict_btn = st.button("💰 Predict Price", type="primary", use_container_width=True)

# ================= PREDICTION ================= #

if predict_btn:

    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # PPI calculation
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }])

    prediction = pipe.predict(query)[0]

    final_price = int(np.exp(prediction))  # remove exp if not used in training

    st.write("")

    # Result Card
    st.markdown(f"""
        <div style="
            background-color:#1e1e1e;
            padding:30px;
            border-radius:15px;
            text-align:center;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
        ">
            <h2 style="color: #007BFF;">💰 Predicted Price</h2>
            <h1 style="color:white;">Rs {final_price:,}</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    h3 {
        color: #007BFF !important;
    }
    </style>
""", unsafe_allow_html=True)