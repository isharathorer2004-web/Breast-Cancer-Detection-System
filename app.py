import streamlit as st
import numpy as np
import joblib

# PAGE CONFIG

st.set_page_config(
    page_title="Breast Cancer Detector",
    page_icon="🌸",
    layout="wide"
)


# LOAD MODEL

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# CUSTOM CSS

st.markdown("""
<style>

.main {
    background-color: #FFF5FA;
}

.title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#d63384;
}

.subtitle {
    text-align:center;
    color:#6f42c1;
    font-size:18px;
    margin-bottom:20px;
}

.stButton > button {
    width:100%;
    height:55px;
    border-radius:15px;
    border:none;
    background:linear-gradient(90deg,#ff69b4,#c77dff);
    color:white;
    font-size:20px;
    font-weight:bold;
}

.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:25px;
    font-weight:bold;
}

.metric-card {
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown(
    "<div class='title'>🌸 Breast Cancer Detection System 🌸</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-Powered Healthcare Classification using Logistic Regression 💖</div>",
    unsafe_allow_html=True
)

st.divider()


# SIDEBAR

st.sidebar.title("💗 About")

st.sidebar.info("""
This application detects whether a breast tumor is:

🌷 Benign (Non-Cancerous)

⚠️ Malignant (Cancerous)

using Machine Learning.
""")

st.sidebar.success("Model: Logistic Regression")


# EXPLANATION SECTION

with st.expander("ℹ️ What do these measurements mean?"):

    st.markdown("""
### How are these values obtained?

These features are extracted from breast cell images obtained through
Fine Needle Aspiration (FNA) and analyzed using image processing.

#### Feature Meaning

🌸 **Mean Radius**
- Average distance from center to boundary.

🌸 **Mean Texture**
- Variation in pixel intensity.

🌸 **Mean Perimeter**
- Average boundary length of the tumor.

🌸 **Mean Area**
- Average tumor size.

🌸 **Mean Smoothness**
- Measures smoothness of tumor edges.

💖 **Mean Compactness**
- Describes density and irregularity of shape.

💖 **Worst Radius**
- Largest radius observed.

💖 **Worst Texture**
- Highest texture variation observed.

💖 **Worst Perimeter**
- Maximum perimeter observed.

💖 **Worst Area**
- Largest tumor area observed.

⚠️ These values are usually obtained from pathology or imaging reports.
    """)

# INPUT SECTION

st.header("🩺 Enter Tumor Measurements")

col1, col2 = st.columns(2)

with col1:

    radius = st.number_input(
        "🌸 Mean Radius",
        min_value=0.0,
        value=14.0,
        help="Average distance from tumor center to boundary."
    )

    texture = st.number_input(
        "🌸 Mean Texture",
        min_value=0.0,
        value=20.0,
        help="Variation in pixel intensity."
    )

    perimeter = st.number_input(
        "🌸 Mean Perimeter",
        min_value=0.0,
        value=90.0,
        help="Average tumor boundary length."
    )

    area = st.number_input(
        "🌸 Mean Area",
        min_value=0.0,
        value=600.0,
        help="Average tumor area."
    )

    smoothness = st.number_input(
        "🌸 Mean Smoothness",
        min_value=0.0,
        value=0.10,
        help="Measures smoothness of tumor boundary."
    )

with col2:

    compactness = st.number_input(
        "💖 Mean Compactness",
        min_value=0.0,
        value=0.20,
        help="Shape density and irregularity."
    )

    worst_radius = st.number_input(
        "💖 Worst Radius",
        min_value=0.0,
        value=17.0,
        help="Largest radius observed."
    )

    worst_texture = st.number_input(
        "💖 Worst Texture",
        min_value=0.0,
        value=25.0,
        help="Highest texture variation observed."
    )

    worst_perimeter = st.number_input(
        "💖 Worst Perimeter",
        min_value=0.0,
        value=110.0,
        help="Maximum perimeter observed."
    )

    worst_area = st.number_input(
        "💖 Worst Area",
        min_value=0.0,
        value=900.0,
        help="Largest tumor area observed."
    )


# PREDICTION

if st.button("💖 Predict Cancer Status"):

    features = np.array([[
        radius,
        texture,
        perimeter,
        area,
        smoothness,
        compactness,
        worst_radius,
        worst_texture,
        worst_perimeter,
        worst_area
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    st.divider()

    if prediction == 1:

        st.markdown(f"""
        <div class='result-box'
        style='background:#d8f3dc;color:#2d6a4f'>
        🌷 BENIGN TUMOR DETECTED
        <br><br>
        Confidence: {probability[1]*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class='result-box'
        style='background:#ffd6e8;color:#d63384'>
        ⚠️ MALIGNANT TUMOR DETECTED
        <br><br>
        Confidence: {probability[0]*100:.2f}%
        </div>
        """, unsafe_allow_html=True)


# PERFORMANCE SECTION

st.divider()

st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "97%")
col2.metric("Precision", "96%")
col3.metric("Recall", "98%")
col4.metric("ROC-AUC", "99%")


# FOOTER

st.divider()

st.markdown(
"""
<center>
💜 Developed by Isha Rathore <br>
Artificial Intelligence & Machine Learning Internship Project 🌸
</center>
""",
unsafe_allow_html=True
)
