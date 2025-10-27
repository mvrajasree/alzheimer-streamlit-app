import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os

# --- 1. CONFIGURATION AND MODEL LOADING ---

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'multimodal_nn_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'clinical_scaler.joblib')

# Case-sensitive list used by model/scaler
CLINICAL_FEATURES = [
    'Age', 'Gender', 'EducationLevel', 'BMI', 'SystolicBP',
    'DiastolicBP', 'HeartRate', 'Cholesterol', 'Triglycerides', 'HDL',
    'LDL', 'SmokingStatus', 'AlcoholConsumption', 'PhysicalActivityLevel',
    'SleepQuality', 'StressLevel', 'FamilyHistoryAlzheimers',
    'APOEE4Status', 'HBA1CLevel', 'CortisolLevel', 'VitaminDLevel',
    'B12Level', 'FolateLevel', 'ThyroidHormoneLevel', 'CRPLevel',
    'EEGAlphaPower', 'EEGBetaPower', 'EEGThetaPower', 'EEGDeltaPower',
    'MRIHippocampalVolume', 'MRICorticalThickness', 'MRIVentricularSize',
    'MRIAmygdalaVolume', 'MRIWhiteMatterLesionLoad', 'DiabetesStatus'
]


@st.cache_resource
def load_assets():
    """Loads the trained multimodal model and scaler once."""
    try:
        with tf.device('/cpu:0'):
            model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        return None, None


# --- 2. PREPROCESSING FUNCTIONS ---

def preprocess_image(image_bytes):
    """Apply the same preprocessing as in training."""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_clinical(user_input_map, scaler, features):
    """
    Converts user input to standardized NumPy array using the fitted scaler.
    Automatically aligns feature names and fills missing ones safely.
    """
    scaler_features = list(getattr(scaler, "feature_names_in_", features))
    df = pd.DataFrame(np.ones((1, len(scaler_features))), columns=scaler_features)
    lower_map = {c.lower(): c for c in scaler_features}

    for user_feature, value in user_input_map.items():
        match_key = user_feature.lower()
        if match_key in lower_map:
            df.at[0, lower_map[match_key]] = value
        else:
            st.sidebar.info(f"ℹ️ Ignored unmatched feature: '{user_feature}'")

    df_scaled = pd.DataFrame(
        scaler.transform(df[scaler_features]),
        columns=scaler_features
    )

    return df_scaled.values


# --- 3. STREAMLIT APP UI ---

st.set_page_config(page_title="🧠 Alzheimer's Predictor", layout="centered")
st.title("🧠 Alzheimer's Multimodal Risk Predictor")
st.markdown("---")

model, scaler = load_assets()

if model and scaler:

    # --- Sidebar for Clinical Inputs ---
    with st.sidebar:
        st.header("🧬 Patient Clinical Data")

        age = st.slider("Age", 50, 90, 70)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1)
        apoe_e4 = st.selectbox("APOE E4 Status", ["Negative", "Positive"])
        mri_hippocampal_volume = st.slider("MRI Hippocampal Volume (mm³)", 500.0, 5000.0, 3000.0, step=10.0)

        st.markdown("_Other clinical fields are automatically standardized._")

        user_input_map = {
            'Age': age,
            'Gender': 1 if gender == "Male" else 0,
            'BMI': bmi,
            'APOEE4Status': 1 if apoe_e4 == "Positive" else 0,
            'MRIHippocampalVolume': mri_hippocampal_volume,
        }

    # --- Main Section ---
    st.header("🧩 Upload MRI Scan")
    uploaded_file = st.file_uploader("Upload T1-weighted MRI Scan (.jpg or .png)", type=['jpg', 'png'])

    if st.button("Run Multimodal Prediction", type="primary", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Analyzing clinical and image data..."):

                # Preprocess both modalities
                image_array = preprocess_image(uploaded_file)
                clinical_array = preprocess_clinical(user_input_map, scaler, CLINICAL_FEATURES)

                # Model prediction
                predictions = model.predict({
                    'image_input': image_array,
                    'clinical_input': clinical_array
                })

                prob_non_demented = predictions[0][0]
                prob_demented = predictions[0][1]
                prediction_class = "Demented/MCI" if prob_demented > 0.5 else "NonDemented"

                # --- Display Results ---
                st.subheader("🧾 Prediction Results")
                if prediction_class == "Demented/MCI":
                    st.error(f"## 🚨 Predicted Class: {prediction_class} (High Risk)")
                else:
                    st.success(f"## ✅ Predicted Class: {prediction_class} (Low Risk)")

                col1, col2 = st.columns(2)
                col1.metric("Confidence (NonDemented)", f"{prob_non_demented * 100:.1f}%")
                col2.metric("Confidence (Demented/MCI)", f"{prob_demented * 100:.1f}%")

                st.markdown("---")
                st.image(Image.open(uploaded_file), caption="Uploaded MRI Scan", use_container_width=True)
        else:
            st.warning("Please upload an MRI scan image before running the prediction.")
