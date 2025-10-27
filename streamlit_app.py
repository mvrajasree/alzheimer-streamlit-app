import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os

--- 1. CONFIGURATION AND MODEL LOADING ---

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'multimodal_nn_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'clinical_scaler.joblib')

Definitive list of 35 clinical features (matching training)

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
"""Loads the multimodal model and StandardScaler once."""
try:
with tf.device('/cpu:0'):
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
return model, scaler
except Exception as e:
st.error(f"❌ Error loading model or scaler. Ensure files exist in '{MODEL_DIR}'.\n\nDetails: {e}")
return None, None

--- 2. PREPROCESSING FUNCTIONS ---

def preprocess_image(image_bytes):
"""Preprocess MRI image as done during model training."""
file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Resize
img = cv2.resize(img, (128, 128))

# Denoising
img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# CLAHE enhancement
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Normalize and reshape
img = img.astype('float32') / 255.0
return np.expand_dims(img, axis=0)


def preprocess_clinical(user_input_map, scaler, features):
"""
Converts user input to standardized NumPy array using fitted scaler.
Handles mismatched feature names automatically.
"""
# Get feature names expected by the scaler
scaler_features = list(getattr(scaler, "feature_names_in_", features))

# Create base DataFrame (fill missing with 1.0)
df = pd.DataFrame(np.ones((1, len(scaler_features))), columns=scaler_features)

# Map user inputs to correct columns (case-insensitive)
lower_map = {c.lower(): c for c in scaler_features}
for user_feature, value in user_input_map.items():
    match_key = user_feature.lower()
    if match_key in lower_map:
        df[lower_map[match_key]] = value
    else:
        st.warning(f"⚠️ Feature '{user_feature}' not found in scaler feature list; skipping.")

# Scale safely
try:
    df_scaled = pd.DataFrame(
        scaler.transform(df[scaler_features]),
        columns=scaler_features
    )
except Exception as e:
    st.error(f"Scaler transformation failed: {e}")
    st.write("Expected by scaler:", scaler_features)
    st.write("Given columns:", df.columns.tolist())
    raise e

return df_scaled.values

--- 3. STREAMLIT APP ---

st.set_page_config(page_title="Alzheimer's Predictor", layout="centered")
st.title("🧠 Alzheimer's Multimodal Risk Predictor")
st.markdown("---")

model, scaler = load_assets()

if model and scaler:
# Sidebar inputs
with st.sidebar:
st.header("Patient Clinical Data (Simplified)")
age = st.slider("1. Age", 50, 90, 70, key='age')
gender = st.selectbox("2. Gender", ["Male", "Female"], key='gender')
bmi = st.number_input("3. BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1, key='bmi')
apoe_e4 = st.selectbox("4. APOE E4 Status", ["Negative", "Positive"], key='apoe')
mri_hippocampal_volume = st.slider("5. Hippocampal Volume (mm³)", 500.0, 5000.0, 3000.0, step=10.0, key='hipvol')

    st.markdown("_Note: Remaining unlisted features are placeholder-normalized._")

    # Map to model feature names
    user_input_map = {
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'BMI': bmi,
        'APOEE4Status': 1 if apoe_e4 == "Positive" else 0,
        'MRIHippocampalVolume': mri_hippocampal_volume,
    }

# Main App Body
st.header("MRI Scan Upload")
uploaded_file = st.file_uploader("Upload T1-weighted MRI Scan (.jpg or .png)", type=['jpg', 'png'])

if st.button("Run Multimodal Prediction", type="primary", use_container_width=True):
    if uploaded_file is not None:
        with st.spinner("Analyzing clinical and image data..."):
            # 1. Preprocess
            image_array = preprocess_image(uploaded_file)
            clinical_array = preprocess_clinical(user_input_map, scaler, CLINICAL_FEATURES)

            # 2. Predict
            predictions = model.predict({
                'image_input': image_array,
                'clinical_input': clinical_array
            })

            prob_non_demented = predictions[0][0]
            prob_demented = predictions[0][1]

            # 3. Display Results
            st.subheader("Final Prediction")
            prediction_class = "Demented/MCI" if prob_demented > 0.5 else "NonDemented"

            if prediction_class == "Demented/MCI":
                st.error(f"## 🚨 Predicted Class: {prediction_class} (High Risk)")
            else:
                st.success(f"## ✅ Predicted Class: {prediction_class} (Low Risk)")

            col1, col2 = st.columns(2)
            col1.metric("Confidence (NonDemented)", f"{prob_non_demented * 100:.1f}%")
            col2.metric("Confidence (Demented/MCI)", f"{prob_demented * 100:.1f}%")

            st.markdown("---")
            st.image(Image.open(uploaded_file), caption="Uploaded MRI Scan", use_column_width=True)
    else:
        st.warning("Please upload an MRI scan image to proceed with the prediction.")


else:
st.error("Model or scaler could not be loaded. Please check your files and try again.")
