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

@st.cache_resource
def load_assets():
"""Loads model, scaler, and feature list once."""
try:
with tf.device('/cpu:0'):
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
CLINICAL_FEATURES = [
'age', 'gender_numeric', 'education_level', 'bmi', 'systolic_bp',
'diastolic_bp', 'heart_rate', 'cholesterol', 'triglycerides', 'hdl',
'ldl', 'smoking_status_numeric', 'alcohol_consumption', 'physical_activity_level',
'sleep_quality', 'stress_level', 'family_history_alzheimers_numeric',
'apoe_e4_status_numeric', 'hba1c_level', 'cortisol_level', 'vitamin_d_level',
'b12_level', 'folate_level', 'thyroid_hormone_level', 'crp_level',
'eeg_alpha_power', 'eeg_beta_power', 'eeg_theta_power', 'eeg_delta_power',
'mri_hippocampal_volume', 'mri_cortical_thickness', 'mri_ventricular_size',
'mri_amygdala_volume', 'mri_white_matter_lesion_load', 'diabetes_status_numeric'
]
return model, scaler, CLINICAL_FEATURES
except Exception as e:
st.error(f"❌ Error loading model or scaler: {e}")
return None, None, None

--- 2. PREPROCESSING FUNCTIONS ---

def preprocess_image(image_bytes):
"""Apply the same preprocessing (resize, CLAHE, denoise, normalize) as training."""
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

def preprocess_clinical(user_input_dict, scaler, features):
"""
Converts user input to standardized NumPy array using the fitted scaler.
Automatically aligns feature names, ignoring mismatched names safely.
"""
scaler_features = list(getattr(scaler, "feature_names_in_", features))
df = pd.DataFrame(np.ones((1, len(scaler_features))), columns=scaler_features)
lower_map = {f.lower(): f for f in scaler_features}

for k, v in user_input_dict.items():
    key = k.lower()
    if key in lower_map:
        df.at[0, lower_map[key]] = v
    else:
        st.sidebar.warning(f"Ignored unmatched feature: '{k}'")

df_scaled = pd.DataFrame(
    scaler.transform(df[scaler_features]),
    columns=scaler_features
)

return df_scaled.values

--- 3. STREAMLIT APP ---

st.set_page_config(page_title="🧠 Alzheimer's Predictor", layout="centered")
st.title("🧠 Alzheimer's Multimodal Risk Predictor")
st.markdown("---")

model, scaler, CLINICAL_FEATURES = load_assets()

if model and scaler:
with st.sidebar:
st.header("🧬 Patient Clinical Data")

    age = st.slider("Age", 50, 90, 70)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
    apoe_e4 = st.selectbox("APOE E4 Status", ["Negative", "Positive"])
    mri_hippocampal_volume = st.slider("MRI Hippocampal Volume (mm³)", 500.0, 5000.0, 3000.0, step=10.0)

    st.markdown("_Other clinical features are auto-filled with baseline mean values._")

    # Use scaler means for missing features
    try:
        scaler_means = scaler.mean_
    except AttributeError:
        scaler_means = np.ones(len(CLINICAL_FEATURES))

    mock_input = scaler_means.copy()
    mock_input[0] = age
    mock_input[1] = 1 if gender == "Male" else 0
    mock_input[3] = bmi
    mock_input[17] = 1 if apoe_e4 == "Positive" else 0
    mock_input[29] = mri_hippocampal_volume

    clinical_input_dict = dict(zip(CLINICAL_FEATURES, mock_input))

    with st.expander("🔍 Show clinical input data"):
        st.dataframe(pd.DataFrame([clinical_input_dict]))


# --- Main Body ---
st.header("🧩 MRI Scan Upload")
uploaded_file = st.file_uploader("Upload T1-weighted MRI Scan (.jpg or .png)", type=['jpg', 'png'])

# ✅ MRI-only toggle
test_mode = st.radio(
    "Select Test Mode:",
    ["Full Multimodal (MRI + Clinical Data)", "MRI-Only (Use Clinical Baseline)"],
    horizontal=True
)

if st.button("Run Prediction", type="primary", use_container_width=True):
    if uploaded_file is not None:
        with st.spinner("Analyzing MRI and clinical data..."):

            # --- 1. Preprocess MRI ---
            image_array = preprocess_image(uploaded_file)

            # --- 2. Clinical input depending on mode ---
            if test_mode == "Full Multimodal (MRI + Clinical Data)":
                clinical_array = preprocess_clinical(
                    clinical_input_dict,
                    scaler,
                    CLINICAL_FEATURES
                )
            else:
                st.info("🧠 MRI-only mode active — using baseline clinical profile.")
                baseline_vector = scaler.mean_.reshape(1, -1)
                clinical_array = baseline_vector

            # --- 3. Predict ---
            predictions = model.predict({
                'image_input': image_array,
                'clinical_input': clinical_array
            })

            prob_non_demented = predictions[0][0]
            prob_demented = predictions[0][1]
            prediction_class = "Demented/MCI" if prob_demented > 0.5 else "NonDemented"

            # --- 4. Display ---
            st.subheader("Prediction Results")

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
        st.warning("Please upload an MRI scan to proceed.")


else:
st.error("❌ Model or scaler failed to load. Please verify the files exist in the 'models' folder.")
