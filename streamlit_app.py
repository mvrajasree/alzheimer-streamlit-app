import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import io

# --- 1. CONFIGURATION AND MODEL LOADING ---

# Define paths relative to the Streamlit app root
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'multimodal_nn_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'clinical_scaler.joblib')

# THE DEFINITIVE LIST OF 35 CLINICAL FEATURES 
# (Case-sensitive names extracted from the fitted scaler/model)
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

# Map simplified input names to their index in the 35-feature array
# This mapping assumes the features are in the exact order of CLINICAL_FEATURES
FEATURE_INDEX_MAP = {
    'Age': 0,
    'Gender': 1,
    'BMI': 3,
    'APOEE4Status': 17,
    'MRIHippocampalVolume': 29,
}


@st.cache_resource
def load_assets():
    """Loads the Multimodal Model and StandardScaler only once."""
    try:
        # Ensure TensorFlow uses Keras model
        with tf.device('/cpu:0'): # Load model on CPU to avoid GPU conflicts on some servers
            model = load_model(MODEL_PATH)
        
        scaler = joblib.load(SCALER_PATH)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models or scaler. Ensure files are in the '{MODEL_DIR}' folder. Error: {e}")
        return None, None


# --- 2. PREPROCESSING FUNCTIONS (MUST MATCH TRAINING) ---

def preprocess_image(image_bytes):
    """
    Applies the same preprocessing steps (Resize, CLAHE, Denoising, Normalization) 
    as done during model training.
    """
    # Read image using OpenCV
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 1. Resize to 128x128
    img = cv2.resize(img, (128, 128))
    
    # 2. Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 3. CLAHE 
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 4. Normalize and reshape: (1, 128, 128, 3)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_clinical(user_input_map, scaler, features):
    """Converts user input to a standardized NumPy array using the fitted scaler."""
    
    # 1. Create a base array of 1.0s (a dummy value to pass through standardization)
    # The scaler expects all 35 features, so we fill the uncollected ones with a placeholder.
    mock_input_array = np.ones((1, len(features)))[0]
    
    # 2. Map the simplified user inputs to their correct position in the array
    for feature_name, value in user_input_map.items():
        index = features.index(feature_name)
        mock_input_array[index] = value

    # 3. Convert the array back to a DataFrame with the CORRECT column names
    df = pd.DataFrame([mock_input_array], columns=features)
    
    # 4. Apply the fitted scaler (this is where the name validation happens)
    # The scaler MUST be transformed on all 35 columns
    df[features] = scaler.transform(df[features])
    
    # 5. Convert to NumPy array for DNN input: (1, 35)
    return df.values


# --- 3. STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Alzheimer's Predictor", layout="centered")
st.title("🧠 Alzheimer's Multimodal Risk Predictor")
st.markdown("---")

model, scaler = load_assets()

if model and scaler:
    
    # --- Sidebar for Clinical Input (Simplified) ---
    with st.sidebar:
        st.header("Patient Clinical Data (Simplified)")
        
        # Gather key features
        age = st.slider("1. Age", 50, 90, 70, key='age')
        gender = st.selectbox("2. Gender", ["Male", "Female"], key='gender')
        bmi = st.number_input("3. BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1, key='bmi')
        apoe_e4 = st.selectbox("4. APOE E4 Status", ["Negative", "Positive"], key='apoe')
        mri_hippocampal_volume = st.slider("5. Hippocampal Volume (mm³)", 500.0, 5000.0, 3000.0, step=10.0, key='hipvol')
        
        st.markdown("_Note: Full prediction requires 35 features. Unlisted values are normalized placeholders._")
        
        # Map user inputs to the exact feature names the scaler expects
        user_input_map = {
            'Age': age,
            'Gender': 1 if gender == "Male" else 0,
            'BMI': bmi,
            'APOEE4Status': 1 if apoe_e4 == "Positive" else 0,
            'MRIHippocampalVolume': mri_hippocampal_volume,
        }
        
        
    # --- Main App Body for Image Input and Prediction ---
    
    st.header("MRI Scan Upload")
    uploaded_file = st.file_uploader("Upload T1-weighted MRI Scan (.jpg or .png)", type=['jpg', 'png'])

    if st.button("Run Multimodal Prediction", type="primary", use_container_width=True):
        if uploaded_file is not None:
            
            with st.spinner("Analyzing clinical and image data..."):
                
                # --- 1. Preprocess Data ---
                
                # Image Preprocessing
                image_array = preprocess_image(uploaded_file)
                
                # Clinical Preprocessing
                clinical_array = preprocess_clinical(
                    user_input_map, 
                    scaler, 
                    CLINICAL_FEATURES
                )

                # --- 2. Run Multimodal Model ---
                
                predictions = model.predict({
                    'image_input': image_array, 
                    'clinical_input': clinical_array
                })
                
                # Output is a probability array: [[prob_class_0, prob_class_1]]
                prob_non_demented = predictions[0][0]
                prob_demented = predictions[0][1]
                
                # --- 3. Display Results ---
                
                st.subheader("Final Prediction")
                
                prediction_class = "Demented/MCI" if prob_demented > 0.5 else "NonDemented"
                
                # Use markdown for styling the final result
                if prediction_class == "Demented/MCI":
                    st.error(f"## 🚨 Predicted Class: {prediction_class} (High Risk)")
                else:
                    st.success(f"## ✅ Predicted Class: {prediction_class} (Low Risk)")
                    
                
                # Display confidence scores
                col1, col2 = st.columns(2)
                col1.metric("Confidence (NonDemented)", f"{prob_non_demented * 100:.1f}%")
                col2.metric("Confidence (Demented/MCI)", f"{prob_demented * 100:.1f}%")
                
                st.markdown("---")
                
                # Show the uploaded image
                st.image(Image.open(uploaded_file), caption="Uploaded Scan", use_column_width=True)
                
        else:
            st.warning("Please upload an MRI scan image to proceed with the prediction.")
