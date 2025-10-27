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
# You will need to manually ensure the image preprocessing function (CLAHE, Denoising)
# is the same as the one used in training!

@st.cache_resource
def load_assets():
    """Loads the Multimodal Model and StandardScaler only once."""
    try:
        # Load the saved Keras model
        model = load_model(MODEL_PATH)
        
        # Load the fitted StandardScaler for clinical data
        scaler = joblib.load(SCALER_PATH)
        
        # The list of clinical features must be defined here, 
        # exactly as they were used in the aligned training data (35 features)
        # --- CORRECTED CLINICAL_FEATURES LIST ---
# --- CORRECTED CLINICAL_FEATURES LIST (35 features) ---
# NOTE: The capitalization and removal of '_numeric' suffixes must match the fitted scaler.
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
# NOTE: The list above is an educated guess based on common clinical feature naming 
# and the error message's 'Seen at fit time' section. You should check the column names 
# of the DataFrame you used right before saving the scaler to be 100% certain.
        
        return model, scaler, CLINICAL_FEATURES
    except Exception as e:
        st.error(f"Error loading models or scaler. Ensure files are in the '{MODEL_DIR}' folder. Error: {e}")
        return None, None, None

# --- 2. PREPROCESSING FUNCTIONS (MUST MATCH TRAINING) ---

def preprocess_image(image_bytes):
    """
    Applies the same preprocessing steps (Resize, CLAHE, Denoising) 
    as done during model training.
    """
    # Convert uploaded image bytes to NumPy array
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 1. Resize to 128x128 (Must match model input)
    img = cv2.resize(img, (128, 128))
    
    # 2. Denoising (Assuming you used a standard denoiser)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 3. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 4. Normalize and reshape for CNN input
    img = img.astype('float32') / 255.0
    # Add batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
    return np.expand_dims(img, axis=0)

def preprocess_clinical(user_input_dict, scaler, features):
    """Converts user input to a standardized NumPy array."""
    # 1. Convert dictionary to DataFrame
    df = pd.DataFrame([user_input_dict], columns=features)
    
    # 2. Apply the fitted scaler
    df[features] = scaler.transform(df[features])
    
    # 3. Convert to NumPy array for DNN input: (1, 35)
    return df.values

# --- 3. STREAMLIT APP LAYOUT ---

st.title("🧠 Alzheimer's Multimodal Risk Predictor")
st.markdown("---")

model, scaler, CLINICAL_FEATURES = load_assets()

if model and scaler:
    
    # --- Sidebar for Clinical Input (35 Features is too much, so we sample key inputs) ---
    with st.sidebar:
        st.header("Patient Clinical Data (Simplified)")
        
        # NOTE: In a real app, you would need all 35 features. We simplify here.
        
        # Gather key features
        age = st.slider("Age", 50, 90, 70)
        gender = st.selectbox("Gender", ["Male (1)", "Female (0)"])
        bmi = st.number_input("BMI (e.g., 25.0)", min_value=15.0, max_value=50.0, value=25.0)
        apoe_e4 = st.selectbox("APOE E4 Status", ["Negative (0)", "Positive (1)"])
        mri_hippocampal_volume = st.slider("Hippocampal Volume (mm³)", 500.0, 5000.0, 3000.0)
        
        # --- Create a mock dictionary with ALL 35 features, filling unknown with mean/zero ---
        # This is a critical simplification. The remaining features MUST be provided in the same format.
        # For simplicity, we are filling missing features with a dummy value (e.g., 0.0 or 1.0)
        # In a real app, these values would come from the full clinical profile.
        
        # Create a base dictionary of the 35 features filled with safe dummy values (e.g., mean)
        # To avoid error, all features must be present. We set a dummy value (like 1.0) for features not collected via input.
        
        # Create a mock array for all 35 features (using placeholder 1.0 for features not in sidebar)
        # This requires knowing the EXACT column order.
        mock_input = np.ones((1, len(CLINICAL_FEATURES)))[0]
        
        # Map simplified user inputs to their correct index in the 35-feature array
        # This requires knowing the index of each feature in CLINICAL_FEATURES
        
        # Example: Index 0=age, Index 1=gender_numeric, Index 3=bmi, Index 17=apoe_e4_status_numeric, Index 29=mri_hippocampal_volume
        
        # Mapped User Inputs
        mock_input[0] = age
        mock_input[1] = 1 if gender == "Male (1)" else 0
        mock_input[3] = bmi
        mock_input[17] = 1 if apoe_e4 == "Positive (1)" else 0
        mock_input[29] = mri_hippocampal_volume
        
        # Convert the mock array back to a dictionary matching feature names
        clinical_input_dict = dict(zip(CLINICAL_FEATURES, mock_input))
        
        
    # --- Main App Body for Image Input and Prediction ---
    
    st.header("MRI Scan Upload")
    uploaded_file = st.file_uploader("Upload T1-weighted MRI Scan (.jpg or .png)", type=['jpg', 'png'])

    if st.button("Run Prediction", type="primary"):
        if uploaded_file is not None:
            
            with st.spinner("Analyzing data and running multimodal prediction..."):
                
                # --- 1. Preprocess Data ---
                
                # Image Preprocessing
                image_array = preprocess_image(uploaded_file)
                
                # Clinical Preprocessing
                clinical_array = preprocess_clinical(
                    clinical_input_dict, 
                    scaler, 
                    CLINICAL_FEATURES
                )

                # --- 2. Run Multimodal Model ---
                
                # Model expects a dictionary of inputs: {'image_input': image_array, 'clinical_input': clinical_array}
                predictions = model.predict({
                    'image_input': image_array, 
                    'clinical_input': clinical_array
                })
                
                # Output is a probability array: [[prob_class_0, prob_class_1]]
                prob_non_demented = predictions[0][0]
                prob_demented = predictions[0][1]
                
                # --- 3. Display Results ---
                
                st.subheader("Prediction Results")
                
                if prob_demented > 0.5:
                    st.error(f"Prediction: **Demented/MCI** (High Risk)")
                    st.write(f"Confidence (Demented/MCI): **{prob_demented:.2f}**")
                    st.write(f"Confidence (NonDemented): **{prob_non_demented:.2f}**")
                    st.markdown("⚠️ **Action Recommended:** Consult a specialist for detailed diagnostic testing.")
                else:
                    st.success(f"Prediction: **NonDemented** (Low Risk)")
                    st.write(f"Confidence (NonDemented): **{prob_non_demented:.2f}**")
                    st.write(f"Confidence (Demented/MCI): **{prob_demented:.2f}**")
                    st.markdown("✅ **Monitor:** Continue regular check-ups and healthy lifestyle.")

                st.markdown("---")
                st.image(Image.open(uploaded_file), caption="Uploaded Scan (Pre-processed)", use_column_width=True)
                
        else:
            st.warning("Please upload an MRI scan image to run the prediction.")
