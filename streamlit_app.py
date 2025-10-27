import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# Load model and scaler
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -------------------------------
# Preprocessing function (FIXED)
# -------------------------------
def preprocess_clinical(user_input_map, scaler, features):
    """
    Converts user input to standardized NumPy array using the fitted scaler.
    Automatically aligns feature names and fills missing ones safely.
    """
    # 1. Get the exact feature names expected by the scaler
    scaler_features = list(getattr(scaler, "feature_names_in_", features))

    # 2. Create a DataFrame with placeholders (1.0) for all expected features
    df = pd.DataFrame(np.ones((1, len(scaler_features))), columns=scaler_features)

    # 3. Normalize both user inputs and scaler feature names for mapping
    lower_map = {c.lower(): c for c in scaler_features}

    # 4. Map user inputs (case-insensitive)
    for user_feature, value in user_input_map.items():
        match_key = user_feature.lower()
        if match_key in lower_map:
            df.at[0, lower_map[match_key]] = value
        else:
            # Show in sidebar only once per unmatched feature
            st.sidebar.info(f"ℹ️ Ignored unmatched feature: '{user_feature}'")

    # 5. Transform only the correct columns
    df_scaled = pd.DataFrame(
        scaler.transform(df[scaler_features]),
        columns=scaler_features
    )

    # 6. Return as NumPy array
    return df_scaled.values

# -------------------------------
# Define clinical features (optional)
# -------------------------------
CLINICAL_FEATURES = [
    "BMI", "AlcoholConsumption", "CardiovascularDisease", "BehavioralProblems",
    "Cholesterol", "CognitiveTestScore", "Diabetes", "DietQuality", "ADL",
    "ExerciseFrequency", "HeartRate", "Hypertension", "KidneyFunction",
    "LiverFunction", "MedicationCount", "MemoryComplaints", "Mood",
    "NeuropsychiatricSymptoms", "PhysicalActivity", "ReactionTime",
    "SleepQuality", "SmokingStatus", "SocialEngagement", "StressLevel",
    "ThyroidFunction", "VitaminDLevel", "VisionProblems", "B12Level",
    "HearingProblems", "APOEe4Status", "Age", "EducationLevel", "Sex"
]

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Alzheimer’s Risk Predictor", layout="wide")
st.title("🧠 Alzheimer’s Risk Prediction App")

st.write("Enter the patient’s clinical data below to estimate Alzheimer’s risk.")

# Create sidebar input fields dynamically
user_input_map = {}
for feature in CLINICAL_FEATURES:
    val = st.sidebar.number_input(f"{feature}", min_value=0.0, value=1.0, step=0.1)
    user_input_map[feature] = val

if st.button("Predict Risk"):
    try:
        # Preprocess
        clinical_array = preprocess_clinical(user_input_map, scaler, CLINICAL_FEATURES)
        # Predict
        prediction = model.predict(clinical_array)[0]
        probability = model.predict_proba(clinical_array)[0, 1]

        st.success(f"🩺 Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        st.progress(float(probability))
        st.write(f"Confidence: {probability*100:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("© 2025 Alzheimer’s Prediction AI • Built with Streamlit & Scikit-Learn")
