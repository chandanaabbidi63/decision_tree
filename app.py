# app.py - Decision Tree Regression for Diabetes Prediction

import streamlit as st
import pickle
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🩺",
    layout="centered"
)

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained model from pickle file."""
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Load model with error handling
try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file 'model.pkl' not found. Please ensure the model file exists in the same directory as this app.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Page title and description
st.title("🩺 Diabetes Progression Predictor")
st.markdown("""
This application uses a **Decision Tree Regression** model to predict diabetes progression 
based on patient health metrics. The model was trained on the scikit-learn diabetes dataset.
""")

st.divider()

# Input section
st.subheader("Patient Health Metrics")
st.write("Enter the following values to get a prediction:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age", 
        value=0.05,
        help="Age of the patient (normalized value)",
        format="%.4f"
    )
    bmi = st.number_input(
        "BMI", 
        value=0.05,
        help="Body Mass Index (normalized value)",
        format="%.4f"
    )
    s1 = st.number_input(
        "S1 (TC)", 
        value=0.05,
        help="Total Cholesterol (normalized value)",
        format="%.4f"
    )
    s3 = st.number_input(
        "S3 (HDL)", 
        value=0.05,
        help="High-Density Lipoproteins (normalized value)",
        format="%.4f"
    )
    s5 = st.number_input(
        "S5 (TCH)", 
        value=0.05,
        help="Thyroid Stimulating Hormone (normalized value)",
        format="%.4f"
    )

with col2:
    sex = st.number_input(
        "Sex", 
        value=0.05,
        help="Sex of the patient (normalized value)",
        format="%.4f"
    )
    bp = st.number_input(
        "Blood Pressure", 
        value=0.05,
        help="Blood Pressure (normalized value)",
        format="%.4f"
    )
    s2 = st.number_input(
        "S2 (LDL)", 
        value=0.05,
        help="Low-Density Lipoproteins (normalized value)",
        format="%.4f"
    )
    s4 = st.number_input(
        "S4 (TCH)", 
        value=0.05,
        help="Thyroid Stimulating Hormone (normalized value)",
        format="%.4f"
    )
    s6 = st.number_input(
        "S6 (Glucose)", 
        value=0.05,
        help="Blood Sugar Level (normalized value)",
        format="%.4f"
    )

st.divider()

# Prediction button
if st.button("🔮 Predict Diabetes Progression", type="primary", use_container_width=True):
    
    # Prepare input data
    input_data = np.array([
        [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]
    ])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.subheader("Prediction Result")
    
    # Create metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Predicted Progression",
            value=f"{prediction[0]:.2f}",
            help="Higher values indicate more severe diabetes progression"
        )
    
    # Interpretation
    st.markdown("### 📊 Interpretation")
    if prediction[0] < 100:
        st.info("🟢 **Low Risk**: The predicted diabetes progression is relatively low.")
    elif prediction[0] < 200:
        st.warning("🟡 **Moderate Risk**: The predicted diabetes progression is moderate. Consult a healthcare provider.")
    else:
        st.error("🔴 **High Risk**: The predicted diabetes progression is high. Please consult a healthcare provider immediately.")
    
    st.markdown("""
    ---
    *Note: This is a machine learning prediction for educational purposes only. 
    Always consult with a qualified healthcare professional for medical advice.*
    """)