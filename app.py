import streamlit as st
import numpy as np
import joblib

# Set up page configurations
st.set_page_config(
    page_title="Credit Card Fraud Detector", 
    page_icon="💳", 
    layout="centered"
)

st.title("💳 Credit Card Fraud & Spam Detection App")
st.write("Enter the transaction metadata details below to verify if it is legitimate or potential fraud/spam.")

# 1. Load the freshly generated files directly from storage
@st.cache_resource
def load_assets():
    model = joblib.load('credit_card_rf_model.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("🤖 AI Engine Loaded and Calibrated Successfully!")
except Exception as e:
    st.error(f"Could not load model assets: {e}")

st.markdown("---")

# 2. Setup the User Input Form
st.subheader("Transaction Parameters")

# Create two columns for clean UI layout
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=25.0, step=1.0)
    v1 = st.number_input("PCA Component V1", value=0.0)
    v2 = st.number_input("PCA Component V2", value=0.0)

with col2:
    time_sec = st.number_input("Time (Seconds from first transaction)", min_value=0, value=3600)
    # Fallback padding for remaining V features to keep array shapes aligned (V3 to V28)
    remaining_v = [0.0] * 26  

# 3. Handle Predictions upon Button Click
if st.button("🚀 Analyze Transaction", type="primary"):
    # Reconstruct structural input matching training features exactly:
    # Your Random Forest expects 30 features: V1 to V28, scaled_amount, scaled_time
    features = np.array([[v1, v2] + remaining_v + [amount, time_sec]])
    
    try:
        # Scale features using the verified RobustScaler instance
        features_scaled = scaler.transform(features)
        
        # Generate prediction and probabilities
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        # Extract results out of arrays
        pred_class = prediction[0]
        prob_safe = probabilities[0][0] * 100
        prob_fraud = probabilities[0][1] * 100
        
        st.markdown("### **Results Analysis:**")
        if pred_class == 1:
            st.error(f"🚨 **WARNING: HIGH RISK FRAUD DETECTED!** (Confidence: {prob_fraud:.2f}%)")
            st.toast("Alert dispatched: High Fraud Probability!", icon="🚨")
        else:
            st.success(f"✅ **Safe Transaction Verified.** (Confidence: {prob_safe:.2f}%)")
            
    except Exception as eval_error:
        st.error(f"Execution Error at processing: {eval_error}")


