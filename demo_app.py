import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ---- Load saved models and scaler ----
nb_model = joblib.load("data/naive_bayes_model.pkl")
lr_model = joblib.load("data/log_reg_calibrated.pkl")
scaler = joblib.load("data/feature_scaler.pkl")

st.set_page_config(page_title="Exoplanet Habitability Predictor",  layout="centered")

st.title(" Exoplanet Habitability Predictor")
st.markdown("### Predict if an exoplanet could be habitable based on its physical and stellar properties.")
st.write("Enter the parameters below:")

# ---- Input fields ----
pl_rade = st.number_input("🌎 Planet Radius (Earth Radii)", min_value=0.1, max_value=100.0, value=1.0)
pl_bmasse = st.number_input("💫 Planet Mass (Earth Masses)", min_value=0.1, max_value=10000.0, value=1.0)
pl_orbsmax = st.number_input("🪩 Semi-Major Axis (AU)", min_value=0.001, max_value=1000.0, value=1.0)
pl_insol = st.number_input("☀️ Insolation Flux (Earth = 1)", min_value=0.0000001, max_value=10000.0, value=1.0)
st_teff = st.number_input("⭐ Stellar Temperature (K)", min_value=2000.0, max_value=20000.0, value=5778.0)
st_rad = st.number_input("🌞 Stellar Radius (Solar Radii)", min_value=0.1, max_value=1000.0, value=1.0)


# ---- Prepare data ----
input_data = np.array([[pl_rade, pl_bmasse, pl_orbsmax, pl_insol, st_teff, st_rad]])
input_scaled = scaler.transform(input_data)

# ---- Predict ----
if st.button(" Predict Habitability"):
    prob_nb = nb_model.predict_proba(input_scaled)[0][1]
    raw_prob_lr = lr_model.predict_proba(input_scaled)[0][1]
    prob_lr = float(np.clip(raw_prob_lr, 0.0, 1.0))  # ensures valid range


    st.subheader(" Prediction Results")
    st.write(f"**Naïve Bayes:** {prob_nb:.4f} probability of being habitable")
    st.write(f"**Logistic Regression (Balanced):** {prob_lr:.4f} probability of being habitable")

    st.progress(int(prob_lr * 100))
    st.markdown("---")

    if prob_lr >= 0.5 or prob_nb >= 0.5:
        st.success(" This planet is *likely habitable!*")
    else:
        st.error("🪐 This planet is *likely non-habitable.*")

st.markdown("---")
st.caption("Developed by Aditya — Statistical Machine Learning Project (Exoplanet Habitability Prediction)")
