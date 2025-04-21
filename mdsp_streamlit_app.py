
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Title
st.title("UTS & Elongation Predictor from MDSP Dataset")

# Load data
df = pd.read_csv("MDSP Dataset.csv")

# Group data to extract max stress and strain per (Thickness, Pattern)
grouped = df.groupby(['Thickness', 'Pattern']).agg({
    'Stress': 'max',
    'Strain': 'max'
}).reset_index()

grouped.rename(columns={'Stress': 'UTS_MPa', 'Strain': 'Elongation_percent'}, inplace=True)
grouped['Elongation_percent'] = grouped['Elongation_percent'] * 100

# Encode pattern
encoder = OneHotEncoder(sparse=False)
pattern_encoded = encoder.fit_transform(grouped[['Pattern']])
pattern_df = pd.DataFrame(pattern_encoded, columns=encoder.get_feature_names_out(['Pattern']))

# Combine features
X = pd.concat([grouped[['Thickness']].reset_index(drop=True), pattern_df], axis=1)

# Scale thickness
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled['Thickness'] = scaler.fit_transform(X[['Thickness']])

# Target values
y_uts = grouped['UTS_MPa'].values
y_elong = grouped['Elongation_percent'].values

# Train models
model_uts = RandomForestRegressor()
model_elong = RandomForestRegressor()
model_uts.fit(X_scaled, y_uts)
model_elong.fit(X_scaled, y_elong)

# User inputs
user_thickness = st.number_input("Enter Layer Thickness (mm)", min_value=0.1, max_value=1.0, step=0.01, value=0.24)
user_pattern = st.selectbox("Select Infill Pattern", encoder.categories_[0].tolist())

# Prepare input
user_pattern_encoded = encoder.transform([[user_pattern]])
user_input_df = pd.DataFrame(user_pattern_encoded, columns=encoder.get_feature_names_out(['Pattern']))
user_input_df.insert(0, 'Thickness', user_thickness)
user_input_df['Thickness'] = scaler.transform(user_input_df[['Thickness']])

# Predict
if st.button("Predict"):
    pred_uts = model_uts.predict(user_input_df)[0]
    pred_elong = model_elong.predict(user_input_df)[0]
    st.success(f"Predicted UTS: {pred_uts:.2f} MPa")
    st.success(f"Predicted Elongation: {pred_elong:.2f} %")

    # Plot stress-strain graph (dark theme)
    strain_vals = np.linspace(0, pred_elong / 100, 300)
    stress_vals = np.piecewise(strain_vals,
        [strain_vals < 0.02, strain_vals < 0.06, strain_vals >= 0.06],
        [lambda x: (pred_uts / 0.02) * x,
         lambda x: -200 * (x - 0.02)**2 + pred_uts,
         lambda x: -300 * (x - 0.06) + pred_uts * 0.9])
    stress_vals = np.maximum(stress_vals, 0)

    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.plot(strain_vals * 100, stress_vals, color='cyan', linewidth=2)
    ax.set_xlabel("Strain (%)", color='white')
    ax.set_ylabel("Stress (MPa)", color='white')
    ax.set_title("Stress-Strain Curve", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
