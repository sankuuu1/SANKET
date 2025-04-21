import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.title("UTS & Elongation Predictor")

# Load dataset
df = pd.read_csv("MDSP Dataset.csv")

# Prepare summary data
grouped = df.groupby(['Thickness', 'Pattern']).agg({
    'Stress': 'max',
    'Strain': 'max'
}).reset_index()

grouped.rename(columns={'Stress': 'UTS_MPa', 'Strain': 'Elongation_percent'}, inplace=True)
grouped['Elongation_percent'] *= 100

# Encode pattern
encoder = OneHotEncoder(sparse=False)
pattern_encoded = encoder.fit_transform(grouped[['Pattern']])
pattern_df = pd.DataFrame(pattern_encoded, columns=encoder.get_feature_names_out(['Pattern']))

# Combine inputs
X = pd.concat([grouped[['Thickness']], pattern_df], axis=1)
scaler = StandardScaler()
X['Thickness'] = scaler.fit_transform(X[['Thickness']])

# Targets
y_uts = grouped['UTS_MPa'].values
y_elong = grouped['Elongation_percent'].values

# Train models
model_uts = RandomForestRegressor()
model_elong = RandomForestRegressor()
model_uts.fit(X, y_uts)
model_elong.fit(X, y_elong)

# --- User Inputs ---
user_thickness = st.number_input("Enter Layer Thickness (mm)", min_value=0.1, max_value=1.0, step=0.01, value=0.24)
user_pattern = st.selectbox("Select Infill Pattern", encoder.categories_[0].tolist())

# Prepare input
user_pattern_encoded = encoder.transform([[user_pattern]])
user_input = pd.DataFrame(user_pattern_encoded, columns=encoder.get_feature_names_out(['Pattern']))
user_input.insert(0, 'Thickness', user_thickness)
user_input['Thickness'] = scaler.transform(user_input[['Thickness']])

# --- Predict & Plot ---
if st.button("Predict"):
    uts = model_uts.predict(user_input)[0]
    elong = model_elong.predict(user_input)[0]

    st.success(f"Predicted UTS: {uts:.2f} MPa")
    st.success(f"Predicted Elongation: {elong:.2f} %")

    strain_vals = np.linspace(0, elong / 100, 300)
    stress_vals = np.piecewise(strain_vals,
        [strain_vals < 0.02, strain_vals < 0.06, strain_vals >= 0.06],
        [lambda x: (uts / 0.02) * x,
         lambda x: -200 * (x - 0.02)**2 + uts,
         lambda x: -300 * (x - 0.06) + uts * 0.9])
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
