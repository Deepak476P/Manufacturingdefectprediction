import streamlit as st
import pandas as pd
import joblib
import math
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Manufacturing",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("🏭 Smart Manufacturing Defect Prediction")

st.markdown(
    """
    **Predict machine failure using manufacturing parameters and machine learning.**

    The system uses a **Random Forest classifier with feature selection**
    to estimate the probability of machine failure.
    """
)

st.divider()


# ============================================================
# SIDEBAR — MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.markdown(
        """
        **Algorithm**

        Random Forest

        **Feature Selection**

        Mutual Information + RFE

        **Input Features**

        • Air Temperature  
        • Process Temperature  
        • Rotational Speed  
        • Torque  
        • Tool Wear
        """
    )

    st.divider()

    st.subheader("📊 Model Performance")

    st.metric("Accuracy", "96.40%")
    st.metric("Recall", "79.41%")
    st.metric("ROC-AUC", "97.32%")

    st.divider()

    st.caption(
        "Model trained using the UCI AI4I 2020 "
        "Predictive Maintenance Dataset."
    )


# ============================================================
# MACHINE INPUTS
# ============================================================

st.subheader("⚙️ Machine Parameters")

st.write(
    "Type the new sample's operating conditions into the boxes, "
    "or use + and - to change a value by 1, then click Predict Machine Failure."
)

# Use a form so typing values does not trigger a rerun
# after every individual input.
# Numeric starting values keep the + and - buttons enabled.
with st.form("machine_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=0.0,
            value=298.0,
            step=1.0,
            format="%.2f",
            key="air_temperature"
        )

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=0.0,
            value=308.0,
            step=1.0,
            format="%.2f",
            key="process_temperature"
        )

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=0,
            value=1500,
            step=1,
            format="%d",
            key="rotational_speed"
        )

    with col2:

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            value=40.0,
            step=1.0,
            format="%.2f",
            key="torque"
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            value=100,
            step=1,
            format="%d",
            key="tool_wear"
        )

    st.write("")

    predict_button = st.form_submit_button(
        "🔍 Predict Machine Failure",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Check every field before passing the new sample to the model.
    values = [air_temperature, process_temperature, rotational_speed, torque, tool_wear]
    if any(value is None or not math.isfinite(value) or value < 0 for value in values):
        st.error("Please enter a valid, non-negative number in all five fields.")
        st.stop()

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error("⚠️ MACHINE FAILURE PREDICTED")

            st.write(
                "The model predicts that the machine is at "
                "risk of failure under the given conditions."
            )

        else:

            st.success("✅ NO MACHINE FAILURE PREDICTED")

            st.write(
                "The model predicts that the machine is unlikely "
                "to fail under the given conditions."
            )

    with result_col2:

        st.metric(
            "Failure Probability",
            f"{probability:.2%}"
        )

        st.progress(float(probability))


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    if probability < 0.30:

        risk_level = "LOW"
        risk_message = (
            "The estimated failure probability is relatively low."
        )

    elif probability < 0.60:

        risk_level = "MEDIUM"
        risk_message = (
            "The machine shows a moderate estimated failure risk. "
            "Monitoring is recommended."
        )

    else:

        risk_level = "HIGH"
        risk_message = (
            "The machine shows a high estimated failure risk. "
            "Inspection or preventive maintenance is recommended."
        )


    st.subheader("🚦 Risk Assessment")

    risk_col1, risk_col2 = st.columns([1, 2])

    with risk_col1:
        st.metric("Risk Level", risk_level)

    with risk_col2:
        st.info(risk_message)


    # ========================================================
    # CURRENT MACHINE STATUS
    # ========================================================

    st.divider()

    st.subheader("📋 Current Machine Status")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Air Temp",
            f"{air_temperature:.2f} K"
        )

    with c2:
        st.metric(
            "Process Temp",
            f"{process_temperature:.2f} K"
        )

    with c3:
        st.metric(
            "Speed",
            f"{rotational_speed} rpm"
        )

    with c4:
        st.metric(
            "Torque",
            f"{torque:.2f} Nm"
        )

    with c5:
        st.metric(
            "Tool Wear",
            f"{tool_wear} min"
        )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.divider()

    st.subheader("🔎 Model Feature Importance")

    try:

        rf_model = model.named_steps["model"]

        feature_names = [
            "Air Temperature",
            "Process Temperature",
            "Rotational Speed",
            "Torque",
            "Tool Wear"
        ]

        importances = rf_model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(
            "Importance",
            ascending=True
        )

        st.bar_chart(
            importance_df.set_index("Feature")
        )

        most_important = importance_df.iloc[-1]["Feature"]

        st.caption(
            f"Most influential feature: **{most_important}**"
        )

    except Exception:

        st.info(
            "Feature importance visualization is unavailable."
        )
