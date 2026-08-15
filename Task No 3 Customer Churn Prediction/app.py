from __future__ import annotations

import streamlit as st

from src.predict import predict_customer


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
)


@st.cache_data
def get_feature_defaults():
    return {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 50000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 80000.0,
    }


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


st.title("Bank Customer Churn Prediction")
st.markdown(
    "Predict whether a customer is likely to churn based on account and demographic data."
)

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=get_feature_defaults()["CreditScore"],
            step=1,
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Spain", "Germany"],
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        age = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=get_feature_defaults()["Age"],
            step=1,
        )

        tenure = st.slider(
            "Tenure (years)",
            min_value=0,
            max_value=10,
            value=get_feature_defaults()["Tenure"],
            step=1,
        )

    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=get_feature_defaults()["Balance"],
            step=500.0,
        )

        num_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4],
        )

        has_cr_card = st.selectbox(
            "Has Credit Card",
            [0, 1],
        )

        is_active_member = st.selectbox(
            "Is Active Member",
            [0, 1],
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=get_feature_defaults()["EstimatedSalary"],
            step=1000.0,
        )

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    try:
        customer_input = {
            "CreditScore": safe_int(credit_score),
            "Geography": geography,
            "Gender": gender,
            "Age": safe_int(age),
            "Tenure": safe_int(tenure),
            "Balance": safe_float(balance),
            "NumOfProducts": safe_int(num_products),
            "HasCrCard": safe_int(has_cr_card),
            "IsActiveMember": safe_int(is_active_member),
            "EstimatedSalary": safe_float(estimated_salary),
        }

        prediction, probability = predict_customer(customer_input)

        risk_label = "High Risk" if probability >= 0.5 else "Low Risk"

        status_color = (
            "#d9534f" if risk_label == "High Risk" else "#2ca02c"
        )

        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div style="
                padding: 1rem;
                border-radius: 0.75rem;
                background-color: {status_color};
                color: white;
                font-size: 1.25rem;
                font-weight: bold;
            ">
                Predicted Churn: {prediction}
                | Risk: {risk_label}
                | Probability: {probability:.2%}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if prediction == "Yes":
            st.warning(
                "This customer is likely to churn. "
                "Consider retention strategies or proactive outreach."
            )
        else:
            st.success(
                "This customer appears stable and has a lower likelihood of churn."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Please verify all fields are complete and valid before submitting."
        )


st.sidebar.title("Project Info")
st.sidebar.markdown("- Objective: predict customer churn")
st.sidebar.markdown("- Model: trained machine learning pipeline")
st.sidebar.markdown("- Data source: Kaggle bank customer churn dataset")