
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "employee_attrition.csv"
MODEL_FILE = BASE_DIR / "models" / "knn_model.pkl"

st.set_page_config(
    page_title="Employee Attrition KNN",
    page_icon="👥",
    layout="wide"
)

st.title("Employee Attrition Prediction")
st.caption("Machine Learning project using K-Nearest Neighbors")

st.warning(
    "Educational demonstration only. This synthetic dataset and model "
    "are not suitable for real employee or hiring decisions."
)

if not DATA_FILE.exists():
    st.error("employee_attrition.csv was not found in the project folder.")
    st.stop()

df = pd.read_csv(DATA_FILE)

col1, col2, col3 = st.columns(3)
col1.metric("Dataset Records", len(df))
col2.metric("Attrition: Yes", int((df["Attrition"] == "Yes").sum()))
col3.metric("Attrition: No", int((df["Attrition"] == "No").sum()))

tab1, tab2 = st.tabs(["Data Dashboard", "KNN Prediction"])

with tab1:
    st.subheader("Sample Dataset")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Attrition Distribution")
    counts = df["Attrition"].value_counts().rename_axis(
        "Attrition"
    ).to_frame("Employees")
    st.bar_chart(counts)

    st.subheader("Average Monthly Income by Attrition")
    income = df.groupby("Attrition")["MonthlyIncome"].mean()
    st.bar_chart(income)

with tab2:
    if not MODEL_FILE.exists():
        st.error("Model not found. Run python train_model.py first.")
        st.stop()

    model = joblib.load(MODEL_FILE)

    st.subheader("Enter Example Employee Details")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            age = st.slider("Age", 18, 65, 32)
            income = st.number_input(
                "Monthly Income",
                min_value=10000,
                max_value=200000,
                value=35000,
                step=1000
            )
            satisfaction = st.select_slider(
                "Job Satisfaction",
                options=[1, 2, 3, 4],
                value=3
            )

        with right:
            years = st.slider("Years at Company", 0, 40, 5)
            balance = st.select_slider(
                "Work-Life Balance",
                options=[1, 2, 3, 4],
                value=3
            )
            overtime = st.selectbox("Overtime", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        sample = pd.DataFrame([{
            "Age": age,
            "MonthlyIncome": income,
            "JobSatisfaction": satisfaction,
            "YearsAtCompany": years,
            "WorkLifeBalance": balance,
            "OverTime": overtime
        }])

        prediction = model.predict(sample)[0]

        st.subheader("Prediction Result")

        if prediction == "Yes":
            st.warning("KNN prediction: Attrition = Yes")
        else:
            st.success("KNN prediction: Attrition = No")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            result = pd.DataFrame({
                "Class": model.classes_,
                "Model Score": probabilities
            }).set_index("Class")

            st.write("Model scores (not calibrated risk estimates)")
            st.bar_chart(result)

        st.caption(
            "This prediction is for learning only. It is not a reliable "
            "assessment of any real employee's future behavior."
        )

with st.expander("How KNN works"):
    st.write(
        "KNN compares a new example with similar examples in the training "
        "data. This project uses the three nearest neighbors and predicts "
        "the majority class. Numerical features are standardized and "
        "OverTime is one-hot encoded before distance calculations."
    )
