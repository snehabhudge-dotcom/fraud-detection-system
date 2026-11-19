import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/best_model.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Machine Learning Based Fraud Detection")

st.info(
    "Enter transaction details to check whether "
    "a transaction is potentially fraudulent."
)

st.subheader("Transaction Details")

time = st.number_input(
    "Transaction Time",
    min_value=0.0,
    value=0.0
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

st.subheader("Transaction Features")

features = {}

cols = st.columns(4)

for i in range(1, 29):
    with cols[(i - 1) % 4]:
        features[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.4f"
        )

if st.button("Check Transaction", type="primary"):

    input_data = {
        "Time": time,
        **features,
        "Amount": amount
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Fraud Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠️ Potentially Fraudulent Transaction")
    else:
        st.success("✅ Predicted as Normal Transaction")

    st.caption(
        "Educational demonstration only. "
        "This prediction is not a real banking decision."
    )
    # Model Comparison Dashboard

st.divider()

st.header("📊 Model Comparison Dashboard")

comparison = pd.read_csv("models/model_comparison.csv")

st.dataframe(comparison, use_container_width=True)

st.subheader("Algorithm Performance")

st.bar_chart(
    comparison.set_index("Algorithm")[
        ["Precision", "Recall", "F1 Score", "PR-AUC"]
    ]
)
# Model Comparison Dashboard

st.divider()

st.header("📊 Model Comparison Dashboard")

comparison = pd.read_csv("models/model_comparison.csv")

st.dataframe(comparison, use_container_width=True)

st.subheader("Algorithm Performance")

st.bar_chart(
    comparison.set_index("Algorithm")[
        ["Precision", "Recall", "F1 Score", "PR-AUC"]
    ]
)# Test with a Sample Transaction
from sklearn.model_selection import train_test_split

st.divider()
st.header("🧪 Test with Sample Transaction")

if st.button("Test Sample Transaction"):

    df = pd.read_csv("dataset/cleaned_creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Recreate the same test split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Select one transaction from the test dataset
    sample = X_test.iloc[[0]]
    actual = int(y_test.iloc[0])

    sample_model = joblib.load("models/best_model.pkl")
    prediction = int(sample_model.predict(sample)[0])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Model Prediction: Fraud Transaction")
    else:
        st.success("Model Prediction: Normal Transaction")

    st.write("Actual Result:", "Fraud" if actual == 1 else "Normal")

    if prediction == actual:
        st.success("Prediction matches the actual result!")
    else:
        st.warning("Prediction does not match the actual result.")