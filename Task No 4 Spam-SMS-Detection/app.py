import os
import joblib
import streamlit as st
from text_utils import clean_text

MODEL_PATH = os.path.join("models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model artifacts not found. Run `python train_model.py` first to generate `spam_model.pkl` and `tfidf_vectorizer.pkl`."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_spam(message: str, model, vectorizer) -> str:
    processed = clean_text(message)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    return "Spam" if prediction == 1 else "Ham"


def main():
    st.set_page_config(page_title="Spam SMS Detection", page_icon="📩", layout="centered")
    st.title("Spam SMS Detection")
    st.write("Enter an SMS message and the model will classify it as Ham or Spam.")

    sms_input = st.text_area("SMS Message", height=150, placeholder="Type your message here...")

    if st.button("Predict"):
        try:
            model, vectorizer = load_artifacts()
            result = predict_spam(sms_input, model, vectorizer)
            if result == "Spam":
                st.error("🚨 Spam Message")
            else:
                st.success("✅ Ham Message")
            st.write(f"**Prediction:** {result}")
        except Exception as error:
            st.error(f"Error loading model artifacts: {error}")


if __name__ == "__main__":
    main()
