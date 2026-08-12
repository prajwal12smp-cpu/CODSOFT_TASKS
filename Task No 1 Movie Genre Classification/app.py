from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.predict import load_model_artifacts, predict_genre


PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="🎬 Movie Genre Classifier", page_icon="🎬", layout="centered")

st.title("🎬 Movie Genre Classifier")
st.markdown(
    "A lightweight movie genre classifier built with TF-IDF and classic NLP models. "
    "Paste a movie plot or summary and get an instant genre prediction."
)

model_loaded = False
with st.spinner("Loading trained model..."):
    try:
        load_model_artifacts()
        model_loaded = True
    except FileNotFoundError:
        st.warning("The trained model could not be found. Run python src/train.py before using the app.")

plot_text = st.text_area(
    "Enter the movie plot/description...",
    height=220,
    placeholder="A young detective investigates..."
)

predict_button = st.button("Predict Genre", use_container_width=True)

if predict_button:
    if not plot_text.strip():
        st.error("Please enter a movie plot before making a prediction.")
    elif not model_loaded:
        st.error("Model files are missing. Run python src/train.py and refresh the app.")
    else:
        try:
            prediction = predict_genre(plot_text)
            st.success(f"Predicted Genre: {prediction['predicted_genre']}")

            if prediction["confidence"] is not None:
                st.write("#### Confidence")
                st.write(f"**{prediction['confidence']:.2f}%**")

                st.write("#### Top 3 predicted genres")
                for item in prediction["top_predictions"]:
                    st.write(f"- **{item['genre']}**: {item['confidence']:.2f}%")
        except FileNotFoundError:
            st.error("Model files have not been created yet. Please run the training script first: python src/train.py")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - UI guard
            st.error(f"Prediction failed: {exc}")
