"""Streamlit interface for the trained SMS spam classifier."""
from pathlib import Path

import joblib
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = Path("models/sms_lstm.keras")
TOKENIZER_PATH = Path("models/tokenizer.joblib")
MAX_LEN = 100

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("SMS Spam Detection")
st.caption("Long Short-Term Memory (LSTM) text classifier")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        return None, None
    return (
        tf.keras.models.load_model(MODEL_PATH),
        joblib.load(TOKENIZER_PATH),
    )


model, tokenizer = load_artifacts()

if model is None:
    st.warning("Model files are missing. Download the dataset and run python train.py first.")
else:
    message = st.text_area(
        "Enter an SMS message",
        height=140,
        placeholder="Example: Congratulations! You have won a prize...",
    )
    if st.button("Classify message", type="primary", use_container_width=True):
        if not message.strip():
            st.error("Please enter a message.")
        else:
            sequence = tokenizer.texts_to_sequences([message])
            padded = pad_sequences(
                sequence, maxlen=MAX_LEN, padding="post", truncating="post"
            )
            spam_probability = float(model.predict(padded, verbose=0)[0][0])
            if spam_probability >= 0.5:
                st.error("Prediction: SPAM")
            else:
                st.success("Prediction: HAM (legitimate)")
            st.metric("Estimated spam probability", f"{spam_probability:.1%}")
            st.caption("This is a model prediction, not a guarantee that a message is safe.")
