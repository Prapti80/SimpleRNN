# =========================
# Step 1: Imports
# =========================
import re
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# =========================
# Step 2: Load IMDB Word Index
# =========================
word_index = imdb.get_word_index()

PAD = 0
START = 1
UNK = 2
UNUSED = 3


# =========================
# Step 3: Load Model
# =========================
model = load_model("simple_rnn_imdb.h5")


# =========================
# Step 4: Clean Text
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# =========================
# Step 5: Preprocess Input
# =========================
def preprocess_text(text):
    text = clean_text(text)
    words = text.split()

    encoded_review = []
    for word in words:
        if word in word_index and word_index[word] < 10000:
            encoded_review.append(word_index[word])
        else:
            encoded_review.append(UNK)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500,          # ✅ MUST be 500
        padding="pre",
        truncating="pre"
    )

    return padded_review


# =========================
# Step 6: Streamlit UI
# =========================
st.set_page_config(page_title="IMDB Sentiment Analysis")

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as Positive or Negative.")

user_input = st.text_area(
    "Movie Review",
    value="This movie was fantastic! The acting was great and the plot was thrilling."
)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        score = prediction[0][0]

        sentiment = "Positive 😀" if score >= 0.5 else "Negative 😞"

        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: **{score:.4f}**")
        st.progress(float(score))
