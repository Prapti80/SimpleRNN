import re
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

VOCAB_SIZE = 10000
MAXLEN = 500
UNK = 2


word_index = imdb.get_word_index()

model = Sequential([
    Embedding(VOCAB_SIZE, 32, input_length=MAXLEN),
    SimpleRNN(32),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.load_weights("simple_rnn_imdb.h5")


def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z\s]", "", text)

def preprocess_text(text):
    words = clean_text(text).split()
    encoded = [
        word_index[word] if word in word_index and word_index[word] < VOCAB_SIZE else UNK
        for word in words
    ]
    return sequence.pad_sequences([encoded], maxlen=MAXLEN)

st.set_page_config(page_title="IMDB Sentiment Analysis")
st.title("IMDB Movie Review Sentiment Analysis")

review = st.text_area("Enter a movie review:")

if st.button("Classify"):
    if review.strip():
        x = preprocess_text(review)
        score = model.predict(x)[0][0]
        st.subheader("Positive 😀" if score >= 0.5 else "Negative 😞")
        st.write(f"Score: {score:.4f}")
    else:
        st.warning("Please enter a review.")
