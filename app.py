import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# --- Configuration ---
MODEL_PATH = 'News_sentment_analyser_GRU.h5'
TOKENIZER_PATH = 'tokenizer.pickle'
MAX_SEQUENCE_LENGTH = 217

# --- Load Model and Tokenizer ---
# Use st.cache_resource to load the model and tokenizer only once
@st.cache_resource
def load_model_and_tokenizer():
    """Loads the pre-trained Keras model and tokenizer."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as file:
            tokenizer = pickle.load(file)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

# --- Prediction Function ---
def predict_sentiment(model, tokenizer, text, max_length):
    """
    Predicts the sentiment of a given text string.
    """
    if model is None or tokenizer is None:
        return "Model not loaded", 0.0

    # Preprocess the text
    token_list = tokenizer.texts_to_sequences([text])
    padded_token_list = pad_sequences(token_list, maxlen=max_length, padding='pre')

    # Make the prediction
    prediction_score = model.predict(padded_token_list, verbose=0)[0][0]

    # Interpret the prediction
    if prediction_score > 0.5:
        return "POSITIVE", prediction_score
    else:
        return "NEGATIVE", prediction_score

# --- Streamlit Web App Interface ---
st.set_page_config(page_title="News Sentiment Analyzer", layout="centered")

st.title("📰 News Sentiment Analyzer")
st.markdown("Enter a news headline or a short text below to determine if its sentiment is positive or negative.")

# --- User Input ---
user_input = st.text_area("Enter News Text:", height=150, placeholder="e.g., The company's stock plummeted after failing to meet expectations.")

# --- Analyze Button ---
if st.button("Analyze Sentiment"):
    if user_input and model and tokenizer:
        with st.spinner('Analyzing...'):
            sentiment, score = predict_sentiment(model, tokenizer, user_input, MAX_SEQUENCE_LENGTH)

            if sentiment == "POSITIVE":
                st.success(f"**Sentiment: {sentiment}**")
                st.write(f"Confidence Score: {score:.4f}")
            else:
                st.error(f"**Sentiment: {sentiment}**")
                st.write(f"Confidence Score: {1-score:.4f} (Strength of negative sentiment)")

            # Progress bar for visual effect
            st.progress(1.0)
    elif not user_input:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
