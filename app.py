# app.py

import streamlit as st
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt

# Setup
nltk.download('punkt')
analyzer = SentimentIntensityAnalyzer()
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")


# Functions
def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score['compound'], score

def simulate_evolution(text):
    prompt = f"Rewrite the following news article in a more sensational, exaggerated style:\n{text}\nRewritten:"
    result = generator(prompt, max_length=300, do_sample=True, temperature=0.9)
    evolved = result[0]['generated_text'].split("Rewritten:")[-1].strip()
    return evolved

# Streamlit UI
st.set_page_config(page_title="Fake News Timeline Tracker", layout="centered")
st.title("📰 Fake News Evolution Predictor")
st.write("Paste a news article below. The AI will simulate how the news might evolve and analyze sentiment change.")

# User Input
user_input = st.text_area("Paste News Article Here", height=200)

if st.button("Track Evolution"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Analyzing and generating..."):
            original_score, original_breakdown = get_sentiment(user_input)
            evolved_text = simulate_evolution(user_input)
            evolved_score, evolved_breakdown = get_sentiment(evolved_text)

        st.subheader("🔹 Original News Sentiment:")
        st.write(f"Sentiment Score: {original_score:.2f}")
        st.json(original_breakdown)

        st.subheader("🔸 Evolved News Prediction:")
        st.write(evolved_text)

        st.subheader("🔹 Evolved News Sentiment:")
        st.write(f"Sentiment Score: {evolved_score:.2f}")
        st.json(evolved_breakdown)

        # Chart
        st.subheader("📊 Sentiment Comparison")
        labels = ['Original', 'Evolved']
        scores = [original_score, evolved_score]
        fig, ax = plt.subplots()
        ax.bar(labels, scores, color=['blue', 'red'])
        ax.set_ylabel("Sentiment Score")
        ax.set_ylim(-1, 1)
        st.pyplot(fig)

        st.success("Done! Scroll up to see results.")

