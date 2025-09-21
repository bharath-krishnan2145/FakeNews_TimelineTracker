# fake_news_timeline_tracker.py

import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# Download stopwords (only once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Functions
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def compare_texts(text1, text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2])
    similarity_score = util.cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()

# Example Article Versions (You can replace these with real scraped content)
version1 = "The government is launching a healthcare policy to support citizens."
version2 = "The government claims it will launch a so-called healthcare policy, but experts remain doubtful."
version3 = "Officials say a healthcare reform is coming, although no action has been taken yet."

# Preprocess
v1_clean = preprocess_text(version1)
v2_clean = preprocess_text(version2)
v3_clean = preprocess_text(version3)

# Sentiment
s1 = analyze_sentiment(version1)
s2 = analyze_sentiment(version2)
s3 = analyze_sentiment(version3)

# Similarity Scores
sim_1_2 = compare_texts(version1, version2)
sim_2_3 = compare_texts(version2, version3)

print(f"\n🔹 Similarity (Version 1 & 2): {sim_1_2:.2f}")
print(f"🔹 Similarity (Version 2 & 3): {sim_2_3:.2f}")

# Sentiment Scores
print("\n🔹 Sentiment Scores:")
print(f"Version 1: {s1}")
print(f"Version 2: {s2}")
print(f"Version 3: {s3}")

# Timeline Plot
labels = ['Version 1', 'Version 2', 'Version 3']
compound_scores = [s1['compound'], s2['compound'], s3['compound']]

plt.figure(figsize=(8, 5))
plt.plot(labels, compound_scores, marker='o', color='blue', linestyle='-')
plt.title("Sentiment Timeline of News Versions")
plt.xlabel("Article Version")
plt.ylabel("Sentiment (Compound Score)")
plt.grid(True)
plt.ylim(-1, 1)
plt.tight_layout()
plt.show()
