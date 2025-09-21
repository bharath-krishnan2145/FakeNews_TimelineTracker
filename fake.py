# fake_news_timeline_tracker.py

import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Sentiment Analysis
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

# Semantic Similarity
def compare_texts(text1, text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2])
    similarity_score = util.cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()

# Sample Article Versions (you can replace these with real data)
version1 = "The government is launching a healthcare policy to support citizens."
version2 = "The government claims it will launch a so-called healthcare policy, but experts remain doubtful."
version3 = "Officials say a healthcare reform is coming, although no action has been taken yet."

versions = [version1, version2, version3]
labels = ['Version 1', 'Version 2', 'Version 3']

# Analyze and Compare
preprocessed_versions = [preprocess_text(v) for v in versions]
sentiments = [analyze_sentiment(v)['compound'] for v in versions]

similarity_1_2 = compare_texts(versions[0], versions[1])
similarity_2_3 = compare_texts(versions[1], versions[2])

# Output results
print("🔍 Sentiment Scores:")
for i, score in enumerate(sentiments):
    print(f"{labels[i]}: {score:.2f}")

print("\n🔗 Similarity Scores:")
print(f"Between Version 1 & 2: {similarity_1_2:.2f}")
print(f"Between Version 2 & 3: {similarity_2_3:.2f}")

# Plot Sentiment Timeline
plt.figure(figsize=(8, 5))
plt.plot(labels, sentiments, marker='o', color='blue', linestyle='-')
plt.title("📰 Sentiment Timeline of News Versions")
plt.xlabel("Article Version")
plt.ylabel("Sentiment Score (Compound)")
plt.ylim(-1, 1)
plt.grid(True)
plt.tight_layout()
plt.show()
