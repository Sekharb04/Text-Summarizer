import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Extractive Model


def textrank_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    # TF-IDF sentence vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Score sentences
    scores = similarity_matrix.sum(axis=1)

    # Rank sentences
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    # Pick top N sentences
    selected_sentences = [s for _, s in ranked_sentences[:num_sentences]]

    return " ".join(selected_sentences)
