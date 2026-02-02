import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def confidence_score(original_text, summary_text):
    vectorizer = CountVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform([original_text, summary_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Convert to percentage
    return round(similarity * 100, 2)


def confidence_label(score):
    if score >= 65:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"
