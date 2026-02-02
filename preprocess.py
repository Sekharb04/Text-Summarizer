import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required resources exist
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stopwords.words('english')
        and word not in string.punctuation
    ]

    return " ".join(tokens)
