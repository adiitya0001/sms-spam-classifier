import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- NLTK Data Download ---
# A robust function to ensure necessary NLTK data is available in the cloud environment.
# It checks for the data and downloads it only if it's missing.
def ensure_nltk_data():
    """
    Checks for and downloads 'punkt_tab' and 'stopwords' from NLTK if not present.
    This is the correct way to handle NLTK dependencies on Streamlit Cloud.
    """
    # A list of NLTK resources your app requires.
    required_resources = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords"
    }

    for path, pkg_id in required_resources.items():
        try:
            # Check if the resource is already available
            nltk.data.find(path)
        except LookupError:
            # If not available, download it
            st.info(f"Downloading necessary NLTK data: '{pkg_id}'. Please wait...")
            nltk.download(pkg_id)
            # Rerun the script after download to ensure the app is in a consistent state
            st.rerun()

# Run the NLTK data check at the beginning of the script
ensure_nltk_data()


# --- Caching for Performance ---
# Use st.cache_resource to load heavy objects like models and vectorizers only once.
# This makes your app much faster after the first load.
@st.cache_resource
def load_vectorizer():
    """Loads the TF-IDF vectorizer from a pickle file."""
    with open('vectorizer.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_model():
    """Loads the classification model from a pickle file."""
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)


# --- Text Preprocessing ---
ps = PorterStemmer()

def transform_text(text):
    """
    Performs text preprocessing:
    1. Lowercasing
    2. Tokenization
    3. Filtering for alphanumeric tokens
    4. Removing stopwords and punctuation
    5. Stemming
    Returns a processed string.
    """
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # A single, efficient list comprehension for all processing steps
    stemmed_tokens = [
        ps.stem(word) for word in tokens
        if word.isalnum() and word not in stopwords.words('english')
    ]

    return " ".join(stemmed_tokens)


# --- Main Application ---

# Load the pre-trained components (this will be cached for performance)
tfidf = load_vectorizer()
model = load_model()

# Streamlit UI elements
st.title("Email/SMS Spam Classifier")
st.write("Enter a message below to classify it as Spam or Not Spam.")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict the class
        result = model.predict(vector_input)
        # 4. Display the result
        if result == 1:
            st.header("Result: Spam")
        else:
            st.header("Result: Not Spam")
    else:
        st.warning("Please enter a message to classify.")
