import streamlit as st
import nltk

st.title("NLTK Download Test")

try:
    st.write("Attempting to download 'punkt'...")
    nltk.download('punkt')
    st.success("✅ 'punkt' downloaded successfully!")

    st.write("Attempting to download 'stopwords'...")
    nltk.download('stopwords')
    st.success("✅ 'stopwords' downloaded successfully!")

    st.write("---")
    st.header("Testing Tokenizer")
    test_sentence = "This is a test sentence."
    tokens = nltk.word_tokenize(test_sentence)
    st.write(f"Tokenizing: '{test_sentence}'")
    st.write("Result:", tokens)
    st.success("✅ Tokenizer test passed!")

except Exception as e:
    st.error("A critical error occurred during the test:")
    st.exception(e)
