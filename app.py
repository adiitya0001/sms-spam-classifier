import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []  # Initialize y as an empty list
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()  # Clear the list y

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()  # Clear the list y again

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Emial/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. preprocess the input text
    transform_sms = transform_text(input_sms)
    # 2. vectorize the input text
    vector_input = tfidf.transform([transform_sms])
    # 3. predict the class of the input text    
    result = model.predict(vector_input)[0]
    # 4. display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


