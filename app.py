# app.py

import streamlit as st
import joblib
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the model and vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

st.title("📰 Fake / Real News Classifier")

user_input = st.text_area("Enter News Article Text")

if st.button("Classify"):
    if len(user_input.strip().split()) < 10:
        st.warning("⚠️ Please enter a longer news article. Short sentences may give unreliable results.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        result = "REAL" if prediction[0] == 1 else "FAKE"
        st.subheader(f"Prediction: {result}")

