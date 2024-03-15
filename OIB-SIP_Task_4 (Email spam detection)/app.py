import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import joblib

# Function for text processing
def process_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    x = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            x.append(i)
    text = x[:]
    x.clear()
    ps = PorterStemmer()
    for i in text:
        x.append(ps.stem(i))
    return " ".join(x)

# Load the trained model
@st.cache_data
def load_model():
    # Load your trained model here
    model = joblib.load("log_model.pkl")
    return model

# Load TF-IDF Vectorizer
@st.cache_data
def load_vectorizer():
    # Load your TF-IDF Vectorizer here
    vectorizer = joblib.load("vectorizer.pkl")
    return vectorizer

# Main function to run the app
def main():
    st.title("SMS Spam Detection App")

    # Text input for user's SMS
    user_input = st.text_area("Enter your SMS:", "")

    if st.button("Classify"):
        if user_input:
            # Preprocess user input
            processed_input = process_text(user_input)

            # Load TF-IDF Vectorizer
            vectorizer = load_vectorizer()

            # Transform user input
            X = vectorizer.transform([processed_input])

            # Load model
            model = load_model()

            # Make prediction
            prediction = model.predict(X)

            # Display prediction result
            if prediction[0] == 1:
                st.markdown("<div style='background-color:green;padding:10px;border-radius:5px'>Prediction: Spam</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color:green;padding:10px;border-radius:5px'>Prediction: Ham (Non-Spam)</div>", unsafe_allow_html=True)
        else:
            st.write("Please enter an SMS.")

# Run the app
if __name__ == "__main__":
    main()
