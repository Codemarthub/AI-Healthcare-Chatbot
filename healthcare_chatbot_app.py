
# Install required libraries if not already installed
# pip install streamlit scikit-learn pandas nltk deep_translator cryptography

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from deep_translator import GoogleTranslator
from cryptography.fernet import Fernet
import streamlit as st
import nltk
import os

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Load or simulate a dataset
data = pd.DataFrame({
    'Symptom': [
        'fever headache', 'sore throat cough', 'abdominal pain nausea',
        'high fever body ache', 'vomiting diarrhoea', 'loss of appetite fever'
    ],
    'Diagnosis': [
        'Malaria', 'Common Cold', 'Food Poisoning',
        'Typhoid', 'Cholera', 'Typhoid'
    ]
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['Symptom'], data['Diagnosis'], test_size=0.2, random_state=42)

# Vectorize symptom text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(X_train_vec, y_train)

# AES encryption setup
key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_message(message):
    return cipher.encrypt(message.encode())

def is_adversarial(input_text):
    if len(input_text) > 1000 or any(c.isdigit() for c in input_text):
        return True
    suspicious_words = ['hack', 'ddos', 'crash']
    return any(word in input_text.lower() for word in suspicious_words)

def translate_input(user_input, src_lang='auto', target_lang='en'):
    try:
        return GoogleTranslator(source=src_lang, target=target_lang).translate(user_input)
    except:
        return user_input

# Streamlit chatbot UI
def run_chatbot():
    st.set_page_config(page_title="AVECARE AI Health Chatbot", page_icon="🩺", layout="centered")

    # Display logo image — make sure 'avecare_logo.png' is in your repo folder
    st.image("avecare_logo.png", width=180)

    # Title and description
    st.markdown("<h1 style='text-align: center; color: #01579B;'>AVECARE AI Healthcare Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Get AI-powered preliminary health advice based on your symptoms.</p>", unsafe_allow_html=True)
    st.write("---")

    # Consent agreement
    consent = st.checkbox("I agree and understand this AI chatbot is for informational advice only and does not replace a medical doctor.")
    if not consent:
        st.warning("Please check the consent box to proceed.")
        st.stop()

    st.markdown("#### Describe your symptoms (separate them with commas):")
    user_input = st.text_area("", placeholder="Example: fever, headache, cough")

    col1, col2 = st.columns([2, 1])
    with col2:
        if st.button("Get Diagnosis"):

            if user_input.strip() == "":
                st.error("Please describe your symptoms.")
            elif is_adversarial(user_input):
                st.error("⚠️ Invalid or suspicious input detected.")
            else:
                translated_input = translate_input(user_input)
                input_vec = vectorizer.transform([translated_input])
                prediction = model.predict(input_vec)

                st.success(f"💡 AI Diagnosis Suggestion: **{prediction[0]}**")
                st.info("This is a preliminary AI-generated suggestion based on recognized symptom patterns. Always consult a certified doctor for clinical confirmation.")

                log_message = f"Symptoms: {translated_input}, Diagnosis: {prediction[0]}"
                encrypted_log = encrypt_message(log_message)
                with open("chatbot_logs.txt", "ab") as log_file:
                    log_file.write(encrypted_log + b"\n")

                st.success("📝 Your interaction has been securely logged (AES-256 encrypted).")

    st.write("---")
    st.markdown("<small style='color: grey;'>Developed by Martins T. Okwuosah, Ave Maria University, Nigeria.</small>", unsafe_allow_html=True)

if __name__ == '__main__':
    run_chatbot()
