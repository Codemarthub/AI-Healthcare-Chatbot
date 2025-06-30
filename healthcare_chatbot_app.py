
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

# Streamlit Chatbot UI
def run_chatbot():
    st.set_page_config(page_title="AI Healthcare Chatbot", page_icon="🩺")
    st.title("AI-Powered Healthcare Chatbot 🩺")
    st.write("Preliminary health advice based on your symptoms. For educational use only.")

    consent = st.checkbox("I understand this AI does not replace a doctor.")
    if not consent:
        st.warning("Please provide consent to continue.")
        st.stop()

    user_input = st.text_area("Describe your symptoms (separate with commas):")

    if st.button("Get Diagnosis"):
        if user_input.strip() == "":
            st.error("Please enter your symptoms.")
        elif is_adversarial(user_input):
            st.error("⚠️ Invalid or unsafe input detected.")
        else:
            translated_input = translate_input(user_input)
            input_vec = vectorizer.transform([translated_input])
            prediction = model.predict(input_vec)

            st.success(f"🩺 AI Diagnosis Suggestion: {prediction[0]}")
            st.info("Diagnosis based on AI recognition of symptom patterns.")

            log_message = f"Symptoms: {translated_input}, Diagnosis: {prediction[0]}"
            encrypted_log = encrypt_message(log_message)
            with open("chatbot_logs.txt", "ab") as log_file:
                log_file.write(encrypted_log + b"\n")

            st.success("Your interaction has been securely logged.")

if __name__ == '__main__':
    run_chatbot()
