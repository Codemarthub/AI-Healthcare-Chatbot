
# Install dependencies if not already installed
# pip install streamlit scikit-learn pandas nltk deep_translator cryptography

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from deep_translator import GoogleTranslator
from cryptography.fernet import Fernet
import streamlit as st
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Expanded dataset
data = pd.DataFrame({
    'Symptom': [
        'fever headache', 'sore throat cough', 'abdominal pain nausea',
        'high fever body ache', 'vomiting diarrhoea', 'loss of appetite fever',
        'cough sneezing', 'blurred vision headache', 'skin rash fever', 'severe headache vomiting'
    ],
    'Diagnosis': [
        'Malaria', 'Common Cold', 'Food Poisoning',
        'Typhoid', 'Cholera', 'Typhoid',
        'Common Cold', 'Migraine', 'Measles', 'Meningitis'
    ]
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['Symptom'], data['Diagnosis'], test_size=0.3, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=0, max_depth=4)
model.fit(X_train_vec, y_train)

# Evaluate accuracy
accuracy = accuracy_score(y_test, model.predict(X_test_vec))

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
    st.image("avecare_logo.png", width=180)
    st.markdown("<h1 style='text-align: center; color: #01579B;'>AVECARE AI Healthcare Chatbot</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Accuracy: {round(accuracy * 100, 2)}%</p>", unsafe_allow_html=True)
    st.write("---")

    consent = st.checkbox("I agree and understand this AI chatbot is for informational advice only and does not replace a medical doctor.")
    if not consent:
        st.warning("Please check the consent box to proceed.")
        st.stop()

    lang_choice = st.selectbox("Choose your input language:", ["English", "French", "Spanish", "Hausa", "Igbo", "Yoruba"])
    lang_map = {'English':'en','French':'fr','Spanish':'es','Hausa':'ha','Igbo':'ig','Yoruba':'yo'}

    st.markdown("#### Describe your symptoms (separate them with commas):")
    user_input = st.text_area("", placeholder="Example: fever, headache, vomiting")

    recommendations = {
        'Malaria': "Drink fluids, avoid self-medication. See a doctor for antimalarial drugs.",
        'Common Cold': "Rest, drink warm fluids, and use mild cold medication.",
        'Food Poisoning': "Stay hydrated, use oral rehydration salts. Visit clinic if vomiting persists.",
        'Typhoid': "See a doctor for antibiotics and stay hydrated.",
        'Cholera': "Immediate oral rehydration and hospital visit advised.",
        'Migraine': "Rest in a dark room. Use prescribed pain relief and stay hydrated.",
        'Measles': "Hydrate, use fever reducers, and consult a healthcare professional.",
        'Meningitis': "Seek urgent hospital care. Do not self-medicate."
    }

    col1, col2 = st.columns([2, 1])
    with col2:
        if st.button("Get Diagnosis"):
            if user_input.strip() == "":
                st.error("Please describe your symptoms.")
            elif is_adversarial(user_input):
                st.error("⚠️ Invalid or suspicious input detected.")
            else:
                translated_input = translate_input(user_input, src_lang=lang_map[lang_choice])
                input_vec = vectorizer.transform([translated_input])
                prediction = model.predict(input_vec)[0]

                st.success(f"💡 AI Diagnosis Suggestion: **{prediction}**")
                advice = recommendations.get(prediction, "Please consult a healthcare professional.")
                st.warning(f"🩺 Recommended Advice: {advice}")

                log_message = f"Symptoms: {translated_input}, Diagnosis: {prediction}"
                encrypted_log = encrypt_message(log_message)
                with open("chatbot_logs.txt", "ab") as log_file:
                    log_file.write(encrypted_log + b"\n")

                st.success("📝 Interaction securely logged (AES-256 encrypted).")

    st.write("---")
    st.markdown("<small style='color: grey;'>Developed by Martins T. Okwuosah, Ave Maria University, Nigeria.</small>", unsafe_allow_html=True)

if __name__ == '__main__':
    run_chatbot()
