import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📩 Spam Message Detector (Logistic Regression)")

message = st.text_area("Enter a message to classify")

if st.button("Predict"):
    if message.strip() != "":
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Ham (Not Spam)")
    else:
        st.warning("Please enter a message.")
