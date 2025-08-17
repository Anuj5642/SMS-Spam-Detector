import streamlit as st
from transformers import pipeline

# Load BERT Spam Classifier (pretrained on SMS Spam dataset)
@st.cache_resource
def load_model():
    classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
    return classifier

st.title("📩 Spam Message Detector")

classifier = load_model()

user_input = st.text_area("Enter a message to classify:")

if st.button("Check Spam"):
    if user_input.strip():
        result = classifier(user_input)[0]
        label = result['label']
        score = round(result['score'], 3)

        if label.lower() == "spam":
            st.error(f"🚨 Spam Detected! (Confidence: {score})")
        else:
            st.success(f"✅ Not Spam (Confidence: {score})")
    else:
        st.warning("Please enter a message.")
