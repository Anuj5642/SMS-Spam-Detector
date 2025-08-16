import streamlit as st
import pickle

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📧 Spam Classifier")
st.write("Enter a message below and check if it's Spam or Not Spam.")

message = st.text_area("Enter your message:")
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        X = vectorizer.transform([message])
        prediction = model.predict(X)[0]
        if prediction == 1:
            st.error("🚨 This message is Spam!")
        else:
            st.success("✅ This message is Not Spam.")
