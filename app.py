import streamlit as st
import pickle

st.set_page_config(page_title="📩 Spam Message Detector", layout="centered")

# Load trained model + vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


st.title("📩 Spam Message Detector")
st.write("Enter any message below to check if it is Spam or Ham.")

user_input = st.text_area("Message:", height=100)

if st.button("Detect"):
    if user_input.strip():
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        st.success(f"Prediction: **{prediction}**")
    else:
        st.warning("Please enter a message.")
