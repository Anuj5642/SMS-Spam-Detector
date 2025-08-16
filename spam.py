import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
    return df

df = load_data()

# Preprocess
X = df["message"]
y = df["label"].map({"ham": 0, "spam": 1})

vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# App UI
st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message to check if it is spam or not.")

user_input = st.text_area("Message", "")

if st.button("Predict"):
    if user_input.strip() != "":
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        st.success("✅ Ham (Not Spam)" if prediction == 0 else "🚨 Spam")
    else:
        st.warning("Please enter a message.")

# Show accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.sidebar.write(f"Model Accuracy: {acc:.2f}")
