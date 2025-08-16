import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
texts = ["Free money now!!!", "Hello friend, how are you?", "Win a lottery by clicking here", "Meeting at 5pm tomorrow"]
labels = [1, 0, 1, 0]  # 1=spam, 0=ham

# Train
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")
