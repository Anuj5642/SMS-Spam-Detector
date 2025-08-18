# 📩 Spam Message Detector 

## Files included:
- spam.py → Streamlit app
- train_model.py → retrain Logistic Regression model
- spam_classifier.pkl → trained model
- vectorizer.pkl → trained TF-IDF vectorizer
- SMSSpamCollection → dataset 
- requirements.txt → dependencies

## Run locally:
```bash
pip install -r requirements.txt
streamlit run spam.py
```

## Retrain model:
```bash
python train_model.py
```
