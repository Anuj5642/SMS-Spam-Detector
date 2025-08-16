# 📩 SMS Spam Detector

A simple Streamlit app that detects whether an SMS message is **spam** or **ham** using machine learning.

## Files
- `spam.py` : Main Streamlit app
- `requirements.txt` : Dependencies
- `Procfile` : For deployment on Render
- `SMSSpamCollection` : Dataset

## Run Locally
```bash
pip install -r requirements.txt
streamlit run spam.py
```

## Deploy on Render
1. Push repo to GitHub
2. Connect GitHub to Render
3. Deploy with `Procfile`
