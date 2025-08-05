# Email/SMS Spam Classifier (with Streamlit)

This project classifies messages as Spam or Not Spam using a trained Multinomial Naive Bayes model.
The app uses a Streamlit frontend.

## Files
- `app.py` : Streamlit app
- `model.pkl` : trained model
- `vectorizer.pkl` : TF-IDF vectorizer
- `requirements.txt` : required libraries

## To Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
