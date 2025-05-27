# import joblib
# import json

# def load_logistic_model():
#     try:
#         model = joblib.load("models/logistic-tcdf/best_random_pipeline_model.joblib")
#         with open("models/logistic-tcdf/id2label.json") as f:
#             id2label = json.load(f)
#         return model, id2label
#     except Exception as e:
#         print(f"Error loading joblib model: {e}")
#         return None, None

# def predict_logistic(text, model):
#     try:
#         return model.predict([text])[0]
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return None

import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

def load_logistic_model():
    try:
        model = joblib.load("models/logistic-tcfd/best_random_pipeline_model.joblib")
        return model
    except Exception as e:
        print(f"Error loading Logistic model: {e}")
        return None

def predict_logistic(text, model):
    return model.predict([text])[0]

tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

def clean_logistic(text):
    text_clean = " ".join(text.lower().strip().split())
    text_clean = re.sub(r'[^\w\s%\:\-\.]', '', text_clean)
    tokens = tokenizer.tokenize(text_clean) 
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmas)
