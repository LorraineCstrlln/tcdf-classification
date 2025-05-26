# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import torch
# from transformers import BertTokenizerFast, BertForSequenceClassification

# # Classical ML
# def classify_text_classical(text, vectorizer, model):
#     cleaned = " ".join(text.lower().strip().split())
#     X = vectorizer.transform([cleaned])
#     pred = model.predict(X)[0]
#     return pred

# def classify_text_bert(text, tokenizer, model, device='cpu'):
#     model.eval()
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#     predicted_class = torch.argmax(logits, dim=1).item()
#     return predicted_class