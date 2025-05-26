# models/bert_model.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer once
@torch.no_grad()
def load_bert_model():
    try:
        model = BertForSequenceClassification.from_pretrained("models/bert-tcfd")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None, None

@torch.no_grad()
def predict_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction
