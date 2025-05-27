import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
@torch.no_grad()
def load_bert_model():
    try:
        model = BertForSequenceClassification.from_pretrained("models/bert-tcfd")
        tokenizer = BertTokenizer.from_pretrained("models/bert-tcfd")
        # model.eval()
        # return tokenizer, model
        with open("models/bert-tcfd/id2label.json") as f:
            id2label = json.load(f)
        return tokenizer, model, id2label
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None, None, None

# Predict using the BERT model
@torch.no_grad()
def predict_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction
