import streamlit as st
from classify import classify_text_classical, classify_text_bert
import pickle
from transformers import BertTokenizerFast, BertForSequenceClassification

st.set_page_config(page_title="TCFD Classifier", layout="wide")

st.title("📄 TCFD Classification")
st.markdown("Upload a text file to classify it into one of the categories.")

# Sidebar
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.radio(
    "What prediction model do you want to use?",
    ("Traditional", "CNN")
)

save_result = st.sidebar.toggle("Save classification result")
filename = st.sidebar.text_input("Enter filename to save as:", value="results.txt")

uploaded_file = st.file_uploader("Upload text file", type=["txt"])

# Load models
# @st.cache_resource
# def load_classical_model():
#     with open("models/vectorizer.pkl", "rb") as f:
#         vectorizer = pickle.load(f)
#     with open("models/logreg_model.pkl", "rb") as f:
#         model = pickle.load(f)
#     return vectorizer, model

# @st.cache_resource
# def load_bert_model():
#     tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
#     model = BertForSequenceClassification.from_pretrained("models/bert")
#     return tokenizer, model

# # Predict
# if uploaded_file is not None:
#     content = uploaded_file.read().decode("utf-8")
#     st.text_area("File Content", content, height=250)

#     if st.button("Classify"):
#         if model_choice == "Traditional":
#             vectorizer, model = load_classical_model()
#             result = classify_text_classical(content, vectorizer, model)
#         else:
#             tokenizer, model = load_bert_model()
#             result = classify_text_bert(content, tokenizer, model)

#         label_names = ["none", "metrics", "strategy", "risk", "governance"]
#         label = label_names[result]
#         st.success(f"Predicted Label: **{label}**")

#         # Save to file
#         if save_result:
#             with open(filename, "w") as f:
#                 f.write(f"Predicted label: {label}\n")
#             st.info(f"Result saved to `{filename}`")