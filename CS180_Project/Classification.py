import streamlit as st
import pickle
import os

# Page configuration
st.set_page_config(page_title="TCFD Classifier", layout="wide")

# App Header
st.title("TCFD Text Classification")
st.markdown("""
Classify corporate disclosure paragraphs into categories based on the Task Force on Climate-related Financial Disclosures (TCFD) framework.
""")

# Sidebar for model and save settings
st.sidebar.header("Settings")

model_choice = st.sidebar.radio(
    "Select a prediction model:",
    ("Logistic Regression (TF-IDF)", "BERT Transformer"),
    key="model_choice"
)

save_result = st.sidebar.checkbox("Save result to file")
filename = st.sidebar.text_input("Filename (e.g., results.txt):", value="results.txt")

# Input and classification section
st.subheader("Enter Text Paragraph")

input_text = st.text_area("Paste or type text here:", height=250)

@st.cache_resource
def load_logistic_model():
    try:
        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("models/logreg_model.pkl", "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        st.error("TF-IDF model files not found.")
        return None, None

@st.cache_resource
def load_bert_model():
    from models.bert_model import load_bert_model as load, predict_bert
    tokenizer, model = load()
    return tokenizer, model, predict_bert

# Use variable to map to model function
model_loaders = {
    "Logistic Regression (TF-IDF)": load_logistic_model,
    "BERT Transformer": load_bert_model
}

selected_model_loader = model_loaders.get(model_choice, lambda: (None, None))

if st.button("Classify Text"):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        prediction = None

        if model_choice == "Logistic Regression (TF-IDF)":
            vectorizer, model = selected_model_loader()
            if vectorizer is not None and model is not None:
                prediction = model.predict(vectorizer.transform([input_text]))[0]

        elif model_choice == "BERT Transformer":
            tokenizer, model, predict_bert = selected_model_loader()
            if tokenizer is not None and model is not None:
                prediction = predict_bert(input_text, tokenizer, model)

        # Category labels
        label_names = {
            0: "Irrelevant — General or non-climate info",
            1: "Metrics — Emissions metrics and reporting",
            2: "Strategy — Climate strategies and initiatives",
            3: "Risk — Risk assessment and management",
            4: "Governance — Oversight and sustainability leadership"
        }

        if prediction is not None:
            st.success(f"**Predicted Category:** {label_names[prediction]}")
            if save_result and filename.strip():
                try:
                    with open(filename, "w") as f:
                        f.write(f"Predicted Category: {label_names[prediction]}\n")
                    st.info(f"Result saved to `{filename}`.")
                except Exception as e:
                    st.error(f"Could not save file: {e}")
        else:
            st.warning("Classification failed. Model might be unavailable or not implemented yet.")