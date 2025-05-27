import streamlit as st
import pickle
import torch
import os
import pandas as pd

st.set_page_config(page_title="TCFD Classifier", layout="wide")

st.title("TCFD Text Classification")
st.markdown("""
Classify corporate disclosure paragraphs into categories based on the Task Force on Climate-related Financial Disclosures (TCFD) framework.
""")

# Sidebar
st.sidebar.header("Settings")

model_choice = st.sidebar.radio(
    "Select a prediction model:",
    ("Logistic Regression (TF-IDF)", "BERT Transformer"),
    key="model_choice"
)

want_download = st.sidebar.checkbox("Enable result download")
filename = st.sidebar.text_input("Filename (e.g., results.txt or output.csv):", value="results.txt")

# Loaders
@st.cache_resource
def load_logistic_model():
    from models.logistic_model import load_logistic_model as load, predict_logistic, clean_logistic
    model = load()
    return model, predict_logistic, clean_logistic

@st.cache_resource
def load_bert_model():
    from models.bert_model import load_bert_model as load, predict_bert
    tokenizer, model, _ = load()
    return tokenizer, model, predict_bert

model_loaders = {
    "Logistic Regression (TF-IDF)": load_logistic_model,
    "BERT Transformer": load_bert_model
}

selected_model_loader = model_loaders.get(model_choice, lambda: (None, None))

# Label dictionary
label_names = {
    0: "Irrelevant — General or non-climate info",
    1: "Metrics — Emissions metrics and reporting",
    2: "Strategy — Climate strategies and initiatives",
    3: "Risk — Risk assessment and management",
    4: "Governance — Oversight and sustainability leadership"
}

# Tabs
tab1, tab2 = st.tabs(["Classify Single Text", "Classify CSV File"])

# ========== TAB 1: Single Text ==========
with tab1:
    st.subheader("Enter Text Paragraph")
    input_text = st.text_area("Paste or type text here:", height=250)

    if st.button("Classify Text"):
        if not input_text.strip():
            st.warning("Please enter some text first.")
        else:
            prediction = None
            processed_text = None

            if model_choice == "Logistic Regression (TF-IDF)":
                model, predict_logistic, clean_logistic = selected_model_loader()
                if model:
                    processed_text = clean_logistic(input_text)
                    prediction = predict_logistic(processed_text, model)

            elif model_choice == "BERT Transformer":
                tokenizer, model, predict_bert = load_bert_model()
                if tokenizer and model:
                    prediction = predict_bert(input_text, tokenizer, model)

            if prediction is not None:
                st.success(f"**Predicted Category:** {label_names[prediction]}")
                st.info(f"**Processed Text:** {processed_text}")

                if want_download and filename.strip():
                    st.download_button(
                        label="Download Result as Text File",
                        data=f"Predicted Category: {label_names[prediction]}",
                        file_name=filename,
                        mime="text/plain"
                    )
            else:
                st.warning("Classification failed. Model might be unavailable or not implemented yet.")

# ========== TAB 2: CSV File ==========
with tab2:
    st.subheader("Upload CSV File for Batch Classification")
    uploaded_file = st.file_uploader("Upload a CSV file with a column of text to classify", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            text_column = st.selectbox("Select the column containing text data:", df.columns)

            if st.button("Classify CSV"):
                predictions = []

                if model_choice == "Logistic Regression (TF-IDF)":
                    model, predict_logistic, clean_logistic = load_logistic_model()
                    if model:
                        for text in df[text_column]:
                            processed = clean_logistic(str(text))
                            pred = predict_logistic(processed, model)
                            predictions.append(pred)

                elif model_choice == "BERT Transformer":
                    tokenizer, model, predict_bert = load_bert_model()
                    if tokenizer and model:
                        for text in df[text_column]:
                            pred = predict_bert(str(text), tokenizer, model)
                            predictions.append(pred)

                df["Predicted Category ID"] = predictions
                df["Predicted Category"] = df["Predicted Category ID"].map(label_names)

                st.success("Classification completed.")
                st.dataframe(df)

                if want_download and filename.strip():
                    csv_download = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV with Predictions",
                        data=csv_download,
                        file_name=filename,
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")