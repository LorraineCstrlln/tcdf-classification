import streamlit as st
from classify import classify_text_classical, classify_text_bert
import pickle
from transformers import BertTokenizerFast, BertForSequenceClassification

st.set_page_config(page_title="TCFD Classifier", layout="wide")

st.title("📄 TCFD Classification")

# Create three tabs
tab1, tab2, tab3 = st.tabs(["Classification", "About", "Team / Poster"])

with tab1:
    st.markdown("Upload a text file to classify it into one of the categories.")

    # Sidebar inside tab? Usually sidebar stays the same, but you can organize settings here if preferred.
    st.sidebar.header("⚙️ Settings")

    model_choice = st.sidebar.radio(
        "What prediction model do you want to use?",
        ("Traditional", "CNN")
    )

    save_result = st.sidebar.toggle("Save classification result")
    filename = st.sidebar.text_input("Enter filename to save as:", value="results.txt")

    uploaded_file = st.file_uploader("Upload text file", type=["txt"])

    @st.cache_resource
    def load_classical_model():
        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("models/logreg_model.pkl", "rb") as f:
            model = pickle.load(f)
        return vectorizer, model

    @st.cache_resource
    def load_bert_model():
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("models/bert")
        return tokenizer, model

    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        st.text_area("File Content", content, height=250)

        if st.button("Classify"):
            if model_choice == "Traditional":
                vectorizer, model = load_classical_model()
                result = classify_text_classical(content, vectorizer, model)
            else:
                tokenizer, model = load_bert_model()
                result = classify_text_bert(content, tokenizer, model)

            label_names = ["none", "metrics", "strategy", "risk", "governance"]
            label = label_names[result]
            st.success(f"Predicted Label: **{label}**")

            # Save to file
            if save_result and filename.strip() != "":
                with open(filename, "w") as f:
                    f.write(f"Predicted label: {label}\n")
                st.info(f"Result saved to `{filename}`")

with tab2:
    st.header("About This Project")
    st.markdown("""
    This project is a Text Classification tool for TCFD (Task Force on Climate-related Financial Disclosures) reports.  
    It classifies uploaded text files into categories such as:
    - None
    - Metrics
    - Strategy
    - Risk
    - Governance

    You can choose between a traditional classical ML model or a CNN-based model using BERT transformers.

    The goal is to support better analysis of climate-related disclosures for investors and stakeholders.
    """)

with tab3:
    st.header("Team & Poster")
    st.markdown("""
    ### Project Team
    - Lorraine Gwen Castrillon
    - Gavril Benedict Coronel
    - Gabriel Inigo De Guzman

    ### Project Poster
    Below is our project poster summarizing the methodology and results.
    """)
    # st.image("images/project_poster.png", caption="TCFD Classifier Project Poster")