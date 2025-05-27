import streamlit as st

st.header("About This Project")

st.markdown("""
This tool is designed to automatically classify corporate disclosure paragraphs based on the **Task Force on Climate-related Financial Disclosures (TCFD)** recommendation categories. It leverages both machine learning and deep learning techniques to support sustainability-related reporting analysis.

---

### What It Does
Given a paragraph of corporate text, the model predicts its appropriate TCFD category.

**Classification Categories:**
- **Irrelevant** – General or unrelated information
- **Metrics** – Emissions metrics and reporting
- **Strategy** – Sustainability strategies and climate initiatives
- **Risk** – Climate risk assessment and management
- **Governance** – Sustainability governance and oversight

---

### How It Works

This tool supports two classification approaches:

#### Logistic Regression (TF-IDF)
- Uses a classical machine learning pipeline.
- Text is converted to lowercase and cleaned by removing punctuation, special characters, and stopwords.
- Applies **TF-IDF Vectorization** to numerically represent input text based on term frequency and inverse document frequency.
- A **Logistic Regression** classifier is trained on these TF-IDF vectors to predict the TCFD category.
- Evaluation involves accuracy measurement and confusion matrix analysis.
- Improvements implemented include:
  - Hyperparameter tuning via **Grid Search** (e.g., `C` regularization parameter)
  - Use of **n-grams** and optimized `min_df`/`max_df` values for better token coverage
  - Balanced class weights to address data imbalance and improve generalization

#### BERT Transformer
- Utilizes the **HuggingFace Transformers** library for fine-tuning the BERT base (uncased) model.
- Necessary libraries: Transformers, Datasets, PyTorch, Scikit-learn, Pandas, NumPy, Matplotlib.
- The text data is loaded from labeled CSV files and tokenized using BERT’s tokenizer.
- Labels are numerically encoded using `LabelEncoder`.
- The tokenized text is wrapped into PyTorch `Dataset` and `DataLoader` objects.
- Training involves multiple epochs, using batched input and gradient descent to fine-tune BERT’s internal layers.
- Evaluation uses:
  - Accuracy score
  - F1-score
  - Full classification report (precision, recall, support)
- Improvements and adjustments:
  - Stratified data split to balance training and validation sets
  - Early stopping and learning rate scheduling to prevent overfitting
  - Batch size and max sequence length tuning for efficiency and performance

---

### Intended Use
This classifier aims to assist researchers, analysts, and institutions in efficiently identifying the focus areas of climate-related disclosures. It is intended for educational and exploratory purposes within the scope of CS 180 coursework at the University of the Philippines - Diliman.

""")