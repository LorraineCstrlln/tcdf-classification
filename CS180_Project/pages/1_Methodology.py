import streamlit as st

st.header("About This Project")

st.markdown("""
This tool is designed to automatically classify corporate disclosure paragraphs based on the **Task Force on Climate-related Financial Disclosures (TCFD)** recommendation categories. It leverages machine learning and deep learning techniques to support sustainability-related reporting analysis.

### What It Does
Given a paragraph of corporate text, the model predicts its appropriate TCFD category.

**Classification Categories:**
- **Irrelevant** – General or unrelated information
- **Metrics** – Emissions metrics and reporting
- **Strategy** – Sustainability strategies and climate initiatives
- **Risk** – Climate risk assessment and management
- **Governance** – Sustainability governance and oversight

### How It Works
- The system uses both classical machine learning (TF-IDF + Logistic Regression) and deep learning (BERT Transformer) models.
- The models are trained on a curated dataset of corporate disclosures labeled according to the TCFD framework.
- The tool processes text inputs using standardized pipelines, ensuring accurate and fair classification.

### Intended Use
This classifier aims to assist researchers, analysts, and institutions in efficiently identifying the focus areas of climate-related disclosures. It is intended for educational and exploratory purposes within the scope of CS 180 coursework at the University of the Philippines - Diliman.

---

Project developed by:  
- Lorraine Gwen M. Castrillon  
- Gavril Benedict L. Coronel  
- Gabriel Inigo De Guzman

(Department of Computer Science, College of Engineering, UP Diliman)
""")
