# Project-3-NLP-Business-Case-Automated-Customer-Reviews

# Amazon Product Review Analyzer - README

## 📌 Project Overview
This project applies Natural Language Processing (NLP) techniques to analyze Amazon product reviews. It includes three main tasks:
1. **Sentiment Classification** using BERT  
2. **Product Clustering** using Sentence-BERT + KMeans  
3. **Review Summarization** using BART  

Each task contributes to understanding customer feedback and extracting useful insights for decision-making.

---

## 🗂️ Directory Structure
```
Project_Root/
├── app/
│   ├── app.py                    # Streamlit app
│   ├── utils.py                  # Utility functions (classification, clustering, summarization)
│   ├── models/                   # Directory containing fine-tuned BERT, SBERT, and BART models
│   ├── data/
│   │   └── df_balanced.csv       # Cleaned and preprocessed review dataset
│   └── requirements.txt          # Dependencies for the app
├── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/amazon-review-analyzer.git
cd amazon-review-analyzer/app
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 📦 Models Used
All models are loaded locally from the `models/` folder. Make sure these directories exist:
- `task1_bert/` – Fine-tuned BERT model for sentiment classification  
- `task2_sentencebert/` – Pretrained `all-MiniLM-L6-v2` Sentence-BERT for clustering  
- `task3_bart/` – Fine-tuned BART model for summarization  

If missing, you can re-download from HuggingFace or fine-tune using provided scripts.

---

## 📊 Reproducing Results
To reproduce the results:
1. Ensure the dataset `df_balanced.csv` is available under `app/data/`  
2. Launch the app or call `utils.py` functions in Jupyter to run individual tasks

---

## 🔗 Google Drive (Models & Output Files)
Add your Google Drive link here:  
`[[Insert Google Drive Link]](https://drive.google.com/drive/folders/1azftjCEttxldx-SDHQqfhvzsTbTz7njk?usp=sharing)
`
.
