import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from transformers import BartTokenizer, BartForConditionalGeneration
import streamlit as st

@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\deept\\SDA Bootcamp\\Week 6\\Project 3\\app\\data\\df_balanced.csv")

@st.cache_resource
def load_all_models():
    df = load_data()
    tokenizer_bert = AutoTokenizer.from_pretrained("C:\\Users\\deept\\SDA Bootcamp\\Week 6\\Project 3\\app\\models\\task1_bert")
    model_bert = AutoModelForSequenceClassification.from_pretrained("C:\\Users\\deept\\SDA Bootcamp\\Week 6\\Project 3\\app\\models\\task1_bert")

    model_sbert = SentenceTransformer("C:\\Users\\deept\\SDA Bootcamp\\Week 6\\Project 3\\app\\models\\task2_sentencebert")

    tokenizer_bart = BartTokenizer.from_pretrained("C:\\Users\\deept\\SDA Bootcamp\\Week 6\\Project 3\\app\\models\\task3_bart")
    model_bart = BartForConditionalGeneration.from_pretrained("C:\\Users\\deept\\SDA Bootcamp\\Week 6\\Project 3\\app\\models\\task3_bart")
    return df, tokenizer_bert, model_bert, model_sbert, tokenizer_bart, model_bart

# Task 1: Review Classification
def classify_review(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[prediction]

# Task 2: Product Name Clustering
def cluster_name(name, df, model_sbert):
    name_embedding = model_sbert.encode(name, convert_to_tensor=True)
    meta_categories = df["meta_category"].dropna().unique()
    meta_embeddings = model_sbert.encode(meta_categories, convert_to_tensor=True)
    scores = util.cos_sim(name_embedding, meta_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    return meta_categories[best_idx]

# Task 3: Summarization
def generate_summary(df, category, tokenizer, model):
    reviews_df = df[df["meta_category"] == category]

    # Compute product stats
    product_stats = reviews_df.groupby("name")["reviews.rating"].agg(["mean", "count"]).reset_index()
    top_products = product_stats.sort_values("mean", ascending=False)
    worst_product = product_stats.sort_values("mean").drop_duplicates(subset=["name"])
    worst_name = worst_product["name"].iloc[0]

    # Remove the worst product from top list
    top_products = top_products[top_products["name"] != worst_name]

    # Filter top products to ensure they're not just color/variant duplicates
    def simplify_name(name):
        return " ".join(name.lower().split()[:6])  # crude heuristic for grouping similar products

    seen = set()
    unique_top_names = []
    for name in top_products["name"]:
        root = simplify_name(name)
        if root not in seen:
            seen.add(root)
            unique_top_names.append(name)
        if len(unique_top_names) == 3:
            break

    # Prepare review blocks for top products
    top_reviews_blocks = []
    for name in unique_top_names:
        sample = reviews_df[reviews_df["name"] == name]["text"].dropna().tolist()[:3]
        review_text = "\n".join(sample)
        top_reviews_blocks.append(f"{name}:\n{review_text}")

    # Prepare worst product review block
    worst_reviews = reviews_df[reviews_df["name"] == worst_name]["text"].dropna().tolist()[:3]
    worst_block = f"{worst_name}:\n" + "\n".join(worst_reviews)

    # Combine review texts
    top_reviews_text = "\n\n".join(top_reviews_blocks)

    
