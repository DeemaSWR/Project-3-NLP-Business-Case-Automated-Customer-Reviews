import streamlit as st
import pandas as pd
from utils import load_all_models, classify_review, cluster_name, generate_summary

# Set page config (must be FIRST)
st.set_page_config(page_title="Amazon Insights", layout="wide")

# Load all models and data
df, tokenizer_bert, model_bert, model_sbert, tokenizer_bart, model_bart = load_all_models()

# Title and navigation
st.title("🛍️ Amazon Review Analyzer")
page = st.sidebar.radio("Choose a Task", ["Task 1: Classify", "Task 2: Cluster", "Task 3: Summarize"])

# Task 1
if page == "Task 1: Classify":
    st.subheader("🔍 Review Sentiment Classification")
    review = st.text_area("Enter a review to classify")
    if st.button("Classify") and review:
        result = classify_review(review, tokenizer_bert, model_bert)
        st.success(f"Predicted Sentiment: **{result}**")

# Task 2
elif page == "Task 2: Cluster":
    st.subheader("📦 Product Category Clustering")
    product_name = st.selectbox("Select a product name", df["name"].dropna().unique())
    if st.button("Predict Cluster"):
        prediction = cluster_name(product_name, df, model_sbert)
        st.success(f"Meta Category: **{prediction}**")

# Task 3
elif page == "Task 3: Summarize":
    st.subheader("📖 Summarize Reviews by Category")
    
    categories = df["meta_category"].dropna().unique()
    selected_cat = st.selectbox("Choose a product category", categories)

    summary_type = st.radio("What would you like to generate?", [
        "Top 3 products and their differences",
        "Top complaints for top products",
        "Worst product and why it should be avoided"
    ])

    if st.button("Generate"):
        reviews_df = df[df["meta_category"] == selected_cat]

        product_stats = reviews_df.groupby("name")["reviews.rating"].agg(["mean", "count"]).reset_index()
        top_products = product_stats.sort_values("mean", ascending=False).head(3)
        worst_product = product_stats.sort_values("mean").head(1)

        if summary_type == "Top 3 products and their differences":
            st.subheader("✨ Top 3 Products and Their Differences")
            for name in top_products["name"]:
                st.markdown(f"**🛍️ {name}**")
                reviews = reviews_df[reviews_df["name"] == name]["text"].dropna().tolist()[:2]
                st.write(" ".join(reviews) or "No reviews found.")

        elif summary_type == "Top complaints for top products":
            st.subheader("⚠️ Complaints About Top Products")
            for name in top_products["name"]:
                st.markdown(f"**🚫 {name}**")
                neg_reviews = reviews_df[
                    (reviews_df["name"] == name) & (reviews_df["label"] == 0)
                ]["text"].dropna().tolist()[:2]
                st.write(" ".join(neg_reviews) or "No negative reviews found.")

        elif summary_type == "Worst product and why it should be avoided":
            name = worst_product["name"].iloc[0]
            st.subheader(f"❌ Worst Product: {name}")
            reviews = reviews_df[reviews_df["name"] == name]["text"].dropna().tolist()[:3]
            st.write(" ".join(reviews) or "No complaints found.")

        