# -*- coding: utf-8 -*-
"""
Improved Movie Recommendation System
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data_url = 'https://raw.githubusercontent.com/Enqey/mo.ai/main/IMDB-Movie-Data.csv'

st.title("🎥 Movie Recommendation System")
st.write("""
    **Find movies similar to your favorites!**  
    Select a movie, and we'll recommend others you'll love.
""")

try:
    df = pd.read_csv(data_url)
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Check for missing values
required_columns = ['Actors', 'Director', 'Genre', 'Title']
missing_data = df[required_columns].isnull().any().any()
if missing_data:
    st.warning("Missing values detected. Filling with placeholder values...")
    df.fillna('Unknown', inplace=True)

# Feature engineering
df['combined_features'] = df['Actors'] + ' ' + df['Director'] + ' ' + df['Genre']

# Vectorize features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Sidebar movie selection
movie_titles = df['Title'].tolist()
selected_movie = st.sidebar.selectbox("Select a movie:", movie_titles)

if selected_movie:
    st.subheader(f"Movies similar to: {selected_movie}")
    
    # Get index of the selected movie
    movie_idx = df[df['Title'] == selected_movie].index[0]
    
    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:8]  # Top 7
    
    # Display recommendations
    for idx, (movie_index, score) in enumerate(sim_scores):
        st.write(f"{idx + 1}. {df.iloc[movie_index]['Title']}")

st.write("---")
st.write("**Developed by Nana Ekow Okusu**")
