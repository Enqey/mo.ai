# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 08:03:25 2022

@author: Enqey De-Ben Rockson
"""

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load image
img = Image.open('C:/Users/Enqey De-Ben Rockson/Downloads/Purple Simple Warner & Spencer Blog Banner.png')
st.image(img, use_column_width=True)

# Display project rationale
st.write("""
         This API recommends a list of movies for users to watch based on user input in the sidebar

        ***Project Rationale*** : This is a proof of concept to illustrate how the algorithm works to recommend 
        items & content to clients on an e-commerce or online business platform   
""")

# Load dataset
data_path = 'https://raw.githubusercontent.com/Enqey/Diabetes_ml/main/diabetes.csv'
try:
    df = pd.read_csv(data_path)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# Show available movies in dataframe
st.subheader('***List of Movies in this catalogue***')
st.dataframe(df)

# Check if there are missing values in the selected columns
cols = ['Actors', 'Director', 'Genre', 'Title']
st.write("Are there any missing values?", df[cols].isnull().any().any())

# Create 'important_features' column
def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(data['Actors'][i] + ' ' + data['Director'][i] + ' ' + data['Genre'][i] + ' ' + data['Title'][i])
    return important_features

df['important_features'] = get_important_features(df)

# Calculate cosine similarity
cm = CountVectorizer().fit_transform(df['important_features'])
cs = cosine_similarity(cm)

# Movie recommendation section
st.subheader('***Movie Recommendation***')

option = st.sidebar.selectbox(
    'What movie do you want similar movies to?',    
    df['Title']
)

st.write('You have selected:', option)

# Get movie_id based on title
title = option
movie_id = df[df['Title'] == title].index[0]

# Calculate similarity scores
scores = list(enumerate(cs[movie_id]))

# Sort the scores and recommend movies
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
sorted_scores = sorted_scores[1:]  # Exclude the movie itself

st.write(f'The 7 recommended movies similar to "{title}" are:')
for j, item in enumerate(sorted_scores[:7]):
    movie_title = df.iloc[item[0]]['Title']
    st.write(f"{j + 1}. {movie_title}")

# Footer
st.write("""
    **Thanks for using this API, kindly Share this with your friends & Family**

    ***Developed by*** : Nana Ekow Okusu 
    
    ***Find me on***: Linkedin + Twitter
    
    ***Get in touch***: nanaokusu@insytecore.com
""")















