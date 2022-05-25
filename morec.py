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


img = Image.open('C:/Users/Enqey De-Ben Rockson/Downloads/Purple Simple Warner & Spencer Blog Banner.png')
st.image(img,use_column_width = True)


st.write("""
         This API recommends a list of movies for users to watch based on user input in the sidebar
         
        ***Project Rationale*** : This is a proof of concept to illustrate how the algorithm works to recommend 
        items & content to clients on an e-commerce or online business platform   
    
""" )

df = pd.read_csv('C:/Users/Enqey De-Ben Rockson/Downloads/IMDB-Movie-Data.csv')

st.subheader('***List of Movies in this catalogue***')

st.dataframe(df)

Shape = df.shape

cols = ['Actors','Director','Genre','Title']

df[cols].head(3)

df[cols].isnull().values.any()


#Filter = st.sidebar.multiselect(
#        'Filter table',
#        (df[cols].head())
#    )

#st.sidebar.write('See table with columns:', Filter)

#st.dataframe(Filter)


def get_important_features(data):
    important_features = []
    for i in range(0,data.shape[0]):
        important_features.append(data['Actors'][i]+' '+data['Director'][i]+' '+data['Genre'][i]+' '+data['Title'][i])
        
    return important_features

df['important_features']= get_important_features(df)

df.head(3)

cm = CountVectorizer().fit_transform(df['important_features'])

cs = cosine_similarity(cm)

st.subheader('***Movie Recommendation***')

option = st.sidebar.selectbox(
    'What movie do you want similar movies to?',    
    (df.Title)
    )

st.write('You have selected:' , option)

#option = st.sidebar.text_input(
#    'Insert title',    
 #   "The Amazing Spider-Man"
  #  (df.Title)
#    )

#st.write('You have selected:' , option)


title = option

movie_id = df[df.Title == title]['Movie_id'].values[0]

scores = list(enumerate(cs[movie_id]))

sorted_scores = sorted(scores, key =lambda x: x[1], reverse = True)

sorted_score = sorted_scores[1:]


j = 0 
st.write('The 7 recommended movies similar to', title, 'are:\n')
for item in sorted_scores:
    movie_title = df[df.Movie_id == item[0]]['Title'].values[0]
    st.write(j+1,movie_title)
    j = j+1
    if j>6:
        break

st.write("""
         
         **Thanks for using this API, kinldy Share this with your friends & Family**
         
         ***Developed by*** : Nana Ekow Okusu 
         
         ***Find me on***: Linkedin + Twitter
         
         ***Get in touch***: nanaokusu@insytecore.com
        """ )
















