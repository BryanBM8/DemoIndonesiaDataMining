# Topic Modeling

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
import matplotlib.colors as mcolors


def show():
    df = st.session_state['df']
    combined_text = " ".join(df['clean_text'].dropna().astype(str))


    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        collocations=False,  
        max_words=200        
    ).generate(combined_text)


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Word Cloud of Clean Text", fontsize=16)

    st.pyplot(fig)


    
    st.header("Most Discussed Topics")
    df = df.dropna(subset=['topic_keywords'])

    Topic_terbanyak = df['topic_keywords'].value_counts().reset_index()
    Topic_terbanyak.columns = ['Topic', 'Total']

    st.subheader("List of Most Popular Topics")
    st.dataframe(Topic_terbanyak)

    st.subheader("Visualization of Popular Topics")
    st.bar_chart(Topic_terbanyak.set_index('Topic'))




    st.header('Tweet Each Keyword')
    df = df.dropna(subset=['topic_keywords', 'full_text'])

    keyword_counts = df['topic_keywords'].value_counts()
    min_count = 3 
    valid_keywords = keyword_counts[keyword_counts >= min_count].index.tolist()

    selected_keywords = st.selectbox("Choose Combination of Keywords", valid_keywords)

    filtered_df = df[df['topic_keywords'] == selected_keywords]


    st.subheader(f"Tweet with Keywords: {selected_keywords}")
    st.write(f"Number of tweets found: {len(filtered_df)}")
    st.dataframe(filtered_df[[ 'full_text', 'topic_keywords']].reset_index(drop=True))




    