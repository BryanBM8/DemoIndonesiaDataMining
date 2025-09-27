# Hate

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import altair as alt

def show():
    df = st.session_state['df']
    st.header('Percentage Distribution of Hate')
    hate_counts = df['HS_label'].value_counts()
    hate_percentages = (hate_counts / hate_counts.sum()) * 100

    hate_df = pd.DataFrame({
        'hate': hate_counts.index,
        'Percentage': hate_percentages.values,
        'Count': hate_counts.values
    })

    fig = px.pie(
        hate_df,
        names='hate',
        values='Count',  
        color='hate',
        title="General Hate Distribution (Based on Number)",
        hole=0.4
    )

    fig.update_traces(
        textinfo='percent+label',
        hovertemplate='%{label}<br>Total: %{value}<extra></extra>'
    )

    st.plotly_chart(fig)

    

    df = df.dropna(subset=['full_text', 'HS_label'])

    st.title("Tweets based on Hate")

    hates = df['HS_label'].unique()
    selected_hate = st.radio("Choose hate or not hate", sorted(hates), key="selected_hate")

    filtered_df = df[
        (df['HS_label'] == selected_hate)
    ]

    st.subheader(f"Tweet with {selected_hate} label")
    st.write(f"Total found tweet: {len(filtered_df)}")

    st.dataframe(
        filtered_df[['full_text', 'HS_label']].reset_index(drop=True),
        use_container_width=True
    )




    df_hs = df[df['HS_label'] == 'hate'].copy()

    df_hs['clean_text'] = df_hs['clean_text'].astype(str)

    combined_text = " ".join(df_hs['clean_text'].dropna())

    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(combined_text)

    st.subheader("WordCloud of Hate Speech Tweets")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

   

    df_hs = df[(df['HS_label'] == 'hate') & df['entity_role'].notna()].copy()

    df_hs['entity_list'] = df_hs['entity_role'].apply(lambda x: [e.strip() for e in str(x).split(',')])

    df_exp = df_hs.explode('entity_list')
    df_exp = df_exp.rename(columns={'entity_list': 'Entity|Role'})

    entity_user_counts = df_exp.groupby(['Entity|Role']).size().reset_index(name='Count')
    
    st.subheader("Total Entity Target Hate Speech")
    st.dataframe(entity_user_counts.sort_values(by='Count', ascending=False))

    hate_df = filtered_df[filtered_df['HS_label'] == 'hate']


    grouped = hate_df.groupby(['day']).size().reset_index(name='Total Tweet')

    chart = alt.Chart(grouped).mark_bar().encode(
    x=alt.X('day:N', title='Day'),
    y=alt.Y('Total Tweet:Q', title='Total Tweet'),
    tooltip=['day:N', 'Total Tweet:Q']
    ).properties(
        width=400,
        height=300,
        title="Hate Tweet Distribution per Day"
    ).configure_axisX(
        labelAngle=-45
    )

    st.altair_chart(chart, use_container_width=True)

    grouped = hate_df.groupby(['hour']).size().reset_index(name='Total Tweet')


    
    chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X('hour:O', title='Hour (0–23)'),
        y=alt.Y('Total Tweet:Q', title='Total Tweet'),
        tooltip=['hour:O', 'Total Tweet:Q']
    ).properties(
        width=400,
        height=300,
        title="Hate Tweet Distribution per Hour"
    )

    st.altair_chart(chart, use_container_width=True)