import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Demonstration 2025 in Indonesia",layout="wide")

st.title('Dashboard Demonstration 2025 in Indonesia')
st.subheader('Topic: A Unified Framework for Sentiment Analysis, Hate Speech Detection, and Entity Extraction: A Twitter-Based Case Study on the 2025 Indonesian Protests')
# st.markdown()
st.divider()
file_path_new='data_scrape.csv'

df = pd.read_csv(file_path_new,
                        header= 0)
# Data Clean
clean=pd.read_csv('Data/CleanDataFull.csv')
clean_cols = clean[['ID', 'full_text']].copy()
clean_cols = clean_cols.rename(columns={'full_text': 'clean_text'})
df = df.merge(clean_cols, on='ID', how='left')

# Hate
hate=pd.read_csv('predicted_result_full.csv')
hs_cols = hate.loc[:, 'HS':'Abusive_label'].copy()
hs_cols['ID'] = hate['ID'] 
df= df.merge(hs_cols, on='ID', how='left')
df['HS_label'] = df['HS_label'].map({0: 'non_hate', 1: 'hate'})

# Topik
topik=pd.read_csv('Topic/hasil_topik_per_tweet.csv')
topic_cols = topik.loc[:, 'topic':'topic_keywords'].copy()
topic_cols['ID'] = topik['ID'] 
df= df.merge(topic_cols, on='ID', how='left')


# Sentiment
sentiment=pd.read_csv('hasil_prediksi_sentiment_half.csv')
sentiment_cols = sentiment.loc[:, 'Predicted':'Predicted Label'].copy()
sentiment_cols['ID'] = sentiment['ID'] 
df= df.merge(sentiment_cols, on='ID', how='left')


# NER
ner=pd.read_csv('NER Full/hasil_ner_global.csv')
# st.dataframe(ner)

ner['entity_role'] = ner['entity'] + ' - ' + ner['label']
ner_grouped = ner.groupby('ID')['entity_role'].agg(lambda x: ', '.join(map(str, x))).reset_index()
df = df.merge(ner_grouped, on='ID', how='left')
df['created_at'] = pd.to_datetime(df['created_at'], format='%d/%m/%Y %H:%M')
df['hour'] = df['created_at'].dt.hour
df['hour'] = df['hour'].astype(int)
df['day'] = df['created_at'].dt.day_name()
# df.to_csv('final.csv')
# st.dataframe(df)

if "df" not in st.session_state:
    st.session_state['df'] = df



st.subheader("Distribution of Tweets per University")

total_tweet = len(df)
st.metric(label="Total Tweets (Overall)", value=total_tweet)




tabs1, tabs2, tabs3, tabs4= st.tabs( ["General Information", "Topic Modeling", "Sentiment Analysis", "Hate"])

with tabs1:
    import page1
    page1.show()
with tabs2:
    import page2
    page2.show()
with tabs3:
    import page3
    page3.show()
with tabs4:
    import page4
    page4.show()


