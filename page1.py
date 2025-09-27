# Informasi umum

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

def show():
    df = st.session_state['df']
   
    # st.dataframe(df, use_container_width=True)

    st.title("Mapping Tweets by Time Based on Hours")


    hist = (
    alt.Chart(df)
    .mark_bar(color='skyblue')
    .encode(
        x=alt.X('hour:O', title='Hour (0-23)'), 
        y=alt.Y('count()', title='Total Tweets'),
        tooltip=['hour', 'count()']
    )
    .properties(
        width=400,
        height=300
    )
    )

    st.altair_chart(hist, use_container_width=True)

    st.title('Distribution of Tweets per Day')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    selected_days = st.multiselect(
        "Choose Day:", day_order, default=day_order, 
    key="day_selector"
    )

    df_filtered = df[
        (df["day"].isin(selected_days))
    ]

    if df_filtered.empty:
        st.warning("No data matches the selected filter.")
    else:
        bar_chart = (
            alt.Chart(df_filtered)
            .mark_bar()
            .encode(
                x=alt.X('day:N', sort=day_order, title='Day'),
                y=alt.Y('count():Q', title='Total Tweet'),
                tooltip=['day', 'count()']
            )
            .properties(
                width=600,
                height=400,
            )
        )
        st.altair_chart(bar_chart, use_container_width=True)




    # st.title('Distribution of Tweets per Day')
    # day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # selected_users = st.multiselect(
    #     "Choose Username:", usernames, default=list(usernames), 
    #     key="user_selector"
    # )

    # selected_days = st.multiselect(
    #     "Choose Day:", day_order, default=day_order, 
    #     key="day_selector"
    # )

    # df_filtered = df[
    #     (df["username"].isin(selected_users)) &
    #     (df["day"].isin(selected_days))
    # ]

    # if df_filtered.empty:
    #     st.warning("No data matches the selected filter.")
    # else:
    #     line_chart = (
    #         alt.Chart(df_filtered)
    #         .mark_line(point=True)
    #         .encode(
    #             x=alt.X('day:N', sort=day_order, title='Day'),
    #             y=alt.Y('count():Q', title='Total Tweet'),
    #             color='username:N',
    #             tooltip=['day', 'username', 'count()']
    #         )
    #         .properties(
    #             width=600,
    #             height=400,
    #         )
    #     )
    #     st.altair_chart(line_chart, use_container_width=True)

    st.title("Heatmap Tweet Activity")

    heatmap_data = (
        df.groupby(['day', 'hour'])
        .size()
        .reset_index(name='count')
        .pivot_table(index='day', columns='hour', values='count', fill_value=0)
        .stack()
        .reset_index(name='count')
    )

    heatmap_chart = (
        alt.Chart(heatmap_data)
        .mark_rect()
        .encode(
            x=alt.X('hour:O', title='Hour (0–23)'),
            y=alt.Y('day:N', sort=day_order, title='Day'),
            color=alt.Color('count:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['day', 'hour', 'count']
        )
        .properties(
            width=700,
            height=300,
            title="Tweet Activity per Hour & Day"
        )
    )

    st.altair_chart(heatmap_chart, use_container_width=True)
    st.title('Sentiment Distribution per Day')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day'] = pd.Categorical(df['day'], categories=day_order, ordered=True)

    grouped = df.groupby(['day', 'Predicted Label']).size().reset_index(name='Total Tweet')
    chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X('day:N', title='Day'),
        y=alt.Y('Total Tweet:Q', title='Total Tweet'),
        color=alt.Color('Predicted Label:N', title='Sentimen'),
        tooltip=['day', 'Predicted Label', 'Total Tweet'],
    ).properties(

    ).configure_axis(
        labelAngle=-45
    ).encode(
        x=alt.X('day:N', sort=day_order),
        column='Predicted Label:N' 
    )

    chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X('day:N', title='Day', sort=day_order),
        y=alt.Y('Total Tweet:Q', title='Total Tweet'),
        color=alt.Color('Predicted Label:N', title='Sentimen'),
        tooltip=['day', 'Predicted Label', 'Total Tweet']
    )

    chart = chart.encode(
        x=alt.X('day:N', title='Day', sort=day_order),
        y='Total Tweet:Q',
        color='Predicted Label:N'
    ).properties(
        width=600,
        height=400,

    )

    st.altair_chart(chart, use_container_width=True)




    total_counts = df['entity_role'].value_counts().reset_index()
    total_counts.columns = ['entity_role', 'Total Count']


    summary = total_counts.copy()


    st.title("Most Entity Roles")
    st.dataframe(summary)

