import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
import joblib
#import shap

def score_text(score):
    if score == 0:
        return "0 – 5 : Never going to watch it", "streamlit_visuals_needed/emojis/emoji1.png"
    elif score == 1:
        return "5 – 6.5 : If there’s nothing else on", "streamlit_visuals_needed/emojis/emoji2.png"
    elif score == 2:
        return "6.5 – 7.5 : Might be interested", "streamlit_visuals_needed/emojis/emoji3.png"
    elif score == 3:
        return "7.5 – 8.5 : Great content", "streamlit_visuals_needed/emojis/emoji4.png"
    elif score == 4:
        return "8.5+ : Excellent, potentially all-time great content", "streamlit_visuals_needed/emojis/emoji5.png"


def main():
    st.set_page_config(page_title="Movie & TV Show IMDB Rating Predictor", layout="wide")
    st.image("streamlit_visuals_needed/StreamOracle_logo.png", width=1200)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Select an option (from this list of completely made up movies/TV shows) to see the predicted IMDB rating and votes")

    with col2:
        additional_info_df = pd.read_csv('data/Actual_movie_show_info.csv')
        option_title = st.selectbox("Choose an option:", additional_info_df["title"].values)

    selected_movie = additional_info_df[additional_info_df["title"] == option_title]
    selected_index = selected_movie.index.values[0]

    for index, row in selected_movie.iterrows():
        st.markdown("### Details")
        st.markdown(f"<b>Title:</b> {row['title']}", unsafe_allow_html=True)
        st.markdown(f"<b>Description:</b> {row['description']}", unsafe_allow_html=True)
        st.markdown(f"<b>Stars:</b> {row['starring']}", unsafe_allow_html=True)
        st.markdown(f"<b>Directed By:</b> {row['director']}", unsafe_allow_html=True)
        st.markdown(f"<b>Genres:</b> {row['genres']}", unsafe_allow_html=True)
        st.markdown(f"<b>Type:</b> {row['type']}", unsafe_allow_html=True)
        st.markdown(f"<b>Seasons:</b> {row['seasons']}", unsafe_allow_html=True)
        st.markdown(f"<b>Runtime:</b> {row['runtime']}", unsafe_allow_html=True)
        st.markdown(f"<b>Production countries:</b> {row['production_countries']}", unsafe_allow_html=True)
        st.markdown(f"<b>Total size of cast:</b> {row['Other_Actors']}", unsafe_allow_html=True)

    if st.button("Predict"):
        votes_df = pd.read_csv('data/prepared_content_model_ready_votes.csv')
        score_df = pd.read_csv('data/prepared_content_model_ready_score.csv')

        votes_model = joblib.load('models/XGBRegressor_Votes.joblib')
        score_model = joblib.load('models/XGBClassifier_Score.joblib')

        selected_votes_features = votes_df.iloc[[selected_index]]
        selected_score_features = score_df.iloc[[selected_index]]

        predicted_votes = votes_model.predict(selected_votes_features)[0]
        predicted_score = score_model.predict(selected_score_features)[0]
        predicted_score_text, emoji_path = score_text(predicted_score)

        # Display IMDB rating as text
        st.markdown(f"### **IMDB Rating:** {predicted_score_text}")

        # Display emoji
        st.image(emoji_path, width=50)

        # Display IMDB votes as a progress bar
        max_votes = 200000  # Set an arbitrary maximum number of votes
        progress_bar = st.progress(0)
        progress_bar.progress(np.exp(predicted_votes) / max_votes)
        st.write(f"### IMDB Votes: {round(np.exp(predicted_votes))}")

if __name__ == "__main__":
    main()