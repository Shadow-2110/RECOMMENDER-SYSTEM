import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity


API_KEY = "14270b3c76be75bae1ddb3e98f600c5a"

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)



st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    text-align: center;
    color: white;
}

.stButton>button {
    width: 100%;
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():

    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)

    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    df = pd.read_pickle("df.pkl")

    return df, tfidf_matrix, indices, tfidf


df, tfidf_matrix, indices, tfidf = load_data()


def fetch_poster(movie_name):

    try:

        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"

        response = requests.get(url, timeout=10)

        data = response.json()

        if "results" not in data:
            return None

        if len(data["results"]) == 0:
            return None

        poster_path = data["results"][0].get("poster_path")

        if poster_path is None:
            return None

        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

        return full_path

    except requests.exceptions.RequestException as e:

        print("Error:", e)

        return None


def recommend(movie, n=10):

    if movie not in indices:
        return [], []

    idx = indices[movie]

    similarity = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    movie_indices = similarity.argsort()[::-1][1:n+1]

    recommended_movies = []
    recommended_posters = []

    for i in movie_indices:

        movie_title = df.iloc[i]['title']

        recommended_movies.append(movie_title)

        poster = fetch_poster(movie_title)

        recommended_posters.append(poster)

    return recommended_movies, recommended_posters


st.title("🎬 Movie Recommender System")

st.write("Get recommendations based on your favorite movie")


movie_list = sorted(df['title'].unique())

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

number = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

if st.button("Recommend"):

    names, posters = recommend(selected_movie, number)

    st.subheader("Recommended Movies")

    cols = st.columns(5)

    for i in range(len(names)):

        with cols[i % 5]:

            if posters[i]:

                st.image(posters[i])

            else:

                st.write("No Image Available")

            st.caption(names[i])