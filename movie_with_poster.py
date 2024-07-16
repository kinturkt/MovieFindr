import streamlit as st
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import kaggle
import zipfile
import os

# Authenticate and create the dataset
kaggle.api.authenticate()
dataset = "harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows"
zip_file_name = "imdb-dataset-of-top-1000-movies-and-tv-shows.zip"
kaggle.api.dataset_download_files(dataset, path='./', unzip=False)

# Check if the zip file exists before attempting to unzip
if os.path.exists(zip_file_name):
    with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
        zip_ref.extractall(".")
    print("Dataset created successfully.")
else:
    print(f"File not found: {zip_file_name}")
    print("Files in current directory:", os.listdir('.'))

# Load the dataset
movies = pd.read_csv("imdb_top_1000.csv")

# Clean the columns
columns_to_clean = ['Series_Title', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']
for col in columns_to_clean:
    movies[col] = movies[col].fillna('unknown').astype(str)
    movies[col] = movies[col].str.replace('[^\w\s]', '', regex=True)

# Concatenate columns for embedding
movies['concatenated_text'] = movies[columns_to_clean].apply(lambda x: ' '.join(x), axis=1)

# Initialize the Hugging Face Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

st.title('Movie Semantic Search Engine')

# Generate embeddings and store them in a DataFrame
if 'embeddings' not in st.session_state:
    movies['embeddings'] = movies['concatenated_text'].apply(lambda x: model.encode(x).tolist())
    st.session_state['embeddings'] = movies['embeddings']
else:
    movies['embeddings'] = st.session_state['embeddings']

# Elasticsearch vector DB Setup
es = Elasticsearch("http://localhost:9200",
                    basic_auth=('elastic', 'elastic'))
index_name = 'movies'

# Define the mappings for your index
mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "overview": {"type": "text"},
            "IMDB_Rating": {"type": "float"},
            "genre": {"type": "text"},
            "director": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 768},
            "poster_link": {"type": "text"}
        }
    }
}

# Create the index if it doesn't exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)


# Bulk insert documents into Elasticsearch
def bulk_data(movies):
    for _, row in movies.iterrows():
        yield {
            "_index": index_name,
            "title": row["Series_Title"],
            "genre": row["Genre"],
            "director": row["Director"],
            "embedding": row["embeddings"],
            "poster_link": row["Poster_Link"]
        }

if not st.session_state.get('es_indexed', False):
    bulk(es, bulk_data(movies))
    st.session_state['es_indexed'] = True

# Semantic search function
def semantic_search(query, movies_df, top_k=5):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], np.array(movies_df['embeddings'].tolist()))[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return movies_df.iloc[top_indices]

with st.form(key='search_form'):
    user_query = st.text_input('Enter a movie description:')
    top_k = st.slider('Number of search results', 1, 10, 5)
    submit_button = st.form_submit_button(label='Search')

if submit_button or user_query:
    with st.spinner('Searching for movies...'):
        results = semantic_search(user_query, movies, top_k=top_k)

        # Check if the results DataFrame is not empty
        if not results.empty:
            for _, result in results.iterrows():
                # Display the movie details
                st.write(f"### {result['Series_Title']}")
                st.image(result['Poster_Link'], width=150)
                st.write(f"**Genre**: {result['Genre']}")
                st.write(f"**Overview**: {result['Overview']}")
                st.write(f"**IMDb Rating**: {result['IMDB_Rating'] if pd.notna(result['IMDB_Rating']) else 'N/A'}")
                st.write(f"**Star Cast**: {', '.join(filter(pd.notna, [result.get('Star1'), result.get('Star2'), result.get('Star3'), result.get('Star4')]))}")
                st.write(f"**Director**: {result['Director']}")
                st.write("---")
        else:
            st.write("No results found.")