# Semantic-Search-Engine-for-Movie-Recommendations

Welcome to the Semantic Search Engine for Movie Recommendations repository! This project aims to provide personalized movie recommendations using advanced natural language processing techniques, by leeveraging sentence transformers for embeddings and the Streamlit library for creating an intuitive user interface. <br> <br>

Features:
1) Semantic Search: Uses sentence embeddings to understand the context and semantics of user queries, providing more accurate movie recommendations. <br>
2) Sentence Embeddings: The application uses sentence transformers to convert movie descriptions and user queries into high-dimensional vectors. These embeddings capture the semantic meaning of the text. <br>
3) Similarity Score: When a user inputs a query, the system calculates the cosine similarity between the query embedding and the movie embeddings to find the most relevant recommendations.
4) Streamlit Interface: The UI built with Streamlit allows users to input their preferences, view recommendations, and interact with the recommendation engine in real-time. <br>
5) Elasticsearch Integration: Movie data is indexed in Elasticsearch (Vector Database), allowing for quick and efficient retrieval of movie information based on user queries. <br>

In the output of the Streamlit Web App, it will show the user up to 10 movie recommendations based on the user searched query. <br>

Note: To run the streamlit framework, type streamlit run main.py in your terminal. <br>

References:

https://www.youtube.com/watch?v=MDEXYjKv7v4&t=50s <br>
https://www.datacamp.com/tutorial/streamlit <br>
https://github.com/gborn/Semantic-Search-Engine-Using-ElasticSearch <br>
https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html <br>

