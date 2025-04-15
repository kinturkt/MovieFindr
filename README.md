# MovieFindr

Welcome to the MovieFindr repository! This project aims to provide personalized movie recommendations using semantic search and advanced natural language processing techniques, by leeveraging sentence transformers for embeddings and the Streamlit library for creating an intuitive user interface. <br>

Features:
1) <b>Semantic Search:</b> Uses sentence embeddings to understand the context and semantics of user queries, providing more accurate movie recommendations. <br>
2) <b>Sentence Embeddings:</b> The application uses **Sentence Transformers** to convert movie descriptions and user queries into high-dimensional vectors. These embeddings capture the semantic meaning of the text. <br>
3) <b>Similarity Score:</b> When a user inputs a query, the system calculates the cosine similarity between the query embedding and the movie embeddings to find the most relevant recommendations.
4) <b>Streamlit Interface:</b> The UI built with Streamlit allows users to input their preferences, view recommendations, and interact with the recommendation engine in real-time. <br>
5) <b>Elasticsearch Integration:</b> Movie data is indexed in Elasticsearch (Vector Database), allowing for quick and efficient retrieval of movie information based on user queries. <br>

In the output of the Streamlit Web App, it will show the user up to 10 movie recommendations based on the user searched query. <br>

Note: To run the streamlit framework, type <b> streamlit run main.py </b> in your terminal. <br>

In addition, if you also want to get the recommended movie poster, run another python file --> Type <b> streamlit movie_with_poster.py </b> in your terminal. <br>

Also, all the requirements are shown in requirements.txt file which should be installed in the system to run above codes.

References:

https://www.youtube.com/watch?v=MDEXYjKv7v4&t=50s <br>
https://www.datacamp.com/tutorial/streamlit <br>
https://github.com/gborn/Semantic-Search-Engine-Using-ElasticSearch <br>
https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html <br>

**Author: Kintur Shah** <br>
[LinkedIn](https://www.linkedin.com/in/kintur-shah/) | [Github](https://github.com/kinturkt)
