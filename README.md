# imdb_movie_topics

## Introduction

This project has two parts: first, we scrape the data for top rated movies from imdb and their corresponding Wikipedia pages; then we use these data to build various topic models in order to extract movie topics from their plots. 

In the first part, we access the IMDb page with 250 top rated movies and scrape the urls of each individual movie. Then, we scrape the basic info, such as movie title, genre and realese date, from each movie page. We then proceed to the synopsis page and scrape a detailed storyline of each movie, which we use later to build the topic models. Moreover, we also scrape the individual movie plots from the Wikipedia pages dedicated to those movies. 

In the second part, we use these movie plots to extract the latent movie topics using different topic models such as Latent Semantic Indexing, Latent Dirichlet Allocation, and Non-Negative Matrix Factorization.

## File description

-  **movie_webscraping_utils.py**: Python file containing all the functions that we used in the Jupyter notebook `Scraping_movie_info_from_IMDb_and_Wikipedia_pages.ipynb` for webscraping movie data.
-  **text_preprocessing_utils.py**: Python file containing all the functions that we used in the first part of the Jupyter notebook `Movie_clustering_based_on_plot_summaries.ipynb` for preprocessing text data.
-  **topic_modeling_utils.py**: Python file containing all the functions that we used in the second part of the Jupyter notebook `Movie_clustering_based_on_plot_summaries.ipynb` to generate and visualize topic models.
-  **Scraping_movie_info_from_IMDb_and_Wikipedia_pages.ipynb**: Main Jupyter notebook for scraping movie data.
-  **Movie_clustering_based_on_plot_summaries.ipynb**: Main Jupyter notebook for building and visualizing topic models.
-  **data/**: Folder containing movie data file `top_250_movies.csv`.
-  **ngram_models/**: Folder containing ngram_models that are built when calling prepare_corpus function in `topic_modeling_utils.py`.

## Software requirements
Here is the list of libraries needed to run the code: numpy, pandas, scikit-learn, matplotlib, seaborn, regex, gensim, spacy, nltk, ldamallet, bs4, urllib.
