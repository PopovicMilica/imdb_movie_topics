# imdb_movie_topics

## Introduction

This project has two parts: first, we scrape the data for top-rated movies from IMDb and their corresponding Wikipedia pages; then we use these data to build various topic models in order to extract movie topics from their plots. 

In the first part, we access the IMDb page with 250 top-rated movies and scrape the URLs of each individual movie. Then, we scrape the basic info, such as movie title, genre, and release date, from each movie page. We then proceed to the synopsis page and scrape a detailed storyline of each movie, which we use later to build the topic models. Moreover, we also scrape the individual movie plots from the Wikipedia pages dedicated to these movies. 

In the second part, we use the movie plots to extract the latent movie topics using different topic models such as Latent Semantic Indexing, Latent Dirichlet Allocation, and Non-Negative Matrix Factorization.

## File description

-  **movie_webscraping_utils.py**: Python file containing all the functions that we used in the Jupyter notebook `Scraping_movie_info_from_IMDb_and_Wikipedia_pages.ipynb` for webscraping movie data.
-  **text_preprocessing_utils.py**: Python file containing all the functions that we used in the first part of the Jupyter notebook `Movie_clustering_based_on_plot_summaries.ipynb` for preprocessing text data.
-  **topic_modeling_utils.py**: Python file containing all the functions that we used in the second part of the Jupyter notebook `Movie_clustering_based_on_plot_summaries.ipynb` to generate and visualize topic models.
-  **ldamallet.py**: Starting with Gensim version 4.0 a wrapper function `gensim.models.wrappers.ldamallet` was removed, so I copied the source code for that wrapper function directly from the last version of Gensim where it was supported (version 3.8.3) and included it directly in a project folder, so I can continue to use it without a need to downgrade my current version of the gensim library.
-  **Scraping_movie_info_from_IMDb_and_Wikipedia_pages.ipynb**: Main Jupyter notebook for scraping movie data.
-  **Movie_clustering_based_on_plot_summaries.ipynb**: Main Jupyter notebook for building and visualizing topic models.
-  **data/**: Folder containing movie data file `top_250_movies.csv`.
-  **ngram_models/**: Folder containing ngram_models that are built when calling prepare_corpus function in `topic_modeling_utils.py`.

## Software requirements
Here is the list of libraries needed to run the code: numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, regex, gensim, spacy, nltk, ldamallet, bs4, urllib.
