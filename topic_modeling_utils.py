# Import modules
import regex as re
import matplotlib.pyplot as plt
import numpy as np
import os.path

from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LsiModel
from gensim.models import LdaModel
from sklearn.decomposition import NMF
from ldamallet import *
import os
os.environ.update({'MALLET_HOME':r'C:/mallet/'})


def prepare_corpus(norm_documents, min_count=5, threshold=10,
                   scoring='npmi', ngram_value=3, save_ngram_models=False):
    """
    Takes in tokenized and preprocessed documents which are in the format
    of the list of lists, where the length of a list is equal to the number
    of documents. Then it creates ngrams to the degree set by `ngram_value`. 
    If the function variable `save_ngram_models` is set to True it will
    create folder './ngram_models/' and save ngram_models to that folder.
    At last it will build our movie corpus and term dictionary and return them
    as a function outputs.
    
    Parameters
    ----------
    norm_documents: list of lists
        Tokenized and preprocessed documents which are in the format of
        the list of list, where the length of a list is equal to the
        number of documents.
    min_count: float, optional
        Ignore all words and bigrams with total collected count lower
        than this value.
    threshold: float, optional
        Represent a score threshold for forming the phrases (higher
        means fewer phrases). A phrase of words `a` followed by `b` is
        accepted if the score of the phrase is greater than threshold.
        Heavily depends on concrete scoring-function, see the `scoring`
        parameter.
        Default value is 10.
    scoring : {'default', 'npmi', function}, optional
        Specify how potential phrases are scored. `scoring` can be set
        with either a string that refers to a built-in scoring function,
        or with a function with the expected parameter names. Two
        built-in scoring functions are available by setting `scoring`
        to a string.
    ngram_value: int, optional
        A number that determines if we want to search for ngrams of a
        higher values and include them to our dictionary. A value of 3 
        would mean that dictionary will include all single words,
        bigrams and trigrams that met the criterion for elimination.
        Default value is 3.
    save_ngram_models: boolean, optional
        If set to true, it saves all the ngram_models built in a folder
        called './ngram_models/'. Should be set to True if we are
        planning to predict topics on unseen data.
        Default value: False
    """ 
    if ngram_value == 1:
        norm_corpus = norm_documents
    else:
        norm_corpus = norm_documents
        delimeters = '_.,;:#^'
        for i in range(ngram_value - 1):
            ngram = Phrases(norm_corpus, min_count=min_count,
                            threshold=threshold, delimiter=delimeters[i],
                            connector_words=ENGLISH_CONNECTOR_WORDS)
            ngram_model = Phraser(ngram)

            if save_ngram_models:
                path = './ngram_models/'
                if not os.path.exists(path):
                    os.makedirs(path)
                ngram_model.save(
                    path + 'ngram_model' + str(i+1) + '.pkl')
            norm_corpus = list(ngram_model[norm_corpus])

    norm_corpus = [
        [re.sub(r"[_.,;:#^]+", "_", w) for w in d] for d in norm_corpus]
    
    # Creating the term dictionary of our corpus, where every unique
    #  term is assigned an index.
    dictionary = Dictionary(norm_corpus)
    return dictionary, norm_corpus

def adjust_tokenized_corpus(norm_corpus, dictionary):
    '''Splits the bigrams and trigrams, that were removed after we applied
    `filter_extreme` function on our dictionary, to unigrams.'''
    corpus_higher_ngrams = set(
        [word for i in range(len(norm_corpus))
         for word in norm_corpus[i]  if '_' in word])
    dict_higher_ngrams = set(
        [word for word in list(dictionary.values()) if '_' in word])
    filtered_higher_ngrams = corpus_higher_ngrams - dict_higher_ngrams
    nn_corp = []
    for movie in norm_corpus:
        temp_list = []
        for word in movie:
            if word in filtered_higher_ngrams:
                word = word.split('_')
            else:
                word = [word]
            temp_list = temp_list + word        
        nn_corp.append(temp_list)
    return nn_corp


def compute_coherence_values(doc_term_matrix, dictionary, texts,
                             coherence_metric, start, stop, step,
                             model_name, n_passes=1, iterations=50,
                             seed=None):
    '''
    Compute specified coherence metric for various number of topics.
    Also, returns all the models run during this process as a list and
    also as a list coherence values achieved for each model.
    
    Parameters
    ----------
    doc_term_matrix: either bow_corpus or tfidf_corpus
    dictionary : Gensim dictionary
    texts : List of tokenized input documents.
    start: Min num of topics
    stop : Max num of topics
    step: Increase the number of topics each step by
    model_name: {'lsi', 'lda', 'lda_mallet', 'nmf'}
    n_passes: num of passes through the data
    iterations: num of training iterations
    seed: seed number. 
    
    Returns
    -------
    model_list : All models trained returned as list
    coherence_values_list : Coherence values corresponding to each of
    the models trained.
    '''
    coherence_values_list = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        if model_name == 'lsi':
            model = LsiModel(doc_term_matrix, id2word=dictionary,
                             num_topics=num_topics, power_iters=n_passes)
        elif model_name == 'lda':
            model = LdaModel(doc_term_matrix, id2word=dictionary,
                             num_topics=num_topics, passes=n_passes,
                             iterations=iterations)
        elif model_name == 'lda_mallet':
            MALLET_PATH = 'C:\\mallet\\bin\\mallet'
            model = LdaMallet(mallet_path=MALLET_PATH, corpus=doc_term_matrix,
                              num_topics=num_topics, id2word=dictionary,
                              iterations=iterations, random_seed=seed)
        elif model_name == 'nmf':
            nmf = NMF(n_components=num_topics, solver='cd', random_state=seed,
                      max_iter=iterations, alpha=.1, l1_ratio=.85)
            dense_matrix = gensim.matutils.corpus2dense(
                doc_term_matrix, len(dictionary)).T

            model = nmf.fit(dense_matrix)

        # Appending trained model to the model list
        model_list.append(model)

        if not model_name == 'nmf':
            top_terms, weights = get_topn_words_and_weights_gensim_model(
                model, n_top_terms=20)
        else:
            top_terms, weights = get_topn_words_and_weights_scikit_model(
                model, list(dictionary.values()), n_top_terms=20)

        if coherence_metric == 'c_v':
            coherence_model = CoherenceModel(
                topics=top_terms, texts=texts, dictionary=dictionary,
                coherence=coherence_metric)
        elif coherence_metric == 'u_mass':
            corpus_model = model[doc_term_matrix]
            coherence_model = CoherenceModel(
                topics=top_terms, corpus=corpus_model, dictionary=dictionary,
                coherence=coherence_metric)
        # Appending calculated coherence value for the current model
        coherence_values_list.append(coherence_model.get_coherence())
    return model_list, coherence_values_list


def plot_coherence_values_vs_num_topics(coherence_values, start, stop, step,
                                        coherence_metric_name=None, title=''):
    '''Plots coherence values as a function of the number of topics.'''
    x = range(start, stop, step)
    plt.plot(x, coherence_values,  marker='o')
    plt.xticks(x)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.title('Coherence values vs num topics' + title)
    if coherence_metric_name is not None:
        plt.legend([coherence_metric_name], loc='best')
    plt.show()


def get_topn_words_and_weights_scikit_model(model, feature_names,
                                            n_top_terms=10):
    '''Retrieves n top terms and their weights for a specific
    scikit_learn model.'''
    ntopics = len(model.components_)
    # Retrieve indices sorted in ascending order for each topic, then select the
    # last n_top_terms indices and reverse their order, so that the index 
    # corresponding to the bigest term weight is now on the first spot
    top_term_inds = model.components_.argsort()[:, :-n_top_terms - 1:-1].tolist()
    # Select n_top_terms
    top_terms = [
        [feature_names[i] for i in top_term_inds[n]] for n in range(ntopics)] 
    # Select their corresponding weights
    weights = [
        model.components_[i][top_term_inds[i]].tolist() for i in range(ntopics)]
    return top_terms, weights


def get_topn_words_and_weights_gensim_model(model, n_top_terms=10):
    '''Retrieves n top terms and their weights for a specific
    gensim model.'''
    ntopics = len(model.get_topics())
    topics = [
        [(term, round(wt, 3)) for term, wt in model.show_topic(n, topn=n_top_terms)]
        for n in range(0, ntopics)]
    top_terms = np.array(topics)[:, :, 0].tolist()
    weights = np.array(topics)[:, :, 1].astype(np.float).tolist()
    return top_terms, weights


def plot_top_words(top_features, weights, title):
    '''Plots top words and their weights.'''
    n_topics = len(top_features)
    n_top_words = len(top_features[0])
    
    if n_topics % 5 == 0:
        dim1 = int(n_topics / 5)
    else:
        dim1 = int(n_topics / 5) + 1
    
    if dim1 == 1 and n_topics % 5 != 0:
        dim2 = n_topics % 5
    else:
        dim2 = 5
    
    fig, axes = plt.subplots(dim1, dim2, figsize=(30, n_top_words*dim1),
                             sharex=True)        
    axes = axes.flatten()
    # Turn all axis off, as we build graph we will turn each individual
    # axis on, that prevents having empty subplots being shown, e.g. if
    # we have only 6 topics and (2, 5) grid
    for ax in axes:
        ax.set_axis_off()

    # Go through each topic and visualize top words and respective weights
    for topic_idx in range(n_topics):
        ax = axes[topic_idx]
        # Turn axis on
        ax.set_axis_on()
        # Make a horizontal bar plot
        ax.barh(top_features[topic_idx], weights[topic_idx], height=0.7)
        ax.set_title(f'Topic {topic_idx +1}', fontdict={'fontsize': 25})
        # labels read top-to-bottom
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=23)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        
    fig.suptitle(title, fontsize=35)
    fig.tight_layout(h_pad=5.0, rect=[0, 0.03, 1, 0.95])
    plt.show()
