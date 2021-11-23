# Import modules
import unicodedata
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import spacy
import string
import re
# import most common stemmers
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer


def remove_accented_chars(text):
    '''Remove any accented characters/letters. e.g. converting é to e.'''
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


def merge_valid_hyphenated_words(hyphenated_compound):
    '''Removes the hyphen and merges hyphenated compound into
       one word if that word is a valid one, otherwise breaks up
       hyphenated compound into words by replacing hyphen with
       a white space.
       Eg. compound 'co-operate' will be returned as 'cooperate', 
       but compound 'up-to-date' will be returned as 'up to date' '''
    if wordnet.synsets(hyphenated_compound.replace('-', '')):
        return hyphenated_compound.replace('-', '')
    else:
        return hyphenated_compound.replace('-', ' ')
    

def replace_hyphenated_compounds_in_text(text):
    '''Finds all hyphenated compounds inside a text. Then
       for each found hyphenated compound  removes the hyphen 
       and merge hyphenated compound into one word if that word
       is a valid one, otherwise breaks up hyphenated compound 
       into words by replacing hyphen with a white space. At last,
       replace found hyphenated compounds with a corrected ones in
       the text.
       Eg. 'We need a stronger co-operation in order to bring everyone up-to-date'
       will be turned into:
       'We need a stronger cooperation in order to bring everyone up to date.'
    '''
    hyphenated_compounds = re.findall(r'\b\S+-\S+\b', text)
    for hyphenated_compound in hyphenated_compounds:
        text = text.replace(hyphenated_compound,
                            merge_valid_hyphenated_words(hyphenated_compound))
    return text


def preprocess_text(text, hyphenated_compounds_replacement=True,
                    accented_char_removal=True, stop_words_removal=True):
    '''
    Removes from the text all the punctuation, special characters and
    all the tokens that don't contain letters. Also, makes all the
    tokens lowercased. Also, if appropriate variable is set to True
    it can perform:
    * removal of accented_characters (e.g. converting é to e),
    * removal of stop words, 
    * hyphenated compounds replacement
     (e.g. 'co-operate' will be returned as 'cooperate')
    
    Returns the fully preprocessed text.
    '''
    # Replace a special character
    text = text.replace('’', "'")
    #text = text.replace('”', ' ')
    #text = text.replace('“', ' ')

    # Remove tricky substrings
    for substring in ["'s", "'m", "'ll", "'t", "'ve", "'d", "'re"]:
        text = text.replace(substring, '')

    # Remove any accented characters/letters
    if accented_char_removal:
        text =  remove_accented_chars(text)

    # Replace hyphenated compounds
    if hyphenated_compounds_replacement:
        text = replace_hyphenated_compounds_in_text(text)

    # Remove punctuation
    translator = str.maketrans(
        {key: ' ' for key in string.punctuation+'—”“'})
    text_clean = text.translate(translator).lower()

    # Keeping only text tokens that are words
    text_only_words =  " ".join(
        [w for w in text_clean.split() if re.search(r'[a-zA-Z]', w)])

    if stop_words_removal:
        # Remove stopwords from the text
        stop_words = (set(stopwords.words('english')))
        text_only_words = " ".join([
            w for w in text_only_words.split() if not w in stop_words])
    return text_only_words


def stem_text(text, stemmer_name='Snowball', language='english'):
    '''
    Stems all the words in the text to their stem base.

    Parameters
    ----------
    text: text on which stemming should be performed.
    stemmer_name: {'Porter', 'Snowball', 'Lancaster'}
        Which optimizer to use.
    language: {'arabic','danish','dutch','english','finnish','french',
               'german', 'hungarian', 'italian', 'norwegian', 'porter',
               'portuguese", 'romanian', 'russian', 'spanish', swedish'}
        Parameter only valid if `stemmer_name` is set to 'Snowball'.
        It invokes appropriate stemmer for a given language.
    '''
    if stemmer_name == 'Porter':
        stemmer = PorterStemmer()
    elif stemmer_name == 'Snowball':
        stemmer = SnowballStemmer(language=language, ignore_stopwords=False)
    elif stemmer_name =='Lancaster':
        stemmer = LancasterStemmer()
    else:
        print('Stemming was not performed, check the name of the stemmer used!')
        return text
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


def lemmatize_text(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    '''
    Extracts the lemma for each token if a token postag is in 
    'allowed_postags` and add to the `lemmatize_word_list` which will
    be returned as a result. If allowed_postags is set to None than
    lemmatizes all the tokens in the text.
    
    Parameters
    ----------
    text: text on which lemmatizing should be performed.
    allowed_postags: list or None
        If allowed_postags is set to None than all the words in the text
        will be lemmatized, otherwise only words that have appropriate postag
        will be lemmatized and returned.
    '''
    # Initialize spacy 'en' model, keeping only tagger component needed
    #  for lemmatization
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # Parse the sentence using the loaded 'en' model object `nlp`
    doc = nlp(text)

    # Extract the lemma for relevant tokens and return them as a list of
    #  documents.
    if allowed_postags is not None:
        lemmatize_words_list = (
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    else:
        lemmatize_words_list = (
            [token.lemma_ for token in doc])
    return lemmatize_words_list
