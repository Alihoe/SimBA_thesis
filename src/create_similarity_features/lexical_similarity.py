from nltk import word_tokenize
from nltk.corpus import stopwords

import nltk
#nltk.download('stopwords')

characters = ["",'']


def get_lexical_entities(sequence_dictionary, lex_feature):
    entity_dict = {}
    if lex_feature == "similar_words_ratio" or lex_feature == "similar_words_ratio_length":
        for id, text in sequence_dictionary.items():
            entity_dict[id] = tokenize_and_filter_out_stop_words(text)
    return entity_dict


def tokenize_and_filter_out_stop_words(sequence):
    stop_words = set(stopwords.words('english'))
    try:
        return [w for w in tokenize(sequence) if not w.lower() in stop_words and not w in characters and len(w) > 1]
    except:
        print(sequence)
        return []


def tokenize(sequence):
    return word_tokenize(sequence)


def get_lexical_similarity(query, target):
    query_entities = set(tokenize_and_filter_out_stop_words(query))
    target_entities = set(tokenize_and_filter_out_stop_words(target))
    len_query_entities = len(query_entities)
    len_target_entities = len(target_entities)
    len_intersection = len(query_entities.intersection(target_entities))
    len_union = len_query_entities + len_target_entities
    return (100 / (len_union / 2)) * len_intersection





