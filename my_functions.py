import pandas as pd
import re
from copy import deepcopy
from collections import defaultdict
from sklearn.feature_extraction import text
from nltk import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import pyLDAvis
import pyLDAvis.gensim
from gensim.corpora.dictionary import Dictionary

class LemmaStemTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ss = SnowballStemmer('english')
    
    def __call__(self, text):
        words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
        lemma_word = [self.wnl.lemmatize(t, pos = 'v') for t in word_tokenize(str(words))]
        words = [self.ss.stem(word) for word in lemma_word]
        return words
    
def replace(string, substitutions):
    for i, substitution in enumerate(substitutions):
        string = deepcopy(([s.replace(list(substitutions.keys())[i], list(substitutions.values())[i]) for s in string]))
    return string

def tokenized_df_lookup(token_list):
    df = pd.DataFrame(token_list)
    df[1] = [item[1] for item in df[0].str.split(', ')]
    df[0] = [item[0] for item in df[0].str.split(', ')]
    df.rename({0: 'stem', 1: 'original_word'}, axis = 1, inplace = True)
    df['original_word_copy'] = deepcopy(df['original_word'])
    aggregations = {
        'original_word_copy': 'count'
        }
    df = deepcopy(df.groupby(['stem', 'original_word']).agg(aggregations).reset_index())
    duplicate_list = ['stem']
    df.drop_duplicates(subset = duplicate_list, keep = 'first', inplace = True)
    df = df.reset_index()
    df.drop('index', axis = 1, inplace = True)
    return df

def id2word_rename(id2word_input, token_df):
    for b, word in enumerate(id2word_input.values()):
        for i, stem_index in enumerate(token_df['stem']):
            if word == stem_index:
                id2word_input[list(id2word_input.keys())[b]] = token_df['original_word'][i]
    return id2word_input

def dtm_unigram(vectorizer, token_df, df_input, stop_word_input, min_df):
    stop_words = text.ENGLISH_STOP_WORDS.union(stop_word_input)
    vec = vectorizer(tokenizer = LemmaStemTokenizer(),
                      strip_accents = 'unicode',
                      stop_words = stop_words,
                      lowercase = True,
                      min_df = min_df)
    X = vec.fit_transform(df_input)
    dtm_df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
    for column in dtm_df.columns:
        for i, stem_index in enumerate(token_df['stem']):
            if column == stem_index:
                dtm_df.rename({column: token_df['original_word'][i]}, axis = 1, inplace = True)
    return X, dtm_df, vec

def dtm_bigram(vectorizer, token_df, df_input, stop_word_input, min_df):
    stop_words = text.ENGLISH_STOP_WORDS.union(stop_word_input)
    vec = vectorizer(tokenizer = LemmaStemTokenizer(),
                     ngram_range = (2, 2),
                      strip_accents = 'unicode',
                      stop_words = stop_words,
                      lowercase = True,
                      min_df = min_df)
    X = vec.fit_transform(df_input)
    dtm_df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
    column_df = deepcopy(pd.DataFrame(dtm_df.columns))
    column_df[1] = [item[1] for item in column_df[0].str.split(' ')]
    column_df[0] = [item[0] for item in column_df[0].str.split(' ')]
    for i, column in enumerate(column_df[0]):
        for z, stem_index in enumerate(token_df['stem']):
            if column == stem_index:
                column_df[0].loc[i] = token_df['original_word'][z]
    for i, column in enumerate(column_df[1]):
        for z, stem_index in enumerate(token_df['stem']):
            if column == stem_index:
                column_df[1].loc[i] = token_df['original_word'][z]
    for i, column in enumerate(dtm_df.columns):
        dtm_df.rename({column: column_df[0][i] + " " + column_df[1][i]}, axis = 1, inplace = True)
    return X, dtm_df, vec

def topics_df_func(lda_doc_input):
    topics_list = []
    for document in lda_doc_input:
        document_dict = defaultdict(list)
        for topic in document:
            document_dict[topic[0]] = topic[1]
        topics_list.append(document_dict)
    topics_df = pd.DataFrame(topics_list)
    topics_df.fillna(0, inplace = True)
    return topics_df

def pyLDAvis_plot(lda_input, id2word_input, corpus_input):
    dic_input = dict((v, k) for k, v in id2word_input.items())
    dic = Dictionary([dic_input])
    data = pyLDAvis.gensim.prepare(lda_input, corpus_input, dic)
    return data

def display_topics(model, feature_names, no_top_words, topic_names = None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '", topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1: -1]]))