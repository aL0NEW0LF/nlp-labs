from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from gensim.models import Word2Vec
import pandas as pd

def tagsDict(taggedSentences: List[str], method: str, removeNonEntities: bool = False):
    """ 
    Extracts tags from tagged sentences.
    Args:
        - taggedSentences (List[str]): List of tagged sentences.
        - method (str): Method used for tagging. can be 'pos' or 'ner'.
    """
    validMethods = ['pos', 'ner']
    if method not in validMethods:
        raise ValueError(f"Invalid method. method must be one of {validMethods}")
    if not isinstance(taggedSentences, list) or not all(isinstance(s, str) for s in taggedSentences):
        raise ValueError("taggedSentences must be a list of strings.")
    if not isinstance(removeNonEntities, bool):
        raise ValueError("removeNonEntities must be a boolean.")
    if method == 'pos' and removeNonEntities:
        raise ValueError("removeNonEntities can only be used with method='ner'.")

    taggedTokens = []

    for p in taggedSentences:
        taggedTokens.append(word_tokenize(p))

    if method == 'pos':
        for p in taggedTokens:
            IndexToRemove = []
            for i in range(len(p)):
                if "/" not in p[i]:
                    p[i+1] = p[i] + p[i+1]
                    IndexToRemove.append(i)
            for i in sorted(IndexToRemove, reverse=True):
                del p[i]

    TagsDictsList = []

    for p in taggedTokens:
        posDict = {}
        for w in p:
            w = w.split('/')
            posDict[w[0]] = w[1]
        TagsDictsList.append(posDict)

    if removeNonEntities:
        for p in TagsDictsList:
            for w in list(p.keys()):
                if p[w] == 'O':
                    p.pop(w)

    return TagsDictsList

def tokenize(text: str):
    """ Tokenizes a text into words.
    Args:
        - text (str): Text to tokenize.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string.")

    # Remove punctuation (except math symbols)
    text = re.sub(r'[^\w\s\+\-\*\/\(\)]', '', text)
    
    text = text.lower().strip()

    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)

    words = [w for w in words if not w in stop_words]
    
    return words

def lemma(words: list):
    """ Lemmatizes a list of words.
    Args:
        - words (list): List of words to lemmatize.
    """
    if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
        raise ValueError("words must be a list of strings.")
    
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in words]

def vectorize_answer(answer_tokens, word2vec_model):
    answer_vector = pd.Series([np.mean([word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv] or [np.zeros(word2vec_model.vector_size)], axis=0) for sentence in answer_tokens])
    return answer_vector

# function to see MSE and R2 variation with the variation of vector size
def max_mse_r2_variation(dataframe, model, sg=0, test_size=0.1):
    best_mse = 1000000000
    best_r2 = -1000000000
    best_vector_size = 0


    for i in range(1, 2000):
        cbow_model = Word2Vec(dataframe['answer'], vector_size=i, window=5, sg=sg, min_count=1)
        answer_vector = vectorize_answer(dataframe['answer'], cbow_model)

        X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(answer_vector.values.tolist(), index=answer_vector.index), dataframe['score'], test_size=test_size, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if mse < best_mse and r2 > best_r2:
            best_mse = mse
            best_r2 = r2
            best_vector_size = i
        
        if i % 100 == 0:
            print('Vector size: ', i, ' - MSE: ', mse, ' - R2: ', r2, ' - Best MSE: ', best_mse, ' - Best R2: ', best_r2, ' - Best Vector Size: ', best_vector_size)

    print('Best MSE: ', best_mse, ' - Best R2: ', best_r2, ' - Best Vector Size: ', best_vector_size)

    return best_mse, best_r2, best_vector_size

def evaluate_model(dataframe, model, vector_size, sg=0, test_size=0.1):
    cbow_model = Word2Vec(dataframe['answer'], vector_size=vector_size, window=5, sg=sg, min_count=1)
    answer_vector = vectorize_answer(dataframe['answer'], cbow_model)

    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(answer_vector.values.tolist(), index=answer_vector.index), dataframe['score'], test_size=test_size, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('MSE: ', mse, ' - R2: ', r2)

    return mse, r2