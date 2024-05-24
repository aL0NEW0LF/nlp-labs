from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
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
    """ Vectorizes a list of tokens using a Word2Vec model.
    Args:
        - answer_tokens (list): List of tokens to vectorize.
        - word2vec_model (Word2Vec): Word2Vec model to use for vectorization.
    """
    answer_vector = pd.Series([np.mean([word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv] or [np.zeros(word2vec_model.vector_size)], axis=0) for sentence in answer_tokens])
    return answer_vector

def vectorize_dataframe(dataframe, sg=0, vector_size=100):
    """
    Vectorizes a dataframe using Word2Vec.
    Args:
        - dataframe (pd.DataFrame): Dataframe to vectorize.
        - sg (int): Skip-gram parameter for Word2Vec.
        - vector_size (int): Vector size for Word2Vec.
    """
    cbow_model = Word2Vec(dataframe['content'], vector_size=vector_size, window=5, sg=sg, min_count=1)
    text_vectors = vectorize_answer(dataframe['content'], cbow_model)
    return pd.concat([pd.DataFrame(text_vectors.values.tolist(), index=text_vectors.index), dataframe['score']], axis=1)


def finetune_mse_r2(dataframe, model, sg=0, test_size=0.1, min_vector_size=1, max_vector_size=1000):
    """ 
    Finetunes a regression model using Word2Vec embeddings and returns the best MSE, R2 and vector size.
    Args:
        - dataframe (pd.DataFrame): Dataframe containing the data.
        - model: Regression model to finetune.
        - sg (int): Skip-gram parameter for Word2Vec.
        - test_size (float): Test size for train-test split.
        - min_vector_size (int): Minimum vector size to test.
        - max_vector_size (int): Maximum vector size to test.
    """
    best_mse = 1000000000
    best_r2 = -1000000000
    best_vector_size = 0


    for i in range(min_vector_size, max_vector_size):
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

def evaluate_reg_model(dataframe, model, vector_size, sg=0, test_size=0.1):
    """
    Evaluates a regression model using Word2Vec embeddings and returns the MSE and R2.
    Args:
        - dataframe (pd.DataFrame): Dataframe containing the data.
        - model: Regression model to evaluate.
        - vector_size (int): Vector size for Word2Vec.
        - sg (int): Skip-gram parameter for Word2Vec.
        - test_size (float): Test size for train-test split.
    """
    cbow_model = Word2Vec(dataframe['answer'], vector_size=vector_size, window=5, sg=sg, min_count=1)
    answer_vector = vectorize_answer(dataframe['answer'], cbow_model)

    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(answer_vector.values.tolist(), index=answer_vector.index), dataframe['score'], test_size=test_size, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('MSE: ', mse, ' - R2: ', r2)

    return mse, r2

def finetune_acc_f1(dataframe_train, dataframe_val, model, sg=0, min_vector_size=1, max_vector_size=1000):
    """
    Finetunes a classification model using Word2Vec embeddings and returns the best accuracy, F1 and vector size.
    Args:
        - dataframe_train (pd.DataFrame): Dataframe containing the training data.
        - dataframe_val (pd.DataFrame): Dataframe containing the validation data.
        - model: Classification model to finetune.
        - sg (int): Skip-gram parameter for Word2Vec.
        - min_vector_size (int): Minimum vector size to test.
        - max_vector_size (int): Maximum vector size to test.
    """
    best_acc = -1000000000
    best_f1 = -1000000000
    best_vector_size = 0


    for i in range(min_vector_size, max_vector_size):
        cbow_model = Word2Vec(dataframe_train['text'], vector_size=i, window=5, sg=sg, min_count=1)
        text_vectors = vectorize_answer(dataframe_train['text'], cbow_model)
        valid_text_vectors = vectorize_answer(dataframe_val['text'], cbow_model)

        X_train = pd.DataFrame(text_vectors.values.tolist(), index=text_vectors.index)
        y_train = dataframe_train['label']
        X_test = pd.DataFrame(valid_text_vectors.values.tolist(), index=dataframe_val.index)
        y_test = dataframe_val['label']

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        if acc > best_acc and f1 > best_f1:
            best_acc = acc
            best_f1 = f1
            best_vector_size = i
        
        if i % 100 == 0:
            print('Vector size: ', i, ' - Accuracy: ', acc, ' - F1: ', f1, ' - Best Accuracy: ', best_acc, ' - Best F1: ', best_f1, ' - Best Vector Size: ', best_vector_size)

    print('Best Accuracy: ', best_acc, ' - Best F1: ', best_f1, ' - Best Vector Size: ', best_vector_size)

    return best_acc, best_f1, best_vector_size

def evaluate_clf_model(dataframe_train, dataframe_val, model, vector_size, sg=0):
    """
    Evaluates a classification model using Word2Vec embeddings and returns the accuracy and F1.
    Args:
        - dataframe_train (pd.DataFrame): Dataframe containing the training data.
        - dataframe_val (pd.DataFrame): Dataframe containing the validation data.
        - model: Classification model to evaluate.
        - vector_size (int): Vector size for Word2Vec.
        - sg (int): Skip-gram parameter for Word2Vec.
    """
    cbow_model = Word2Vec(dataframe_train['text'], vector_size=vector_size, window=5, sg=sg, min_count=1)
    text_vectors = vectorize_answer(dataframe_train['text'], cbow_model)
    valid_text_vectors = vectorize_answer(dataframe_val['text'], cbow_model)

    X_train = pd.DataFrame(text_vectors.values.tolist(), index=text_vectors.index)
    y_train = dataframe_train['label']
    X_test = pd.DataFrame(valid_text_vectors.values.tolist(), index=dataframe_val.index)
    y_test = dataframe_val['label']

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print('Accuracy: ', acc, ' - F1: ', f1)

    return acc, f1

def encode_sentence(text, vocab2index, N=200):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length

# We can load the vectors using our custom functions
def load_glove_vectors(glove_file = '../utils/models/vectors.txt'):
    """Load the glove word vectors"""
    word_vectors = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    return word_vectors

def get_emb_matrix(pretrained, word_counts, emb_size = 256):
    """ Creates embedding matrix from word vectors"""
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in pretrained:
            W[i] = pretrained[word]
        else:
            W[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1   
    return W, np.array(vocab), vocab_to_idx

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()