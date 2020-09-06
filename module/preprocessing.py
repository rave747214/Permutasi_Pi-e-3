import json
import pickle
import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(name='A', clean=False):
    clean = 'clean_' if clean else ''
    path = 'dataset/{}data_%s_{}.csv'.format(clean, name)
    
    data = pd.read_csv(path % 'train').dropna()
    X_train, y_train = data['RESPONSE'].values, data['LABEL'].values
    
    data = pd.read_csv(path % 'dev').dropna()
    X_val, y_val = data['RESPONSE'].values, data['LABEL'].values
    
    data = pd.read_csv(path % 'test').dropna()
    X_test, y_test = data['RESPONSE'].values, data['LABEL'].values

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return (X_train, y_train, X_val, y_val, X_test, y_test)

def load_matrix(path):
    matrix = np.load(path, allow_pickle=True)
    return matrix['matrix']

def load_preprocessor(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class Preprocessor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.steps = {
            'stem': self.stem,
            'remove_stop_word': self.remove_stop_word,
            'to_sequence': self.to_sequence,
            'pad': self.pad
        }
        
    def stem(self, X):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        for i in range(X.shape[0]):
            X[i] = stemmer.stem(X[i])
        return X
    
    def remove_stop_word(self, X):
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        for i in range(X.shape[0]):
            X[i] = stopword.remove(X[i])
            if len(X[i]) == 0:
                X[i] = ''
        return X
    
    def to_sequence(self, X):
        seq = self.tokenizer.texts_to_sequences(X)
        return seq
    
    def pad(self, X):
        return pad_sequences(X, padding='post')
    
    def fit_on_texts(self, X):
        if isinstance(X, np.ndarray):
            self.tokenizer.fit_on_texts(X)
        elif isinstance(X, list):
            self.tokenizer.fit_on_texts(np.hstack(X))
        self.word_index = self.tokenizer.word_index
    
    def set_pipeline(self, steps):
        self.pipeline = steps
    
    def run(self, data):
        for step in self.pipeline:
            data = self.steps[step](data)
        return data

    def word_index_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.word_index, f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
