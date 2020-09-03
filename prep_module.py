import json
import numpy as np
import pandas as pd
import Sastrawi

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(name='A'):
    data = pd.read_csv('dataset/%s/data_train_A.csv' % name)
    X_train, y_train = data['RESPONSE'].values, data['LABEL'].values
    
    data = pd.read_csv('dataset/%s/data_dev_A.csv' % name)
    X_val, y_val = data['RESPONSE'].values, data['LABEL'].values
    
    data = pd.read_csv('dataset/%s/data_test_A.csv' % name)
    X_test, y_test = data['RESPONSE'].values, data['LABEL'].values

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return (X_train, y_train, X_val, y_val, X_test, y_test)

def load_matrix(dataset='train'):
    path = 'embedding_matrix/emb_%s.npz' % dataset
    matrix = np.load(path, allow_pickle=True)
    return matrix['matrix']

class Preprocessor:
    def __init__(self):
        pass
    
    def tokenize(self, X):
        tokenizer = Tokenizer()
        if isinstance(X, np.ndarray):
            tokenizer.fit_on_texts(X)
        elif isinstance(X, list):
            tokenizer.fit_on_texts(np.hstack(X))
        return tokenizer
    
    def to_sequence(self, X, tokenizer):
        seq = tokenizer.texts_to_sequences(X)
        return seq
    
    def pad(self, X):
        return pad_sequences(X, padding='post')
    
    def run(self, data, tokenizer):
        data = self.to_sequence(data, tokenizer)
        data = self.pad(data)
        return data

    def word_index_to_json(self, tokenizer, path):
        with open(path, 'w') as f:
            json.dump(tokenizer.word_index, f)
