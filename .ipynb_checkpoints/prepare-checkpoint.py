import fasttext
import numpy as np

from tqdm import trange, tqdm
from pandas import DataFrame
from module.preprocessing import Preprocessor, load_dataset, load_preprocessor

model = None

def preprocess(data='A'):
    preps = [Preprocessor() for _ in range(5)]
    preps[0].set_pipeline([])
    preps[1].set_pipeline(['remove_stop_word'])
    preps[2].set_pipeline(['stem'])
    preps[3].set_pipeline(['remove_stop_word', 'stem'])
    preps[4].set_pipeline(['stem', 'remove_stop_word'])

    for i in trange(5):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(data)
        X_train_clean = preps[i].run(X_train)
        X_val_clean = preps[i].run(X_val)
        X_test_clean = preps[i].run(X_test)

        preps[i].fit_on_texts([X_train, X_val, X_test])
        preps[i].save('misc/prep_%s%d.pkl' % (data, i))

        df = DataFrame({'RESPONSE':X_train_clean, 'LABEL':y_train.ravel()})
        df.to_csv('dataset/clean_data_train_%s%d.csv' % (data, i), index=False)
        df = DataFrame({'RESPONSE':X_val_clean, 'LABEL':y_val.ravel()})
        df.to_csv('dataset/clean_data_dev_%s%d.csv' % (data, i), index=False)
        df = DataFrame({'RESPONSE':X_test_clean, 'LABEL':y_test.ravel()})
        df.to_csv('dataset/clean_data_test_%s%d.csv' % (data, i), index=False)

def generate_matrix(data):
    prep = load_preprocessor('misc/prep_%s.pkl' % data)
    word_index = prep.word_index

    matrix = np.zeros((len(word_index) + 1, 300))
    for word, index in word_index.items():
        matrix[index] = model[word]

    path = 'matrix/matrix_%s.npz' % (data)
    np.savez_compressed(path, matrix=matrix)


if __name__ == '__main__':
    # generate preprocessor
    print('Start processing data A...')
    preprocess('A')
    print('Start processing data B...')
    preprocess('B')
    
    # generate embedding matrix from FastText model
    model = fasttext.load_model('../fasttext_model/cc.id.300.bin')

    data_list = ['A0', 'A1', 'A2', 'A3', 'A4', \
        'B0', 'B1', 'B2', 'B3', 'B4']

    for data in tqdm(data_list):
        generate_matrix(data)
