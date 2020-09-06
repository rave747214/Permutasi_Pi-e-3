import numpy as np
import tensorflow as tf

from time import time
from pandas import DataFrame
from scipy.stats import hmean
from tensorflow.keras.preprocessing.text import Tokenizer

from module.modeling import get_model, train, plot_charts
from module.preprocessing import load_dataset, load_matrix, load_preprocessor

def get_metrics(model, X, y):
    loss, acc, prec, rec = model.evaluate(X, y, verbose=1)
    f1 = hmean([prec, rec])
    return (loss, acc, f1)

def run(data='A0', batch_size=16, n_hidden=16):
    path_prep = 'misc/prep_%s.pkl' % data
    prep = load_preprocessor(path_prep)
    X_train, y_train, X_val, y_val, X_test, y_test = \
        load_dataset(data, clean=True)

    X_train = prep.to_sequence(X_train)
    X_val = prep.to_sequence(X_val)
    X_test = prep.to_sequence(X_test)
    X_train = prep.pad(X_train)
    X_val = prep.pad(X_val)
    X_test = prep.pad(X_test)

    # build the model
    matrix_path = 'matrix/matrix_%s.npz' % data
    init_weights_path = './model/init_weights_%s_%02d.hdf5' % (data, n_hidden)
    model = get_model(n_hidden=n_hidden, matrix_path=matrix_path, \
                      init_weights_path=init_weights_path, summary=False)

    # train the model
    history_path = 'csv/hist.%s.%03d.%02d.csv'
    history_path = history_path % (data, batch_size, n_hidden)
    model_path = 'model/model.%s.%03d.%02d.hdf5'
    model_path = model_path % (data, batch_size, n_hidden)

    steps_per_epoch = np.ceil(len(X_train) / batch_size)
    epochs = int(np.ceil(1500 / steps_per_epoch))

    model, hist = train(model, X_train, y_train, history_path, model_path, \
                        epochs, batch_size=batch_size, shuffle=False, verbose=0, \
                        validation_data=(X_val, y_val))

    # evaluation
    metrics_train = get_metrics(model, X_train, y_train)
    metrics_val = get_metrics(model, X_val, y_val)
    metrics_test = get_metrics(model, X_test, y_test)
    metrics = np.array([metrics_train, metrics_val, metrics_test]).ravel()

    metrics = DataFrame(data=metrics, \
                       index=['train_loss', 'train_acc', 'train_f1', \
                              'val_loss', 'val_acc', 'val_f1', \
                              'test_loss', 'test_acc', 'test_f1']).T
    metrics = metrics[['train_loss', 'val_loss', 'test_loss', \
                       'train_acc', 'val_acc', 'test_acc', \
                       'train_f1', 'val_f1', 'test_f1']]

    metrics_path = 'csv/metrics.%s.%03d.%02d.csv'
    metrics_path = metrics_path % (data, batch_size, n_hidden)
    metrics.to_csv(metrics_path, index=False)

if __name__ == '__main__':
    print()
    if tf.test.gpu_device_name():
        print('Default GPU Device:', tf.test.gpu_device_name())
    else:
        print('No GPU Found!')
    print()

    files = ['A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 'B3', 'B4']
    batch_sizes = [4, 16, 64, 512]
    n_hiddens = [16, 32, 64]

    total = len(files) * len(batch_sizes) * len(n_hiddens)
    counter = 1
    start_prog = time()

    for n_hidden in n_hiddens:
        for batch_size in batch_sizes:
            for data in files:
                report = '[%02d/%02d] Running %s %03d %2d...'
                print(report % (counter, total, data, batch_size, n_hidden))

                start_run = time()
                run(data, batch_size, n_hidden)

                time_run, time_prog = time() - start_run, time() - start_prog
                mean_time = time_prog / counter
                est_time = (total - counter) * mean_time / 60
                report = '[%.2fs] to run, [%.2fm] to finish\n'
                print(report % (time_run, est_time))
                counter += 1
