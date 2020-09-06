import numpy as np
from pandas import DataFrame
from scipy.stats import hmean
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

from module.preprocessing import load_dataset, load_matrix, Preprocessor
from module.modeling import get_model, train, plot_charts

def get_patience(epochs):
    if epochs < 30:
        return epochs
    elif 30 <= epochs and epochs <= 300:
        return epochs // 10
    else:
        return 30

def get_metrics(model, X, y):
    loss, acc, prec, rec = model.evaluate(X, y)
    f1 = hmean([prec, rec])
    return (loss, acc, f1)
    
def run(data='A0', batch_size=16, n_hidden=16):
    prep = load_preprocessor(data)
    X_train, y_train, X_val, y_val, X_test, y_test = \
        load_dataset(data, clean=True)
    X_train = prep.to_sequence(X_train)
    X_val = prep.to_sequence(X_val)
    X_test = prep.to_sequence(X_test)

    if batch_size > 1:
        X_train = prep.pad(X_train)
        X_val = prep.pad(X_val)
        X_test = prep.pad(X_test)
        
    # build the model
    matrix_path = 'matrix/matrix_%s.npz' % data
    init_weights_path = './model/init_weights_%02d.hdf5' % n_hidden
    model = get_model(n_hidden=n_hidden, matrix_path=matrix_path, \
                      init_weights_path=init_weights_path)

    # train the model
    history_path = 'csv/hist.%s.%03d.%02d.csv'
    history_path = history_path % (data, batch_size, n_hidden)
    model_path = 'model/model.%s.%03d.%02d.hdf5'
    model_path = model_path % (data, batch_size, n_hidden)
    
    steps_per_epoch = np.ceil(X_train.shape[0] / batch_size)
    epochs = np.ceil(1000 / steps_per_epoch)
    patience = get_patience(epochs)
    
    es = EarlyStopping(monitor='val_loss', min_delta=0.0001, \
                       patience=patience, restore_best_weights=True)
    model, hist = train(model, X_train, y_train, \
                        history_path, model_path, \
                        epochs=epochs, batch_size=batch_size, \
                        validation_data=(X_val, y_val), \
                        callbacks=[es], shuffle=False)
    
    # evaluation
    metrics_train = get_metrics(model, X_train, y_train)
    metrics_val = get_metrics(model, X_val, y_val)
    metrics_test = get_metrics(model, X_test, y_test)
    metrics = np.array([metrics_train, metrics_val, metrics_test]).ravel()
    
    metrics = DataFrame(data=metrics, \
                       index=['train_loss', 'train_acc', 'train_f1', \
                              'val_loss', 'val_acc', 'val_f1', \
                              'test_loss', 'test_acc', 'test_f1'])
    metrics_path = 'csv/metrics.%s.%03d.%02d.csv'
    metrics_path = metrics_path % (data, batch_size, n_hidden)
    metrics.to_csv(metrics_path, index=False)
