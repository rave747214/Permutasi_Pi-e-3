import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

from scipy.stats import hmean
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping

from .preprocessing import load_matrix

class FastText(tf.keras.layers.Embedding):
    def __init__(self, input_dim, output_dim, matrix_path, mask_zero=True, \
        trainable=False, **kwargs):
        
        super(FastText, self).__init__(input_dim, output_dim, **kwargs)
        self.matrix_path = matrix_path
        self.mask_zero = mask_zero
        self.trainable = trainable

    def build(self, input_shape):
        self.matrix = load_matrix(self.matrix_path)
        self.input_dim = self.matrix.shape[0]
        self.output_dim = self.matrix.shape[1]
        self.matrix = tf.Variable(self.matrix, \
                                  trainable=self.trainable, dtype=np.float32)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.matrix, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)

    def get_config(self):
        config = super(FastText, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "mask_zero": self.mask_zero,
            "trainable": self.trainable,
            "matrix_path": self.matrix_path
            })
        return config

class ReturnBestEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
        elif self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)

def get_model(n_hidden, matrix_path, summary=True, \
              init_weights_path='./model/init_weights.hdf5'):
    
    # model architecure
    inputs = layers.Input(shape=(None,), name='input', dtype='int32')
    layer = FastText(0, 0, matrix_path, mask_zero=True, name='emb')(inputs)
    lstm = layers.LSTM(n_hidden, name='lstm')(layer)
    dense = layers.Dense(1, activation='sigmoid', name='dense')(lstm)
    model = Model(inputs=inputs, outputs=dense, name='model')

    if os.path.exists(init_weights_path):
        print('Initial weights found. Loading...')
        model.load_weights(init_weights_path)
    else:
        print('Initial weights not found. Saving...')
        model.save_weights(init_weights_path)

    # training setup
    initial_learning_rate = 0.1
    decay_steps = 50
    decay_rate = 1.0
    learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
      initial_learning_rate, decay_steps, decay_rate)
    optimizer = optimizers.RMSprop(learning_rate=learning_rate_fn)

    prec = Precision(name='prec')
    rec = Recall(name='rec')
    metrics = ['acc', prec, rec]
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    if summary:
        model.summary()

    return model

def train(model, X, y, history_path, model_path, epochs, **kwargs):
    rbes = ReturnBestEarlyStopping(monitor='val_loss', min_delta=0.0001, \
                                   patience=epochs+1, restore_best_weights=True)

    hist = model.fit(X, y, epochs=epochs, callbacks=[rbes], **kwargs)
    hist = pd.DataFrame(hist.history)

    prec, rec = hist['prec'], hist['rec']
    val_prec, val_rec = hist['val_prec'], hist['val_rec']
    f1, val_f1 = hmean([prec, rec]), hmean([val_prec, val_rec])

    hist = hist[['loss', 'val_loss', 'acc', 'val_acc']]
    hist['f1'] = f1
    hist['val_f1'] = val_f1
    hist.to_csv(history_path, index=False)

    model.save(model_path)
    return (model, hist)
