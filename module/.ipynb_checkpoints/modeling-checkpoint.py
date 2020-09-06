import os
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

from scipy.stats import hmean
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model, load_model

from .preprocessing import load_matrix

class FastText(tf.keras.layers.Layer):
    def __init__(self, matrix_path, mask_zero=False, trainable=False, \
                 **kwargs):
        
        super(FastText, self).__init__(**kwargs)
        self.matrix_path = matrix_path
        self.input_dim = self.matrix.shape[0]
        self.output_dim = self.matrix.shape[1]
        self.mask_zero = mask_zero
        self.trainable = trainable

    def build(self, input_shape):
        self.weights = load_matrix(matrix_path)
        self.weights = tf.Variable(self.weights, \
                                   trainable=self.trainable)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.weights, inputs)

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
    
def get_model(n_hidden, matrix_path, summary=True, \
              init_weights_path='./model/init_weights.hdf5'):
    
    # model architecure
    inputs = layers.Input(shape=(None,), name='input', dtype='int32')
    layer = FastText(matrix_path, mask_zero=True, name='emb')(inputs)
    lstm = layers.LSTM(n_hidden, name='lstm')(layer)
    dense = layers.Dense(1, activation='sigmoid', name='dense')(lstm)
    model = Model(inputs=inputs, outputs=dense, name='model')

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

    if os.path.exists(init_weights_path):
        model.load_weights(init_weights_path)
    else:
        model.save_weights(init_weights_path)
        
    return model

def train(model, X, y, history_path, model_path, **kwargs):
    hist = model.fit(X, y, **kwargs)
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

def test(model_path, matrix_path, X, y):
    model = load_model(model_path, custom_objects={'FastText': FastText})
    model.layers[1].weights = load_matrix(matrix_path)
    
    loss, acc, prec, rec = model.evaluate(X, y)
    f1 = hmean([prec, rec])
    return (loss, acc, f1)

def plot_charts(history_paths, metrics, epochs, save_path, \
                fig_height=3, scale=1.5, c=('blue', 'orange')):
    
    n_hist = len(history_paths)
    figsize = (fig_height * n_hist * scale, fig_height)
    
    val_met = 'val_' + metrics
    history_paths = [pd.read_csv(hist)[[metrics, val_met]] \
        for hist in history_paths]
    
    fig, ax = plt.subplots(1, n_hist)
    fig.set_size_inches(figsize)
    
    for _ in range(n_hist):
        history = history_paths[_]
        ax[_].plot(history[metrics], c=c[0], linewidth=2)
        ax[_].plot(history[val_met], c=c[1], linewidth=2)
        ax[_].set_xticks(range(0, epochs, epochs // 10))
        ax[_].legend([metrics, val_met], loc='best')
        ax[_].set_xlabel('epoch')
        ax[_].set_ylabel(metrics)
        ax[_].grid()
    
    fig = plt.gcf()
    plt.suptitle('Grafik ' + metrics.title())
    plt.savefig(save_path)