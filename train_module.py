import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

def get_model(n_hidden, emb_matrix, save_init_weights=False, init_weights_path='./model/init_weights.hdf5'):
    # model arch
    inputs = tf.keras.layers.Input(shape=(None,), name='input', dtype='int64')
    layer = FastText(emb_matrix.shape[0], emb_matrix.shape[1], emb_matrix, mask_zero=True, name='emb')(inputs)
    lstm = tf.keras.layers.LSTM(n_hidden, name='lstm')(layer)
    dense = tf.keras.layers.Dense(1, activation='sigmoid', name='dense')(lstm)
    model = tf.keras.models.Model(inputs=inputs, outputs=dense, name='model')

    # training setup
    step = tf.Variable(0, trainable=False)
    boundaries = [50, 100]
    values = [0.1, 0.05, 0.02]
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    prec = Precision(name='prec')
    rec = Recall(name='rec')
    metrics = ['acc', prec, rec]
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    if save_init_weights:
        model.save_weights(init_weights_path)
    else:
        model.load_weights(init_weights_path)
        
    return model

class FastText(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, matrix, mask_zero=False, **kwargs):
        super(FastText, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        self.matrix = matrix

    def build(self, input_shape):
        self.matrix = tf.Variable(self.matrix, trainable=False)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.matrix, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)