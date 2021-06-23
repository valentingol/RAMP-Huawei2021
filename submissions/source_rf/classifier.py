from lightgbm import LGBMClassifier
import tensorflow as tf
import tensorflow.keras as keras

from train_utils.loss import UnbalancedMSE_gb
from train_utils.batch import MakeBatch
from train_utils.model import load_model
from predict_utils.neuralnet import nn_prediction

kl = tf.keras.layers

class Classifier:
    def __init__(self):
        units = 64
        dropout = 0.0
        recurrent_dropout = 0.0
        path_nn = None
        neural_net = tf.keras.Sequential([
            kl.LSTM(units, dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=False, stateful=True),
            kl.Dense(units, activation='relu'),
            kl.Dense(units, activation='linear'),
            kl.Dense(1, activation='linear')])
        neural_net = load_model(neural_net, path_nn)
        self.nn = keras.Sequential(neural_net.layers[:-1])
        # architecture and training parameters
        self.num_leaves = 31
        self.max_depth = -1
        self.lr = 0.1
        self.n_estimators = 100
        self.n_epoch = 1
        # gradient boosting
        self.loss = UnbalancedMSE_gb(data='source')
        self.gb = LGBMClassifier(num_leaves=self.num_leaves,
                                 max_depth=self.max_depth,
                                 learning_rate=self.lr,
                                 n_estimators=self.n_estimators,
                                 objective = self.loss)

    def fit(self, X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target):
        batcher = MakeBatch(batch_size=64, data='source', val_prop=0.2)
        batches = batcher(X_source, y_source)
        features = self.nn(X_source)
        X = features.numpy()
        y = y_source.numpy()
        self.gb.fit(X, y)

    def predict_proba(self, X_target, X_target_bkg):
        features = nn_prediction(self.nn, X_target)
        X = features.numpy()
        y_proba = self.gb.predict_proba(X)
        return y_proba