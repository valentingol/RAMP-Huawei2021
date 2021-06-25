import os

import lightgbm
import numpy as np
import tensorflow as tf

from train_utils.batch import MakeBatch
from train_utils.loss import UnbalancedMSE_nn, UnbalancedMSE_gb
from external_imports.utils.score_types import PrecisionAtRecall
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

class Classifier:
    def __init__(self):
        kl = tf.keras.layers
        # training & architecture parameters
        self.n_epoch_nn = 1
        self.batch_size = 64
        self.lr_nn = 1e-3
        self.lr_gb = 0.1
        self.timestamp = 96
        self.n_features = 64

        # neural network
        self.loss_nn = UnbalancedMSE_nn(data='source')
        self.nn = tf.keras.Sequential([
                kl.LSTM(self.n_features, dropout=0.0, recurrent_dropout=0.0,
                        return_sequences=False, stateful=True,
                        input_shape=(self.timestamp, 10),
                        activation='tanh'),
                kl.Dense(self.n_features, activation='linear'),
                kl.Dense(1, activation='sigmoid')
                ])
        self.nn_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_nn)
        self.nn.compile(self.nn_opt)

        # gradient boosting
        self.loss_gb = UnbalancedMSE_gb(data='source')
        self.gb = lightgbm.LGBMClassifier(num_leaves=31,
                                          max_depth=3,
                                          learning_rate=self.lr_gb,
                                          n_estimators=100,
                                          objective = self.loss_gb)
        # nn logs
        self.logs = {'train_err':[], 'train_acc':[], 'train_auc':[],
                     'train_ap':[], 'train_rec5':[], 'train_rec10':[], 'train_rec20':[],
                    'val_err':[], 'val_acc':[],'val_auc':[],
                    'val_ap':[], 'val_rec5':[], 'val_rec10':[], 'val_rec20':[]}

    def fit(self, X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target):
        batcher = MakeBatch(batch_size=self.batch_size, data='source', val_prop=0.2)
        batches = batcher(X_source, y_source)
        # train neural network
        self.train_nn(batches)
        # train gradient boosting
        self.train_gb(batches)

    def train_nn(self, batches, verbose=True):
        train_batches, val_batches = batches
        for epoch in self.n_epoch_nn:
            if verbose: print(f'epoch {epoch+1}/{self.n_epoch_nn}')
            # train loop
            for i, X, y in enumerate(train_batches):
                with tf.GradientTape as tape:
                    y_pred = self.nn(X)
                    err_tensor = self.loss_nn(y, y_pred)
                grads = tape.gradient(err_tensor, self.nn.trainable_weights)
                self.nn_opt.apply_gradients(zip(grads, self.nn.trainable_weights))
                err = float(err_tensor)

                acc = accuracy_score(y, y_pred)
                auc = roc_auc_score(y, y_pred)
                ap = average_precision_score(y, y_pred)
                rec5 = PrecisionAtRecall(recall = 0.05)(y, y_pred)
                rec10 = PrecisionAtRecall(recall = 0.1)(y, y_pred)
                rec20 = PrecisionAtRecall(recall = 0.2)(y, y_pred)

                if verbose:
                    print(f" batch {i+1}, err {err: .3f}, acc {acc: .3f}, "
                        f"auc {auc: .3f}, ap {ap: .3f}, pr {pr: .3f}, "
                        f"rec5 {rec5: .3f}, rec10 {rec10: .3f}, rec20 {rec20: .3f}        ",
                        end='\r')
                self.logs['train_err'].append(err)
                self.logs['train_acc'].append(acc)
                self.logs['train_auc'].append(auc)
                self.logs['train_ap'].append(ap)
                self.logs['train_rec5'].append(rec5)
                self.logs['train_rec10'].append(rec10)
                self.logs['train_rec20'].append(rec20)
            if verbose: print()
            # validation loop
            err, acc, auc, ap, pr = 0, 0, 0, 0, 0
            n_val = len(val_batches)
            for X, y in val_batches:
                y_pred = self.nn(X, training=False)
                err += float(self.loss_nn(y, y_pred))
                acc = accuracy_score(y, y_pred)
                auc = roc_auc_score(y, y_pred)
                ap = average_precision_score(y, y_pred)
                rec5 = PrecisionAtRecall(recall = 0.05)(y, y_pred)
                rec10 = PrecisionAtRecall(recall = 0.1)(y, y_pred)
                rec20 = PrecisionAtRecall(recall = 0.2)(y, y_pred)
            err, acc, auc, ap, rec5, rec10, rec20 = np.array(err, acc, auc, ap, rec5, rec10, rec20) / n_val
            if verbose:
                print(f" val: err {err: .3f}, acc {acc: .3f}, "
                        f"auc {auc: .3f}, ap {ap: .3f}, pr {pr: .3f}"
                        f"rec5 {rec5: .3f}, rec10 {rec10: .3f}, rec20 {rec20: .3f}")
            self.logs['val_err'].append(err)
            self.logs['val_acc'].append(acc)
            self.logs['val_auc'].append(auc)
            self.logs['val_ap'].append(ap)
            self.logs['val_rec5'].append(rec5)
            self.logs['val_rec10'].append(rec10)
            self.logs['val_rec20'].append(rec20)

    def train_gb(self, batches, verbose=True):
        train_batches, val_batches = batches
        inputs_train, labels_train = train_batches
        inputs_val, labels_val = val_batches
        # get neural network without the last layer
        inputs = self.nn.inputs
        outputs = self.nn.layers[-2].outputs
        partial_nn = tf.keras.Model(inputs=inputs, outputs=outputs)

        X_train = partial_nn(inputs_train).numpy().reshape((-1,))
        y_train = labels_train.numpy().reshape((-1,))
        X_val = partial_nn(inputs_val).numpy().reshape((-1,))
        y_val = labels_val.numpy().reshape((-1,))
        self.gb.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=verbose)

    def predict_proba(self, X_target, X_target_bkg):
        kl = tf.keras.layers
        X = self.batcher_pred(X_target)
        # get neural network without the last layer
        # and copy it into a new algorithm with different seq_len
        inputs = self.nn.inputs
        outputs = self.nn.layers[-2].outputs
        partial_nn = tf.keras.Model(inputs=inputs, outputs=outputs)
        test_nn = tf.keras.Sequential([
                kl.LSTM(self.n_features,
                        return_sequences=False, stateful=False,
                        input_shape=(-1, 10),
                        activation='tanh'),
                kl.Dense(self.n_features, activation='linear'),
                ])
        test_nn.set_weights(partial_nn.get_weights())

        X = test_nn(X).numpy().reshape((-1,))
        y_proba = self.gb.predict_proba(X)
        return y_proba

    def batcher_pred(self, X):
        """Split data to be correctly predicted by neural network
        (1 prediction per day)

        Parameters
        ----------
        X : np.array
            data to split

        Returns
        ------
        X_tensor : tf.Tensor
            new data correctly splitted
        """
        raise NotImplementedError("batcher_pred must be "
                                  "implemented for predictions")

    class UnbalancedMSE_nn(tf.keras.metrics.Metric):
        def __init__(self, data='source', factor=9.0, name='unbalanced_mse', **kwargs):
            """
            Parameters
            ----------
            data : str, optional
                type of data to cut. Should be 'source' or 'target',
                by default 'source'
            factor : float, optional
                multiplicative factor for label 1, by default 9.0
            name : str, optional
                name of the function, by default 'unblanced_mse'
            **kwargs:
                keywords arguments for keras.metrics init
                (could be empty)
            """
            super(UnbalancedMSE_nn, self).__init__(name=name, **kwargs)
            self.data = data
            self.factor = factor

        def __call__(self, y, y_pred):
            """Compute unbalenced MSE

            Parameters
            ----------
            y : tf.Tensor
                true labels
            y_pred : tf.Tensor
                predicted labels

            Returns
            ----------
            err : tf.Tensor
                result of unbalanced MSE loss
            """
            err = tf.square(y_pred - y)
            err = tf.where(y == 1.0, self.factor * err, err)
            err = tf.math.reduce_mean(err)
            return err

    class UnbalancedMSE_gb(tf.keras.metrics.Metric):
        def __init__(self, data='source', factor=9.0):
            """
            Parameters
            ----------
            data : str, optional
                type of data to cut. Should be 'source' or 'target',
                by default 'source'
            factor : float, optional
                    multiplicative factor for label 1, by default 9.0
            """
            self.data = data
            self.factor = factor

        def __call__(self, y, y_pred):
            """Compute unbalenced MSE

            Parameters
            ----------
            y : np.array
                true labels
            y_pred : np.array
                predicted labels

            Returns
            ----------
            grad : np.array
                The value of the first order derivative (gradient)
                of the loss with respect to the elements of y_pred
                for each sample point.
            hess : np.array
                The value of the second order derivative (Hessian)
                of the loss with respect to the elements of y_pred
                for each sample point.
            (see https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html for details)
            """
            # the loss is (y - ypred)**2, if y = 0
            # and self.factor * (y - ypred)**2, otherwise
            grad_mse = 2.0 * (y_pred - y)
            grad = np.where(y == 1.0, self.factor * grad_mse, grad_mse)
            hess = np.where(y == 1.0, self.factor * 2.0, 2.0)
            return grad, hess
