import os

import lightgbm
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

class Classifier:
    def __init__(self):
        kl = tf.keras.layers
        # training & architecture parameters
        self.n_epoch_nn = 1
        self.train_batch_size = 64
        self.test_batch_size = 64
        self.lr_nn = 1e-3
        self.lr_gb = 0.1
        self.timestamp = 96
        self.n_features = 64

        # neural network
        self.loss_nn = self.UnbalancedMSE_nn(data='source')
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
        self.loss_gb = self.UnbalancedMSE_gb(data='source')
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
        batches = self.make_batches_train(X_source, y_source, val_prop=0.2)
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
                # convert y, y_pred to numpy for metrics calculation
                y_np, y_pred_np = y.numpy(), y_pred.numpy()
                acc = accuracy_score(y_np, y_pred_np)
                auc = roc_auc_score(y_np, y_pred_np)
                ap = average_precision_score(y_np, y_pred_np)
                rec5 = PrecisionAtRecall(recall = 0.05)(y_np, y_pred_np)
                rec10 = PrecisionAtRecall(recall = 0.1)(y_np, y_pred_np)
                rec20 = PrecisionAtRecall(recall = 0.2)(y_np, y_pred_np)

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
                 # convert y, y_pred to numpy for metrics calculation
                y_np, y_pred_np = y.numpy(), y_pred.numpy()
                acc += accuracy_score(y_np, y_pred_np)
                auc += roc_auc_score(y_np, y_pred_np)
                ap += average_precision_score(y_np, y_pred_np)
                rec5 += PrecisionAtRecall(recall = 0.05)(y_np, y_pred_np)
                rec10 += PrecisionAtRecall(recall = 0.1)(y_np, y_pred_np)
                rec20 += PrecisionAtRecall(recall = 0.2)(y_np, y_pred_np)
            err, acc, auc, ap, rec5, rec10, rec20 = np.array(err, acc, auc,ap, rec5, rec10, rec20) / n_val
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
        X = self.make_batches_test(X_target)
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
        
        y = []
        for x_batch in X:
            features = test_nn(x_batch).numpy()
            y_proba = self.gb.predict_proba(features).reshape((-1,))
            y.append(y_proba)
        y = np.concatenate(y, axis=0)            
        return y

    def batcher_pred(self, *args, **kwargs):
        raise NotImplementedError("Deprecatd fonction, please use make_batch!")

    def UnbalancedMSE_nn(self, y, y_pred, factor=9.0):
        """Compute unbalenced MSE

        Parameters
        ----------
        y : tf.Tensor
            true labels
        y_pred : tf.Tensor
            predicted labels
        factor : float, optional
            multiplicative factor for label 1, by default 9.0

        Returns
        ----------
        err : tf.Tensor
            result of unbalanced MSE loss
        """
        err = tf.square(y_pred - y)
        err = tf.where(y == 1.0, factor * err, err)
        err = tf.math.reduce_mean(err)
        return err

    def UnbalancedMSE_gb(self, y, y_pred, factor=9.0):
        """Compute unbalenced MSE

        Parameters
        ----------
        y : np.array
            true labels
        y_pred : np.array
            predicted labels
        factor : float, optional
            multiplicative factor for label 1, by default 9.0

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
        """
        # the loss is (y - ypred)**2, if y = 0
        # and self.factor * (y - ypred)**2, otherwise
        grad_mse = 2.0 * (y_pred - y)
        grad = np.where(y == 1.0, factor * grad_mse, grad_mse)
        hess = np.where(y == 1.0, factor * 2.0, 2.0)
        return grad, hess
    
    
    def make_batches_test(self, X):
        test_batches = None
        n_total = len(X)
        full_timestamps = X.shape[1]

        n_test_batches = n_total - n_total % self.test_batch_size
        X_test = X[0:n_test_batches]

        X_test_batches = np.split(X_test, 1, axis=1)
        X_test_batches = np.stack(X_test_batches, axis=0)
        X_test_batches = np.split(X_test_batches, n_test_batches // self.test_batch_size, axis=1)
        X_test_batches = np.stack(X_test_batches, axis=0)
        X_test_batches = X_test_batches.reshape(-1, self.test_batch_size, full_timestamps, 10)       

        test_batches = tf.convert_to_tensor(X_test_batches.squeeze())

        return test_batches

    def make_batches_train(self, X, y, val_prop = 0.2):
        """
        Parameters
        ----------
        X : np.array with dimension (users, timestamps, features)
            data (inputs)
        y : np.array with dimension (users,)
            labels
        Returns
        -------
        if pred_mode == 'train' :
            train_batches : tuple of tf.Tensor (X_train_batches, y_train_batches)
                Batched train data, of length batch_size. The number of batches is : n_train_batches//self.batch_size
                train_batches[0] : data train batches (number of batches, batch_size, timestamps, features)
                train_batches[1] : labels train batches (number of batches, batch_size)
                
            val_batches : tuple of tf.Tensor (X_train_batches, y_train_batches)
                Batched train data, of length batch_size. The number of batches is : n_train_batches//self.batch_size
                val_batches[0] : data validation batches (number of batches, batch_size, timestamps, features)
                val_batches[1] : labels validation batches (number of batches, batch_size)
        
        if pred_mode == 'test':[]:
            test_batches : tuple of tf.Tensor (X_test_batches, y_test_batches)
                Batched train data, of length batch_size. The number of batches is : n_test_batches//self.batch_size
                test_batches[0] : data test batches (number of batches, batch_size, timestamps, features)
                test_batches[1] : labels test batches (number of batches, batch_size)
                
        """
        #train mode
        train_batches, val_batches = None, None
        n_total = len(X)
        full_timestamps = X.shape[1]

        n_train = int((1.0 - val_prop) * n_total)
        n_train_batches = n_train - n_train % self.train_batch_size
        X_train = X[0:n_train_batches]
        y_train = y[0:n_train_batches]

        n_val = int(val_prop * n_total)
        n_val_batches = n_val - n_val % self.train_batch_size
        X_val = X[n_train:(n_train + n_val_batches)]
        y_val = y[n_train:(n_train + n_val_batches)]
            
        X_train_batches = np.split(X_train, full_timestamps // self.timestamp, axis = 1)
        X_train_batches = np.stack(X_train_batches, axis=0)
        X_train_batches = np.split(X_train_batches, n_train_batches // self.train_batch_size, axis=1)
        X_train_batches = np.stack(X_train_batches, axis=0)
        X_train_batches = X_train_batches.reshape(-1, self.train_batch_size, self.timestamp, 10)       
        y_train_batches = np.split(y_train, n_train_batches // self.train_batch_size, axis=0)
        y_train_batches = np.stack(y_train_batches, axis=0)
        y_train_batches = np.repeat(y_train_batches, full_timestamps // self.timestamp, axis = 0)

        X_val_batches = np.split(X_val, full_timestamps // self.timestamp, axis=1)
        X_val_batches = np.stack(X_val_batches, axis=0)
        X_val_batches = np.split(X_val_batches, n_val_batches // self.train_batch_size, axis=1)
        X_val_batches = np.stack(X_val_batches, axis=0)
        X_train_batches = X_train_batches.reshape(-1, self.train_batch_size, self.timestamp, 10)       
        y_val_batches = np.split(y_val, n_val_batches // self.train_batch_size, axis=0)
        y_val_batches = np.stack(y_val_batches, axis=0)
        y_val_batches = np.repeat(y_val_batches, full_timestamps // self.timestamp, axis=0)

        X_train_batches = tf.convert_to_tensor(X_train_batches)
        y_train_batches = tf.convert_to_tensor(y_train_batches)

        X_val_batches = tf.convert_to_tensor(X_val_batches)
        y_val_batches = tf.convert_to_tensor(y_val_batches)

        train_batches = (X_train_batches, y_train_batches)
        val_batches = (X_val_batches, y_val_batches)
    
        return train_batches, val_batches
    