import os

import numpy as np
import tensorflow as tf

from train_utils.loss import UnbalancedMSE_nn
from train_utils.batch import MakeBatch
from train_utils.metric import accuracy, area_ROC, av_precision, prec_recall
from train_utils.model import load_model, save_model
kl = tf.keras.layers

def train_source(model, opt, batches, n_epochs):
    logs = {'train_err':[], 'train_acc':[], 'train_auc':[], 'train_ap':[],
            'train_pr':[], 'val_err':[], 'val_acc':[], 'val_auc':[],
            'val_ap':[], 'val_pr':[]}
    train_batches, val_batches = batches
    loss = UnbalancedMSE_nn(data='source')
    for epoch in n_epochs:
        print(f'epoch {epoch+1}/{n_epochs}')
        for i, X, y in enumerate(train_batches):
            with tf.GradientTape as tape:
                y_pred = model(X)
                err_tensor = loss(y, y_pred)
            grads = tape.gradient(err_tensor, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            err = float(err_tensor)
            acc = accuracy(y, y_pred)
            auc = area_ROC(y, y_pred)
            ap = av_precision(y, y_pred)
            pr = prec_recall(y, y_pred)
            print(f" batch {i+1}, err {err: .3f}, acc {acc: .3f}, "
                  f"auc {auc: .3f}, ap {ap: .3f}, pr {pr: .3f}        ",
                  end='\r')
            logs['train_err'].append(err)
            logs['train_acc'].append(acc)
            logs['train_auc'].append(auc)
            logs['train_ap'].append(ap)
            logs['train_pr'].append(pr)

        print()
        err, acc, auc, ap, pr = 0, 0, 0, 0, 0
        n_val = len(val_batches)
        for X, y in val_batches:
            y_pred = model(X, training=False)
            err += float(loss(y, y_pred))
            acc = accuracy(y, y_pred)
            auc = area_ROC(y, y_pred)
            ap = av_precision(y, y_pred)
            pr = prec_recall(y, y_pred)
        err, acc, auc, ap, pr = np.array(err, acc, auc, ap, pr) / n_val
        print(f" val: err {err: .3f}, acc {acc: .3f}, "
                  f"auc {auc: .3f}, ap {ap: .3f}, pr {pr: .3f}")
        logs['val_err'].append(err)
        logs['val_acc'].append(acc)
        logs['val_auc'].append(auc)
        logs['val_ap'].append(ap)
        logs['val_pr'].append(pr)


def train_target(model, opt, batches, n_epochs):
    logs = {'train_err':[], 'train_acc':[], 'train_auc':[], 'train_ap':[],
            'train_pr':[], 'val_err':[], 'val_acc':[], 'val_auc':[],
            'val_ap':[], 'val_pr':[]}
    train_batches, val_batches = batches
    loss = UnbalancedMSE_nn(data='target')
    for epoch in n_epochs:
        print(f'epoch {epoch+1}/{n_epochs}')
        for i, X, y in enumerate(train_batches):
            with tf.GradientTape as tape:
                y_pred = model(X)
                err_tensor = loss(y, y_pred)
            grads = tape.gradient(err_tensor, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            err = float(err_tensor)
            acc = accuracy(y, y_pred)
            auc = area_ROC(y, y_pred)
            ap = av_precision(y, y_pred)
            pr = prec_recall(y, y_pred)
            print(f" batch {i+1}, err {err: .3f}, acc {acc: .3f}, "
                  f"auc {auc: .3f}, ap {ap: .3f}, pr {pr: .3f}        ",
                  end='\r')
            logs['train_err'].append(err)
            logs['train_acc'].append(acc)
            logs['train_auc'].append(auc)
            logs['train_ap'].append(ap)
            logs['train_pr'].append(pr)

        print()
        err, acc, auc, ap, pr = 0, 0, 0, 0, 0
        n_val = len(val_batches)
        for X, y in val_batches:
            y_pred = model(X, training=False)
            err += float(loss(y, y_pred))
            acc = accuracy(y, y_pred)
            auc = area_ROC(y, y_pred)
            ap = av_precision(y, y_pred)
            pr = prec_recall(y, y_pred)
        err, acc, auc, ap, pr = np.array(err, acc, auc, ap, pr) / n_val
        print(f" val: err {err: .3f}, acc {acc: .3f}, "
                  f"auc {auc: .3f}, ap {ap: .3f}, pr {pr: .3f}")
        logs['val_err'].append(err)
        logs['val_acc'].append(acc)
        logs['val_auc'].append(auc)
        logs['val_ap'].append(ap)
        logs['val_pr'].append(pr)


if __name__ == '__main__':
    ## Model Creation
    # load/save model
    model_name = "NNtest"
    load_model_path = None
    save_model_path = f"models/{model_name}"

    # architecture parameters
    # (should match loading model if load_model_path is not None)
    units = 64
    add_dense = False

    # training parameters
    n_epochs_source = 1
    n_epochs_target = 1
    lr = 1e-3
    batch_size = 128
    dropout = 0.0
    recurrent_dropout = 0.0

    if add_dense:
        model = tf.keras.Sequential([
            kl.LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout,
                    return_sequences=False, stateful=True),
            kl.Dense(units, activation='relu'),
            kl.Dense(units, activation='linear'),
            kl.Dense(1, activation='linear')])
    else:
        model = tf.keras.Sequential([
                kl.LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout,
                        return_sequences=False, stateful=True),
                kl.Dense(units, activation='linear'),
                kl.Dense(1, activation='linear')])

    model = load_model(model, load_model_path)
    model.name = model_name

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt)

    # $$ implement X_source, y_source extraction $$
    X_source, y_source = None, None
    batches = MakeBatch(batch_size=batch_size, data='source')(X_source, y_source)
    train_source(model, opt, batches, n_epochs_source)

    # $$ implement X_source, y_source extraction $$
    X_target, y_target = None, None
    batches = MakeBatch(batch_size=batch_size, data='target')(X_target, y_target)
    train_target(model, opt, batches, n_epochs_target)

    save_model(model, save_model_path, units=units, add_dense=add_dense)
