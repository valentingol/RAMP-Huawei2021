import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show_shapes(a_data=None, a_labels=None, b_data=None, b_labels=None,
                test_data=None, test_labels=None):
    """displays shapes of all input data"""
    print(' shapes:')
    if a_data is not None: print('a_data:', a_data.shape)
    if a_labels is not None: print('a_labels:', a_labels.shape)
    if b_data is not None: print('b_data:', b_data.shape)
    if b_labels is not None: print('b_labels:', b_labels.shape)
    if test_data is not None: print('test_data:', test_data.shape)
    if test_labels is not None: print('test_labels:', test_labels.shape)


def show_labels(a_labels=None, b_labels=None, test_labels=None):
    """displays information about labels given in arguments"""
    if a_labels is not None:
        print("\n labels A")
        print('n_strong:', np.isclose(a_labels, -1.).sum())
        print('n_weak:', np.isclose(a_labels, 0.).sum())
        print('n_fail:', np.isclose(a_labels, 1.).sum())
        print('fails rate:', np.isclose(a_labels, 1.).sum() / a_labels.size)
    if b_labels is not None:
        print("\n labels B")
        print('n_strong:', np.isclose(b_labels, -1.).sum())
        print('n_weak:', np.isclose(b_labels, 0.).sum())
        print('n_fail:', np.isclose(b_labels, 1.).sum())
        print('missing:', np.isnan(b_labels).sum())
        print('mising rate:', np.isnan(b_labels).sum() / b_labels.size)
    if test_labels is not None:
        print("\n labels test")
        print('n_strong:', np.isclose(test_labels, -1.).sum())
        print('n_weak:', np.isclose(test_labels, 0.).sum())
        print('n_fail:', np.isclose(test_labels, 1.).sum())
        print('missing:', np.isnan(test_labels).sum())
        print('mising rate:', np.isnan(test_labels).sum() / test_labels.size)


def show_data(a_data=None, b_data=None, test_data=None):
    """displays information about data given in arguments"""
    columns = ['current', 'err_down_bip', 'err_up_bip', 'olt_recv', 'rdown',
            'recv', 'rup', 'send', 'temp', 'volt']
    if a_data is not None:
        print("\n A data")
        print("missing values rate:", np.isnan(a_data).sum() / a_data.size)
        df_a_data = pd.DataFrame(a_data.reshape(-1, 10), columns=columns)
        print(df_a_data.describe())

    if b_data is not None:
        print("\n B data")
        print("missing values rate:", np.isnan(b_data).sum() / b_data.size)
        df_b_data = pd.DataFrame(b_data.reshape(-1, 10), columns=columns)
        print(df_b_data.describe())

    if test_data is not None:
        print("\n test data")
        print("missing values rate:", np.isnan(test_data).sum() / test_data.size)
        df_test_data = pd.DataFrame(test_data.reshape(-1, 10), columns=columns)
        print(df_test_data.describe())


def show_sequences(data_1, n_sequences=3):
    """Displays a random sequence of features from data input
    The values are normalized to facilitate comparisons, nan value
    are ignored.

    Parameters
    ----------
    data_1 : np array
        first data to display (in the 2 first lines if data_2 is
        not None)
    n_sequences : int, optional
        number of sequences to display (round by the lower
        multiple of 3 if more than 3), by default 3
    Raises
    ----------
    ValueError : if n_sequences < 3
    """
    n_sequences = (n_sequences // 3) * 3
    if n_sequences < 3:
        raise ValueError('n_sequences must be more than 2')
    indicies = np.random.choice(list(range(data_1.shape[0])), n_sequences)
    color = [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 1., 1.], [1., 0., 1.],
            [1., 1., 0.], [1., 1., 1.],[0.5, 0.5, 0.5], [0.2, 0.6, 0.8],
            [0.8, 0.6, 0.2]]
    plt.figure()
    i = 1
    for indx in indicies:
        plt.subplot(n_sequences//3, 3, i)
        data = data_1[indx]
        data[np.isnan(data)] = 0
        data -= np.mean(data, axis=0, keepdims=True)
        # add a small term to avoid overflow
        data /= np.std(data, axis=0, keepdims=True) + 1e-5
        for p in range(10):
            plt.plot(list(range(len(data))), data[:, p], color=color[p],
                        label=f'{p}')
        if i == 1: plt.legend(loc='upper right', ncol=2)
        i += 1
    plt.show()


def compare_sequences(data_1, data_2=None):
    """Displays 12 random sequences of features from data inputs
    The values are normalized to facilitate comparisons, nan value
    are ignored.

    Parameters
    ----------
    data_1 : np array
        first data to display (in the 2 first lines if data_2 is
        not None)
    data_2 : np array, optional
        second data to display (in the 2 last lines if notnNone),
        by default None
    """
    if data_2 is None:
        indicies_1 = np.random.choice(list(range(data_1.shape[0])), 12, replace=False)
        indicies_2 = []
    else:
        indicies_1 = np.random.choice(list(range(data_1.shape[0])),
                                      6, replace=False)
        indicies_2 = np.random.choice(list(range(data_2.shape[0])),
                                      6, replace=False)
    color = [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 1., 1.], [1., 0., 1.],
            [1., 1., 0.], [1., 1., 1.],[0.5, 0.5, 0.5], [0.2, 0.6, 0.8],
            [0.8, 0.6, 0.2]]
    plt.figure()
    i = 1
    for indx in indicies_1:
        plt.subplot(4, 3, i)
        data = data_1[indx]
        data[np.isnan(data)] = 0
        data -= np.mean(data, axis=0, keepdims=True)
        # add a small term to avoid overflow
        data /= np.std(data, axis=0, keepdims=True) + 1e-5
        for p in range(10):
            plt.plot(list(range(len(data))), data[:, p], color=color[p],
                     label=f'{p}')
        if i == 1: plt.legend(loc='upper right', ncol=2)
        i += 1
    for indx in indicies_2:
        plt.subplot(4, 3, i)
        data = data_2[indx]
        data[np.isnan(data)] = 0
        data -= np.mean(data, axis=0, keepdims=True)
        data /= np.std(data, axis=0, keepdims=True) + 1e-5
        for p in range(10):
            plt.plot(list(range(len(data))), data[:, p], color=color[p],
                     label=f'{p}')
        i += 1

    plt.show()


if __name__ == '__main__':
    path_city_a = "./data/city_A"
    path_city_b = "./data/city_B"

    a_data = np.load(os.path.join(path_city_a, "source.npy"))
    a_labels = np.load(os.path.join(path_city_a, "source_labels.npy"))
    b_data = np.load(os.path.join(path_city_b, "target.npy"))
    b_labels = np.load(os.path.join(path_city_b, "target_labels.npy"))
    test_data = np.load(os.path.join(path_city_b, "test.npy"))
    test_labels = np.load(os.path.join(path_city_b, "test_labels.npy"))

    show_shapes(a_data=a_data, a_labels=a_labels, b_data=b_data,
                b_labels=b_labels, test_data=test_data, test_labels=test_labels)
    show_labels(a_labels=a_labels, b_labels=b_labels, test_labels=test_labels)
    show_data(a_data=a_data, b_data=b_data, test_data=test_data)
    show_sequences(a_data, 9)
    compare_sequences(data_1=b_data, data_2=test_data)
