import numpy as np
import pandas as pd
from sklearn import preprocessing

class FeatureExtractor(object):

    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.method = 'linear'
        pass

    def transform(self, X):

        X_new = X.copy()

        for i, user in enumerate(X):
            # set to 0 all columns that are only filled with nan
            user[:, np.isnan(user).all(axis=0)] = 0.0
            # fill all nan that are inside  non-nan values using 'linear'
            df = pd.DataFrame(user).interpolate(method=self.method,
                                                limit=None,
                                                inplace=False,
                                                limit_direction=None,
                                                limit_area=None)
            # fill all nan ouside non-nan values by copying last non-nan values
            df = df.ffill().bfill()
            val = df.values
            val = self.scaler.fit_transform(val)
            X_new[i] = val
        return X_new

    def __call__(self, X):
        return self.transform(self, X)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    # nan_factor (could be positive or negative) greater
    # the factor greater the proportion of nan (0 correspond of
    # ~ 0.5 of values)
    nan_factor = 0
    # get random inputs with different distributions over users and features
    X = np.random.randn(100, 500, 10) * np.random.randn(100, 1, 10) * 10 \
        + np.random.randn(100, 1, 10) * 10
    # fill random values with nan
    X = np.where(np.random.randn(100, 500, 10) < nan_factor, np.nan, X)
    # take a random user
    i = np.random.randint(0, len(X))
    user = X[i]
    df = pd.DataFrame(user)
    print('before transform:')
    print(df.describe())
    print('missing values:\n', np.sum(np.isnan(user), axis=0))
    fe = FeatureExtractor()
    X = fe.transform(X)
    user = X[20]
    df = pd.DataFrame(user)
    print('\nafter transform:')
    print(df.describe())
    print('missing values:\n', np.sum(np.isnan(user), axis=0))
