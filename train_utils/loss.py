import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class UnbalancedMSE_nn(tf.keras.metrics.Metric):
    def __init__(self, data='source', factor=0.9, name='unblanced_mse', **kwargs):
        """
        Parameters
        ----------
        data : str, optional
            type of data to cut. Should be 'source' or 'target',
            by default 'source'
        factor : float, optional
                multiplicative factor for label 1
        name : str, optional
            name of the function, by default 'unblanced_mse'
        **kwargs:
            keywords arguments for keras.metrics init
            (could be empty)
        """
        super(UnbalancedMSE_nn, self).__init__(name=name, **kwargs)
        self.data = data

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
        # $$ implement unbalenced MSE $$
        err = None
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
        grad, hess = None, None
        return grad, hess