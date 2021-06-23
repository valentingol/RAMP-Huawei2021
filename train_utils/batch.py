import tensorflow as tf

class MakeBatch(object):
    """Cut data into batches to be used for LSTM training
    """
    def __init__(self, batch_size=64, data='source', val_prop=0.2):
        """
        Parameters
        ----------
        batch_size : int, optional
            size of batch, by default 64
        data : str, optional
            type of data to cut. Should be 'source' or 'target',
            by default 'source'
        val_prop : float, optional
            proportion of validation data, by default 0.2
        """
        self.batch_size = 64
        self.data = data
        self.val_prop = val_prop

    def __call__(self, X, y):
        """
        Parameters
        ----------
        X : np.array
            data (inputs)
        y : np.array
            labels

        Returns
        -------
        train_batches : tf.Tensor
            Batched train data, of length batch_size
        val_batches : tf.Tensor
            Batched train data, of length batch_size
        """
        # $$ implement batch cut $$
        train_batches, val_batches = None, None
        return train_batches, val_batches