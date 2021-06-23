import numpy as np
import tensorflow as tf

def nn_prediction(model, X):
    """Split the inputs to make prediction each days

    Parameters
    ----------
    model : keras.Model
        neural ntwork model
    X : np.array
        inputs

    Returns
    -------
    predictions : tf.Tensor
        [description]

    Raises
    ------
    NotImplementedError
        raise error if the fonction is not yet implemented
    """
    # $$ implement X splitting to predict each days $$
    raise NotImplementedError("X splitting")
    return model(X)