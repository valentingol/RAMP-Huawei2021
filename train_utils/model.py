import os
import tensorflow.keras as keras

def load_model(model, load_model_path=None):
    """Load weights in load_model_path into model

    Parameters
    ----------
    model : keras.Model
        model that receives weights
    load_model_path : string or None, optional
        path where is the weights to load. If None,
        skip the loading function, by default None

    Returns
    -------
    model : keras.Model
        the input model with loaded weights
    """
    if load_model_path is None:
        return model
    if load_model_path[-3:] != ".h5":
        load_model_path = load_model_path + ".h5"
    model.load_weights(load_model_path)
    return model


def save_model(model, save_model_path, **infos):
    """Save weights of a model and save some
    informations about it

    Parameters
    ----------
    model : keras.Model
        model to save
    save_model_path : string
        path of the directory to write weights and informations
    infos :
        kewords argument, additional informations to save
    """
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
    else:
        print("Warning : the model already exists, it will be overwritten !")
    path = os.path.join(save_model_path, model.name)
    model.save_weights(path + '.h5')
    with open(path + '.txt', "w") as f:
        f.write(f"Infos of {model.name}")
        for info in infos:
            f.write(f"{info} : {infos[info]}")
