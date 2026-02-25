import numpy as np
from pyod.models.iforest import IForest

def lstm_check(model, threshold, window):
    sequence = np.array(window).reshape(1, 100, 3)
    reconstruction = model.predict(sequence, verbose=0)
    mse = np.mean(np.square(sequence - reconstruction))
    return bool(mse > threshold)
