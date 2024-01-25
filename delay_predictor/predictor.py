"""
Util module to call the model and test its performances
"""
import numpy as np
import pandas as pd
import pickle

import delay_predictor.trainer

with open("encoder.pickle", "rb") as f:
    ENCODER = pickle.load(f)

with open("model.pickle", "rb") as f:
    MODEL = pickle.load(f)


def predict_from_raw_data(df) -> np.array:
    """
    Applies ColumnsTransformers and model to the given DataFrame
    Args:
        df: pd.DtaFrame
        encoder: ColumnTransformer to apply to df
        model: tf.keras.Model

    Returns:
        Prediction as a 1-D np.array
    """
    X_encoded = ENCODER.transform(df)
    X_encoded = pd.DataFrame(X_encoded, columns=[delay_predictor.trainer.NUMERICAL + delay_predictor.trainer.CATEGORICAL])
    X_multi_input = delay_predictor.trainer.create_multi_input_data(X_encoded)
    y_pred = MODEL.predict(X_multi_input)
    return y_pred[:, 0]

