"""
Util module to call the model and test its performances
"""
import pandas as pd

import delay_predictor.trainer


def predict_from_raw_data(df, encoder, model):
    """
    Applies ColumnsTransformers and model to the given DataFrame
    Args:
        df: pd.DtaFrame
        encoder: ColumnTransformer to apply to df
        model: tf.keras.Model

    Returns:
        Prediction as a 1-D np.array
    """
    X_encoded = encoder.transform(df)
    X_encoded = pd.DataFrame(X_encoded, columns=[delay_predictor.trainer.NUMERICAL + delay_predictor.trainer.CATEGORICAL])
    X_multi_input = delay_predictor.trainer.create_multi_input_data(X_encoded)
    y_pred = model.predict(X_multi_input)
    return y_pred[:, 0]

