"""
"""
from types import NoneType
from typing import List, Union
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Model 
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Reshape

CATEGORICAL = ["tail_number", "origin", "destination"]
NUMERICAL = ["departure_hour", "departure_month", "departure_day_of_week", "previous_arrival_delay"]
TARGET = "departure_delay"


def split(df: pd.DataFrame):
    
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    
    return df_train, df_val


def create_x_y(df: pd.DataFrame):

    X = df_train[NUMERICAL + CATEGORICAL]
    y = df[TARGET].astype(float)
    return X, y
    

def build_encoder(vocab_size_dict: dict):
    
    transformers=[
        ("min_max", MinMaxScaler(), NUMERICAL),
    ]

    # Embedding layer expects unknown categories index to be len(categories) + 1
    for category in CATEGORICAL:
        ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=vocab_size_dict[category])
        transformer = (f"ordinal_{category}", ordinal, [category])
        transformers.append(transformer)
        
    encoder = ColumnTransformer(
        transformers=transformers
    )
    return encoder


def fit_and_encode(X_train, X_val, encoder):
    encoder.fit(X_train)

    X_train_encoded = encoder.transform(X_train)
    X_val_encoded = encoder.transform(X_val)

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=[NUMERICAL + CATEGORICAL])
    X_val_encoded = pd.DataFrame(X_val_encoded, columns=[NUMERICAL + CATEGORICAL])

    return X_train_encoded, X_val_encoded, encoder


def create_multi_input_data(df: pd.DataFrame):
    input_list = []
    for category in CATEGORICAL:
        input_list.append(df[category].to_numpy())

    input_list.append(df[NUMERICAL].to_numpy())
    return input_list
    
    
def embedding_input(category: str, vocab_size: int, embedding_dim: int):

    categorical_input = Input(shape=(1,), name=f"input_{category}")
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name=f"embed_{category}")(categorical_input)
    embedding = Reshape(target_shape=(embedding_dim,))(embedding)
    
    return categorical_input, embedding


def get_vocab_size(df_train):
    
    return {category: df_train[category].nunique() for category in CATEGORICAL}


def build_model(vocab_sizes: dict, dense_dims: Union[List[int], NoneType]=None, learning_rate: float=0.1):
    
    if not dense_dims:
        dense_dims = [32, 1]
    
    input_layers = []
    layers_to_concatenate = []
    
    for category in CATEGORICAL:
        categorical_input, embedding = embedding_input(category=category, vocab_size=vocab_sizes[category]+1, embedding_dim=2)
        layers_to_concatenate.append(embedding)
        input_layers.append(categorical_input)

    numerical_input = Input(shape=(len(NUMERICAL)), name='numerical_input')
    layers_to_concatenate.append(numerical_input)
    input_layers.append(numerical_input)
    
    layer = Concatenate(axis=1)(layers_to_concatenate)
    for dense_dim in dense_dims:
        layer = Dense(dense_dim, activation='relu')(layer)
        
    model = Model(inputs=input_layers, outputs=layer)
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tf.losses.MAE)
    
    return model


def train(df, learning_rate=0.5, epochs=100):
    
    df_train, df_val = split(df)
    
    X_train, y_train = create_x_y(df=df_train)
    X_val, y_val = create_x_y(df=df_val)
    
    vocab_size_dict = get_vocab_size(df_train=df_train)
    
    encoder = build_encoder(vocab_size_dict)

    X_train_encoded, X_val_encoded, encoder = fit_and_encode(X_train, X_val, encoder)
    
    model = build_model(vocab_sizes=vocab_size_dict, dense_dims=[32, 1], learning_rate=learning_rate)

    X_train_input = create_multi_input_data(X_train_encoded)
    X_val_input = create_multi_input_data(X_val_encoded)

    model.fit(x=X_train_input,
              y=y_train,
              validation_data=(X_val_input, y_val),
              epochs=50,
              batch_size=256
             )

    return encoder, model, df_train, df_val


if __name__=="__main__":
    df = pd.read_parquet("training_data.parquet")
    
    encoder, model, df_train, df_val = run(df)
    
    with open("encoder.pickle", "wb") as f:
        pickle.dump(encoder)
        
    with open("model.pickle", "wb") as f:
        pickle.dump(model)
    df_train.to_parquet("df_train.parquet")
    df_val.to_parquet("df_val.parquet")
        