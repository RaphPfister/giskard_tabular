"""
Data and model fixtures for unit tests
"""
import pytest
import pickle

import pandas as pd


@pytest.fixture(scope="function")
def df():
    return pd.read_parquet("./train_data.parquet")


@pytest.fixture(scope="module")
def encoder():
    with open("encoder.pickle", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def model():
    with open("model.pickle", "rb") as f:
        return pickle.load(f)
