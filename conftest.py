"""
Data and model fixtures for unit tests
"""
import pytest
import pickle

import pandas as pd
from giskard import Dataset, transformation_function, slicing_function
from delay_predictor.trainer import CATEGORICAL, NUMERICAL, TARGET


@pytest.fixture(scope="function")
def df():
    return pd.read_parquet("./df_test.parquet")


@pytest.fixture(scope="function")
def giskard_dataset():
    df = pd.read_parquet("./df_test.parquet")
    giskard_dataset = Dataset(
        df=df,
        target="departure_delay",
        name="flight_delays",
        cat_columns=CATEGORICAL
    )
    return giskard_dataset


@pytest.fixture(scope="module")
def encoder():
    with open("encoder.pickle", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def model():
    with open("model.pickle", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
@transformation_function(name="increase previous delay")
def increase_previous_delay(row):
    row["previous_arrival_delay"] = row["previous_arrival_delay"] + 5.
    return row


@pytest.fixture(scope="module")
@transformation_function(name="increase previous delay")
def decrease_previous_delay(row):
    row["previous_arrival_delay"] = row["previous_arrival_delay"] / 1.2
    return row
