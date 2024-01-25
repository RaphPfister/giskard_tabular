"""
Test module for predict, including test cases for the model
"""
import giskard

from delay_predictor.predictor import predict_from_raw_data

CATEGORICAL = ["tail_number", "origin", "destination"]
NUMERICAL = ["departure_hour", "departure_month", "departure_day_of_week", "previous_arrival_delay"]
TARGET = "departure_delay"


def test_predict_from_raw_data(df):
    # GIVEN
    # fixtures
    
    # WHEN
    predictions = predict_from_raw_data(df=df)
    
    # THEN
    assert predictions.shape[0] == len(df)


def test_previous_arrival_delay_increase(df, giskard_dataset, increase_previous_delay):
    # GIVEN
    # fixtures
    giskard_model = giskard.Model(
        model=predict_from_raw_data,
        model_type="regression",
        name="delay_prediction_model",
        feature_names=CATEGORICAL + NUMERICAL,
        cat_columns=CATEGORICAL
    )

    giskard_dataset = giskard.Dataset(
        df=df,
        target="departure_delay",
        name="flight_delays",
        cat_columns=CATEGORICAL
    )

    test = giskard.testing.test_metamorphic_increasing(
        model=giskard_model,
        dataset=giskard_dataset,
        transformation_function=increase_previous_delay
    )

    # WHEN
    result = test.execute()

    # THEN
    assert result.passed
