"""
Functions to create the training dataset
"""
import pandas as pd


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters only American Airlines data
    Args:
        df: pd.DataFrame

    Returns:
        pd.DataFrame
    """
    mask = df["UniqueCarrier"] == "AA"
    df = df[mask].copy()
    return df


def process_date_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the departure datetime column
    Args:
        df:  pd.DataFrame

    Returns:
        pd.DataFrame
    """
    df["DepTime"] = df["DepTime"].str.pad(width=4, side="left", fillchar="0")
    df["DepTime"] = df["DepTime"].str.replace("2400", "0000")
    df["departure_datetime"] = df["Date"] + "T" + df["DepTime"]
    df["departure_datetime"] = pd.to_datetime(df["departure_datetime"], format="%d-%m-%YT%H%M")
    return df


def build_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the departure datetime
    Args:
        df: pd.DataFrame

    Returns:
        pd.DataFrame
    """
    df["departure_hour"] = df["departure_datetime"].dt.hour
    df["departure_month"] = df["departure_datetime"].dt.month
    df["departure_day_of_week"] = df["departure_datetime"].dt.dayofweek
    return df


def get_previous_arrrival_delay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get previous arrival delay of a given tail number (= aircraft unique id)
    If previous flight is not sane (e.g. previous dest != actial origin), or
    if the previous flight is unkown, this value is replaces by -1
    Args:
        df: pd.DataFrame that contains a list of flight

    Returns:
        pd.DataFrame
    """
    df = df.sort_values("departure_datetime").reset_index(drop=True)
    df[["previous_arrival_delay", "previous_origin", "previous_destination"]] = df.groupby("TailNum")[
        ["ArrDelay", "Origin", "Dest"]].shift(1)

    mask_unsane_previous_flights = df["previous_destination"] != df["Origin"]

    df[mask_unsane_previous_flights][["previous_arrival_delay"]] = -1
    df["previous_arrival_delay"] = df["previous_arrival_delay"].fillna(-1)
    df.reset_index(drop=True)

    return df


def select_and_rename_training_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames CamelCase column name to snake_cane
    Selects the training columns
    Args:
        df:

    Returns:

    """
    cols = [
        "departure_hour",
        "departure_month",
        "departure_day_of_week",
        "previous_arrival_delay",
        "TailNum",
        "Origin",
        "Dest",
        "DepDelay"
    ]

    df = df[cols].copy()

    df = df.rename(columns={
        "TailNum": "tail_number",  # Unique id of an aircraft
        "Origin": "origin",  # IATA Code of the departure airport
        "Dest": "destination",  # IATA Code of the departure airport
        "DepDelay": "departure_delay"  # Departure delay in minutes
    })

    return df


def run() -> pd.DataFrame:
    """
    Feature engineering routine
    Returns:
        pd.DataFrame
    """
    df = pd.read_csv("Flight_delay.csv", dtype={"DepTime": str, "ArrTime": str, "CRSArrTime": str})
    df = filter_df(df)
    df = process_date_and_time(df)
    df = build_datetime_features(df)
    df = get_previous_arrrival_delay(df)
    df = select_and_rename_training_variables(df)

    return df


def main():
    df = run()
    df.to_parquet("training_data.parquet")


if __name__ == "__main__":
    main()
