import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

from sklearn.metrics import r2_score
from matplotlib.pyplot import figure

import config
import data_cleaning
import clustering
import ml_pipeline


def ml_main(trip_date_year, station, weather_period_gb, forecast_date, trace_back_days=365, test_days=1, n_clusters=16):
    print(f"forecast_date: {forecast_date}")

    # Filter data in data range
    print(f"date start: {forecast_date - dt.timedelta(days=trace_back_days)}")
    print(f"date end: {forecast_date + dt.timedelta(days=test_days)}")

    print(f"trip_date_year.dtypes: {trip_date_year.dtypes}")

    trip_date_year = trip_date_year[
        (trip_date_year['starttime_dt'] >= (forecast_date - dt.timedelta(days=trace_back_days))) & (
                trip_date_year['starttime_dt'] <= (forecast_date + dt.timedelta(days=test_days)))]

    # process and merge trip, station, weather data into cleaned data df
    data = data_cleaning.merge_trip_station_weather(trip_date_year, station, weather_period_gb)

    # save processed data
    # data.to_pickle("precessed_data.pkl")

    # Clustering sites by activity
    similarity_df, station_clustered = clustering.process_clustering(trip_date_year, n_clusters, station)

    # Generate time series data
    data_train = data[data['date'] < forecast_date]
    data_test = data[data['date'] >= forecast_date]

    data_train = data_train.drop('date', axis=1)
    data_test = data_test.drop('date', axis=1)
    sites_list = sorted(station_clustered['staID'].unique())
    whole_pred = ml_pipeline.whole_data_forecast(data_train, data_test, station_clustered, sites_list)

    r2 = 0

    try:
        print(f"data_test['trip count']: {data_test['trip count'].shape}, whole_pred shape: {len(whole_pred)}")
        r2 = r2_score(data_test['trip count'], whole_pred)
        print(f"r2: {r2}")
    except:
        print("r2 score calculation error!")
        print(f"whole_pred shape: {len(whole_pred)}, data_test shape: {data_test.shape}")

    figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(range(len(data_test['trip count'])), data_test['trip count'], "b*")
    plt.plot(range(len(data_test['trip count'])), whole_pred, "r.")
    plt.show()

    return whole_pred, r2


if __name__ == "__main__":
    forecast_end_date = '2016-08-01'
    process_days = 5
    trace_back_days = 365
    test_days = 1
    n_clusters = 2
    load_processed_data = False

    print("Start data cleansing")

    # 1. data cleansing
    data_folder_path = config.DATA_FOLDER_PATH

    # Load data
    saved_processed_data_path = config.PROCESSED_DATA_SAVE_PATH

    if not load_processed_data:
        station, weather, trip_date_year = data_cleaning.load_data(data_folder_path)
        # Process daily weather data
        weather_period_gb, weather_daily_gb = data_cleaning.weather_process(weather)
        # Process trip data
        trip_date_year = data_cleaning.trip_data_process(trip_date_year)
        print(f"weather_period_gb.dtypes: {weather_period_gb.dtypes}")

        station.to_pickle(os.path.join(saved_processed_data_path, "station.pkl"))
        weather_period_gb.to_pickle(os.path.join(saved_processed_data_path, "weather_period_gb.pkl"))
        trip_date_year.to_pickle(os.path.join(saved_processed_data_path, "trip_date_year.pkl"))

        print("End data cleansing")
    else:
        station = pd.read_pickle(os.path.join(saved_processed_data_path, "station.pkl"))
        weather_period_gb = pd.read_pickle(os.path.join(saved_processed_data_path, "weather_period_gb.pkl"))
        trip_date_year = pd.read_pickle(os.path.join(saved_processed_data_path, "trip_date_year.pkl"))

        print("End load processed data")

    # 2. forecast for each date in period
    forecast_date_range = pd.date_range(end=forecast_end_date, periods=process_days, closed='right')

    all_predictions = []
    all_r2_score = []
    for forecast_date in forecast_date_range:
        print(f"Current forecast date: {forecast_date}")
        whole_pred, r2_score = ml_main(trip_date_year, station, weather_period_gb, forecast_date, trace_back_days, test_days, n_clusters)
        print(f"r2: {r2_score}")
        all_predictions.append(whole_pred)
        all_r2_score.append(r2_score)

    print(f"Average r2: {sum(all_r2_score)/len(all_r2_score)}")
