import os
import pytest
import pandas as pd
import numpy as np
import clustering
import data_utils


def test_process_clustering():
    trip_date_year = pd.read_pickle(os.path.join(".", "test_data", "trip_data_year.pkl"))
    station = pd.read_pickle(os.path.join(".", "test_data", "station_clustered.pkl"))
    station = station.drop("cluster", axis=1)
    n_clusters = 16
    similarity_df, station_clustered = clustering.process_clustering(trip_date_year, n_clusters, station,
                                                                     mode="absolute")
    assert len(station_clustered['cluster'].unique()) == 16


