import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

from random import shuffle
import folium


def get_similarity_df(similarity_df, mode="relative"):
    if mode == "relative":
        similarity_df = similarity_df.div(similarity_df.sum(axis=1), axis=0)

    scaler = MinMaxScaler()
    scaler.fit(similarity_df)
    similarity_data_scaled = scaler.transform(similarity_df)
    similarity_data_scaled = pd.DataFrame(similarity_data_scaled, columns=similarity_df.columns,
                                          index=similarity_df.index)
    return similarity_data_scaled, scaler


def show_station_clustering(station, df_cluster_labeled):
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                  'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                  'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
                  'gray', 'black', 'lightgray']

    #     color_list = list(mcolors.CSS4_COLORS.keys())
    #     shuffle(color_list)
    print(len(color_list))

    def mark_color(cluster_label, color_list):
        unique_label = np.unique(cluster_label)
        label_size = len(unique_label)
        color_size = len(color_list)
        label_color = dict()
        for label in unique_label:
            label_color[label] = color_list[label % color_size]

        return label_color

    label_color_dict = mark_color(df_cluster_labeled["cluster"], color_list)

    station_labeled = station.merge(df_cluster_labeled[['cluster']], left_on='staID', right_index=True)
    station_labeled['label'] = station_labeled['cluster']  ##cluster.labels_
    station_labeled['label_color'] = station_labeled['label'].apply(lambda label: label_color_dict[label])

    # latlon = [(51.249443914705175, -0.13878830247011467),
    #           (51.249443914705175, -0.13878830247011467),
    #           (51.249768239976866, -2.8610415615063034)]
    latlon = [tuple(x) for x in
              station_labeled[['start_station_lat', 'start_station_lon', 'label', 'label_color']].values]
    mapit = folium.Map(location=[41.8781, -87.623177], zoom_start=12)
    for coord in latlon:
        folium.Marker(location=[coord[0], coord[1]], popup=str(coord[2]), icon=folium.Icon(color=coord[3])).add_to(
            mapit)
    #     folium.Marker( location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=8 ).add_to( mapit )

    mapit.save('map.html')

    return mapit, station_labeled


def process_clustering(trip_df, n_clusters, station, mode="absolute"):
    trip_date_year_businessday = trip_df
    trip_date_year_groupby = trip_date_year_businessday[
        ['from_station_id', 'weekday_hour_period', 'starttime']].groupby(
        ['from_station_id', 'weekday_hour_period']).agg(['count'])

    trip_date_year_groupby = trip_date_year_groupby.reset_index()
    trip_date_year_groupby.columns = ['from_station_id', 'weekday_hour_period', 'visit_count']
    similarity_data = trip_date_year_groupby.pivot_table(values='visit_count',
                                                                            index='from_station_id',
                                                                            columns='weekday_hour_period',
                                                                            aggfunc=np.sum)
    similarity_data.fillna(0, inplace=True)

    if mode == "absolute":
        similarity_data_scaled, scaler_activity_absolute = get_similarity_df(similarity_data,
                                                                             mode="absolute")
    else:
        similarity_data_scaled, scaler_activity_relative = get_similarity_df(similarity_data)

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(similarity_data_scaled)
    similarity_data_scaled['cluster'] = ward.labels_
    similarity_data['cluster'] = ward.labels_

    station_labeled = station.merge(similarity_data[['cluster']], left_on='staID', right_index=True)
    return similarity_data, station_labeled



