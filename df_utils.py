import pandas as pd
import datetime as dt
from collections import defaultdict, deque
from sklearn.preprocessing import MinMaxScaler

def time_period(hour):
    if (hour >= 1 and hour <= 4):
        return "1,1-4"
    elif (hour >= 5 and hour <= 8):
        return "2,5-8"
    elif (hour >= 9 and hour <= 12):
        return "3,9-12"
    elif (hour >= 13 and hour <= 16):
        return "4,13-16"
    elif (hour >= 17 and hour <= 20):
        return "5,17-20"
    elif (hour >= 21 or hour == 0):
        return "6,21-24"


def feature_seperate_group_by(df, min_max_mean_columns, max_columns, min_columns, groupby_columns):
    df_new = pd.DataFrame()
    columns_name = []

    if min_max_mean_columns:
        df_new = df[min_max_mean_columns + groupby_columns].groupby(groupby_columns).agg(['min', 'max', 'mean'])
        for col in min_max_mean_columns:
            columns_name = columns_name + [str(col + ' min'), str(col + ' max'), str(col + ' mean')]

    if max_columns:
        df_p_max = df[max_columns + groupby_columns].groupby(groupby_columns).agg(['max'])
        df_new = df_new.merge(df_p_max, left_index=True, right_index=True)
        columns_name = columns_name + max_columns

    if min_columns:
        df_p_min = df[min_columns + groupby_columns].groupby(groupby_columns).agg(['min'])
        df_new = df_new.merge(df_p_min, left_index=True, right_index=True)
        columns_name = columns_name + min_columns

    df_new.columns = columns_name

    return df_new


def add_dummmies_remove_original_col(df, cols, remove_original_col=True, default_columns=None):
    for col in cols:

        dummies_cols = pd.get_dummies(df[col], columns=default_columns)
        df = pd.concat([df, dummies_cols], axis=1)
        if remove_original_col:
            df.drop([col], axis=1, inplace=True)

    return df


def get_site_timeseries(data, site_no):

    def get_site_data(data, site_id):
        data_picked = data[data['s_staID'] == site_id]
        return data_picked

    site_data = get_site_data(data, site_no)
    site_data = site_data.set_index(['s_staID', 'date', 'hour-period'])

    site_data['next_day_trip_count'] = site_data['trip count'].shift(period_no)
    site_data = site_data.dropna()

    site_data_X = site_data.drop(['trip count', 'next_day_trip_count'], axis=1)
    site_data_y = site_data['next_day_trip_count']
    return site_data_X, site_data_y


# def get_cluster_data(data, station_clustered, cluster_no, sites_list=None):
#     station_picked = station_clustered[station_clustered["cluster"] == cluster_no]
#     station_ids = station_picked['start_station_id'].unique()
#
#     data_picked = data[data['s_staID'].isin(station_ids)]
#     #     data_picked = data_picked.drop("s_staID", axis=1)
#     return data_picked, station_ids


def standardization_data(data):

    scaler = MinMaxScaler()
    data_scale = scaler.fit_transform(data)
    data_scale = pd.DataFrame(data_scale, index=data.index, columns=data.columns)

    return data_scale


def standardize_df(df):

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_scale = scaler.fit_transform(df)
    df_scale = pd.DataFrame(df_scale, index=df.index, columns=df.columns)

    return df_scale, scaler


# def get_cluster_data(data, station_cluster, cluster_no, sites_list=None):
#     station_picked = station_cluster[station_cluster["cluster"] == cluster_no]
#     station_ids = station_picked['start_station_id'].unique()
#
#     data_picked = data[data['s_staID'].isin(station_ids)]
#     #     data_picked = data_picked.drop("s_staID", axis=1)
#     return data_picked, station_ids


def generate_ts_data_dict_by_cluster(df, station_cluster, forecast_date, cluster_no,
                                     features_list, hour_period_list, label_field='trip count', episode_len=26,
                                     label_day_len=1, period_num_per_day=6,
                                     train_days=25, test_days=5):
    """Generate timeseries data by feature for each cluster and save in dictionary"""

    df, site_ids = get_cluster_data(df, station_cluster, cluster_no)
    print(df.shape)
    forecast_date_ts = dt.datetime.strptime(forecast_date, "%Y-%m-%d")
    train_data_end_day = forecast_date_ts - dt.timedelta(days=test_days)

    print(train_data_end_day)

    def generate_episodes_func(df, end_date, data_episodes_num, episode_len):
        data_dict = defaultdict(list)

        for episode_end_date in pd.date_range(end=end_date, periods=data_episodes_num + 1, closed='left'):
            print("episode_end_date:", episode_end_date, "label_date:", episode_end_date + pd.DateOffset(1))

            # Add feature time series episodes per feature to result dictionary
            feature_date_range = pd.date_range(end=episode_end_date, periods=episode_len, closed='right')
            feature_df_in_range = df[df['date'].isin(feature_date_range)]
            for feature in features_list:
                # Generate feature pivot dataframe
                episode_feature_df = feature_df_in_range[['s_staID', 'date', 'hour-period', feature]]
                episode_feature_df['date_hour_period'] = episode_feature_df['date'].dt.strftime("%Y-%m-%d") + "_" + \
                                                         episode_feature_df['hour-period']
                episode_feature_pivot = episode_feature_df.pivot_table(index='s_staID', columns='date_hour_period',
                                                                       values=feature,
                                                                       fill_value=0, aggfunc=np.sum)
                data_dict[feature].append(episode_feature_pivot.values.T.tolist())

            # Add label for each time series episodes to result dictionary
            label_date_range = pd.date_range(end=episode_end_date + pd.DateOffset(1), periods=label_day_len,
                                             closed='right')
            label_df_in_range = df[df['date'].isin(label_date_range)]
            # Generate label pivot dataframe
            episode_label_df = label_df_in_range[['s_staID', 'date', 'hour-period', label_field]]
            episode_label_df['date_hour_period'] = episode_label_df['date'].dt.strftime("%Y-%m-%d") + "_" + \
                                                   episode_label_df[
                                                       'hour-period']
            episode_label_pivot = episode_label_df.pivot_table(index='s_staID', columns='date_hour_period',
                                                               values=label_field,
                                                               fill_value=0, aggfunc=np.sum)
            print("label shape:", episode_label_pivot.shape)
            data_dict['label'].append(episode_label_pivot.values.T.tolist())

        return data_dict

    data_train_dict = generate_episodes_func(df, end_date=train_data_end_day,
                                             data_episodes_num=train_days, episode_len=episode_len)
    print("generate test data")
    data_test_dict = generate_episodes_func(df, end_date=forecast_date,
                                            data_episodes_num=test_days, episode_len=episode_len)

    return data_train_dict, data_test_dict, site_ids

