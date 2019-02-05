import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def get_cluster_data(data, station_clustered, cluster_no, sites_list=None):
    station_picked = station_clustered[station_clustered["cluster"] == cluster_no]

    station_ids = station_picked['staID'].unique()

    data_picked = data[data['s_staID'].isin(station_ids)]
    data_picked = data_picked.drop("s_staID", axis=1)
    return data_picked, station_ids


def get_train_test_data(data_train, data_test):
    X_train = data_train.drop(['trip count'], axis=1)
    y_train = data_train[['trip count']]

    X_test = data_test.drop(['trip count'], axis=1)
    y_test = data_test[['trip count']]

    return X_train, X_test, y_train, y_test


def standardize_train_test(X_train, X_test):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_train_scale = pd.DataFrame(X_train_scale)
    X_train_scale.columns = X_train.columns

    X_test_scale = scaler.transform(X_test)
    X_test_scale = pd.DataFrame(X_test_scale)
    X_test_scale.columns = X_train.columns

    return X_train_scale, X_test_scale


# def forecast_per_cluster(data_train, data_test):
#     X_train, X_test, y_train, y_test = get_train_test_data(data_train, data_test)
#     # X_train_scale, X_test_scale = standardize_train_test(X_train, X_test)
#
#     # XGBoost test
#     model = xgb.XGBRegressor(n_estimators=500, max_depth=8,
#                              learning_rate=0.01, subsample=0.8, colsample_bylevel=0.8, seed=0,
#                              gamma=0.2, nthread=10)
#
#     model.fit(X_train, y_train)
#     y_pred = pd.DataFrame(model.predict(X_test))
#
#     print(r2_score(y_test, y_pred))
#     # y_test['pred'] = y_pred
#
#     return y_pred


def forecast_per_cluster(data_train, data_test):
    X_train, X_test, y_train, y_test = get_train_test_data(data_train, data_test)
#     X_train_scale, X_test_scale = standardization(X_train, X_test)

    # XGBoost test
    import xgboost as xgb
    model = xgb.XGBRegressor(n_estimators=500, max_depth=8,
                                learning_rate=0.01, subsample=0.8, colsample_bylevel=0.8, seed=0,
                                 gamma=0.2, nthread= 10)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(r2_score(y_test,y_pred))
    y_test['pred'] = y_pred

    return y_test


# def whole_data_forecast(data_train, data_test, station_clustered, sites_list):
#     cluster_cnt = len(station_clustered['cluster'].unique())
#     # whole_pred = pd.DataFrame()
#     whole_pred = []
#     data_train_scale, data_test_scale = standardize_train_test(data_train, data_test)
#
#     for cluster_no in range(cluster_cnt):
#         print("cluster_no:", cluster_no)
#         #         data_train_picked, station_ids = get_cluster_data(data_train_scale, station_clustered, cluster_no)
#         #         data_test_picked, _ = get_cluster_data(data_test_scale, station_clustered, cluster_no)
#
#         data_train_picked, station_ids = get_cluster_data(data_train, station_clustered, cluster_no, sites_list)
#         data_test_picked, _ = get_cluster_data(data_test, station_clustered, cluster_no, sites_list)
#         print("station_no:", len(station_ids))
#         y_test_pred = forecast_per_cluster(data_train_picked, data_test_picked)
#         # whole_pred = pd.concat([y_test_picked, whole_pred])
#         whole_pred.extend(y_test_pred)
#
#     return whole_pred


def whole_data_forecast(data_train, data_test, station_clustered, sites_list):
    cluster_cnt = len(station_clustered['cluster'].unique())
    whole_pred = pd.DataFrame()

    # data_train_scale, data_test_scale = standardize_train_test(data_train, data_test)

    for cluster_no in range(cluster_cnt):
        print("cluster_no:", cluster_no)
        #         data_train_picked, station_ids = get_cluster_data(data_train_scale, station_clustered, cluster_no)
        #         data_test_picked, _ = get_cluster_data(data_test_scale, station_clustered, cluster_no)

        data_train_picked, station_ids = get_cluster_data(data_train, station_clustered, cluster_no, sites_list)
        data_test_picked, _ = get_cluster_data(data_test, station_clustered, cluster_no, sites_list)
        print("station_no:", len(station_ids))
        y_test_picked = forecast_per_cluster(data_train_picked, data_test_picked)
        whole_pred = pd.concat([y_test_picked, whole_pred])

    return whole_pred

