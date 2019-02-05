import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

import data_utils
import df_utils


def load_data(path):
    print("Path:", os.path.join(path, "station", "station_geo&econ_data_Chicago.csv"))
    station = pd.read_csv(os.path.join(path, "station", "station_geo&econ_data_Chicago.csv"))
    weather = data_utils.read_multiple_csv(os.path.join(path, "weather"))
    trip = data_utils.read_multiple_csv(os.path.join(path, "trip"))

    return station, weather, trip


def weather_process(weather):
    weather = weather.drop(
        ['tempm', 'dewptm', 'wspdm', 'wgustm', 'vism', 'pressurem',
         'windchillm', 'heatindexm', 'precipm'], axis=1)
    # remove too low temperature record
    weather = weather[weather['tempi'] > -80]
    # most heatindex, precipi, wgusti data are invalid, hense drop.
    weather = weather.drop(['heatindexi', 'wgusti'], axis=1)
    # Drop 4 records for wind speed < 0
    weather = weather[weather['wspdi'] > 0]
    # Usually windchill is important to biking activity. But only < 40% records contains valid windchill data. Hense drop column.
    weather = weather.drop(['windchilli'], axis=1)
    # One record pressurem is -9999, hense drop record. Record #: 10346 -> 10345
    weather = weather[weather['pressurei'] > -9999]
    # correct precip
    weather['precipi'] = weather['precipi'].clip(lower=0)
    weather['date'] = pd.to_datetime(weather['datetime'], infer_datetime_format=True).dt.normalize()
    weather['hour'] = pd.to_datetime(weather['datetime'], infer_datetime_format=True).dt.hour
    weather['hour-period'] = weather['hour'].apply(df_utils.time_period)

    # weather = weather.drop(['conds, wdire'], axis=1)
    weather_daily = weather.drop(['datetime', 'hour'], axis=1)
    weather_hourly = weather.drop(['datetime'], axis=1)

    period_min_max_mean_columns = ['tempi', 'dewpti', 'hum', 'wspdi', 'wdird', 'visi', 'pressurei', 'precipi']
    period_max_columns = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']
    period_min_columns = []
    period_groupby_columns = ['date', 'hour-period']

    weather_period_gb = df_utils.feature_seperate_group_by(weather_hourly, period_min_max_mean_columns,
                                                           period_max_columns,
                                                           period_min_columns, period_groupby_columns)

    daily_min_max_mean_columns = ['tempi', 'dewpti', 'hum', 'wspdi', 'wdird', 'visi', 'pressurei', 'precipi']
    daily_max_columns = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']
    daily_min_columns = []
    daily_groupby_columns = ['date']
    weather_daily_gb = df_utils.feature_seperate_group_by(weather_daily, daily_min_max_mean_columns, daily_max_columns,
                                                          daily_min_columns, daily_groupby_columns)

    # Generate rain, snow, hail, thunder, tornado last 24 hours, last 48 hours and last 72 hours
    weather_daily_gb['rain_24h'] = weather_daily_gb['rain'].shift(1)
    weather_daily_gb['snow_24h'] = weather_daily_gb['snow'].shift(1)
    weather_daily_gb['hail_24h'] = weather_daily_gb['hail'].shift(1)
    weather_daily_gb['thunder_24h'] = weather_daily_gb['thunder'].shift(1)
    weather_daily_gb['tornado_24h'] = weather_daily_gb['tornado'].shift(1)

    weather_daily_gb['rain_48h'] = weather_daily_gb['rain'].shift(2)
    weather_daily_gb['snow_48h'] = weather_daily_gb['snow'].shift(2)
    weather_daily_gb['hail_48h'] = weather_daily_gb['hail'].shift(2)
    weather_daily_gb['thunder_48h'] = weather_daily_gb['thunder'].shift(2)
    weather_daily_gb['tornado_48h'] = weather_daily_gb['tornado'].shift(2)

    weather_daily_gb['rain_72h'] = weather_daily_gb['rain'].shift(3)
    weather_daily_gb['snow_72h'] = weather_daily_gb['snow'].shift(3)
    weather_daily_gb['hail_72h'] = weather_daily_gb['hail'].shift(3)
    weather_daily_gb['thunder_72h'] = weather_daily_gb['thunder'].shift(3)
    weather_daily_gb['tornado_72h'] = weather_daily_gb['tornado'].shift(3)

    weather_daily_gb = weather_daily_gb.dropna()

    return weather_period_gb, weather_daily_gb


def get_business_days_in_df(df, data_col="date"):
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=df[data_col].min(), end=df[data_col].max())

    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    business_days = pd.DatetimeIndex(start=df[data_col].min(), end=df[data_col].max(), freq=us_bd)
    return business_days


def trip_data_process(trip_date_year):
    trip_date_year = trip_date_year.drop(['bikeid', 'trip_id', 'from_station_name', 'to_station_name'], axis=1)
    trip_date_year['usertype'] = trip_date_year['usertype'].fillna('Customer')
    # trip['User Type'].unique()
    trip_date_year['gender'] = trip_date_year['gender'].fillna('U')
    trip_date_year['birthyear'] = trip_date_year['birthyear'].fillna(trip_date_year['birthyear'].mean())

    trip_date_year['starttime_dt'] = pd.to_datetime(trip_date_year['starttime'], format="%m/%d/%Y %H:%M:%S",
                                                    infer_datetime_format=True)

    trip_date_year['date'] = pd.to_datetime(trip_date_year['starttime_dt'], infer_datetime_format=True).dt.normalize()
    trip_date_year['hour'] = pd.to_datetime(trip_date_year['starttime_dt'], infer_datetime_format=True).dt.hour
    trip_date_year['hour-period'] = trip_date_year['hour'].apply(df_utils.time_period)
    trip_date_year['week of year'] = pd.to_datetime(trip_date_year['date']).dt.week
    trip_date_year['weekday'] = pd.to_datetime(trip_date_year['date']).dt.weekday
    trip_date_year['weekday_hour_period'] = trip_date_year['weekday'].apply(str) + "_" + trip_date_year['hour-period']

    business_days = get_business_days_in_df(trip_date_year)
    trip_date_year['business_day'] = trip_date_year['date'].isin(business_days)
    return trip_date_year


def trip_groupby_aproach(trip_station, mode=6):
    """
       mode 1: daily whole city,                              2:daily per start_station,
            3: daily per start per end station,               4: hourly per start_station
            5: hour-period per start_station per end station  6: hour-period per start_station
    """

    mode = 6

    if mode == 1:
        group_by_time_range = ['date']
        group_by_field = group_by_time_range
        date_fields = group_by_time_range + ['week of year', 'weekday', 'business_day']
        start_station_fields = ['s_staID', 's_tripCt', 's_ctPopn', 's_avgMedianIncome', 's_numBiz', 's_numBSS',
                                's_totalCap', 's_numSchool', 's_numPark', 's_totalParkArea']
        end_station_fields = ['t_staID', 't_tripCt', 't_ctPopn', 't_avgMedianIncome', 't_numBiz', 't_numBSS',
                              't_totalCap', 't_numSchool', 't_numPark', 't_totalParkArea']
        model_data_fields = date_fields
    elif mode == 2:
        group_by_time_range = ['date']
        group_by_field = group_by_time_range + ['s_staID']
        date_fields = group_by_time_range + ['week of year', 'weekday', 'business_day']
        start_station_fields = ['s_staID', 's_tripCt', 's_ctPopn', 's_avgMedianIncome', 's_numBiz', 's_numBSS',
                                's_totalCap', 's_numSchool', 's_numPark', 's_totalParkArea']
        end_station_fields = ['t_staID', 't_tripCt', 't_ctPopn', 't_avgMedianIncome', 't_numBiz', 't_numBSS',
                              't_totalCap', 't_numSchool', 't_numPark', 't_totalParkArea']
        model_data_fields = date_fields + start_station_fields
    elif mode == 3:
        group_by_time_range = ['date']
        group_by_field = group_by_time_range
        date_fields = group_by_time_range + ['week of year', 'weekday', 'business_day']
        start_station_fields = ['s_staID', 's_tripCt', 's_ctPopn', 's_avgMedianIncome', 's_numBiz', 's_numBSS',
                                's_totalCap', 's_numSchool', 's_numPark', 's_totalParkArea']
        end_station_fields = ['t_staID', 't_tripCt', 't_ctPopn', 't_avgMedianIncome', 't_numBiz', 't_numBSS',
                              't_totalCap', 't_numSchool', 't_numPark', 't_totalParkArea']
        model_data_fields = date_fields + start_station_fields + end_station_fields
    elif mode == 4:
        group_by_time_range = ['date', 'hour']
        group_by_field = group_by_time_range
        date_fields = group_by_time_range + ['week of year', 'weekday', 'business_day']
        start_station_fields = ['s_staID', 's_tripCt', 's_ctPopn', 's_avgMedianIncome', 's_numBiz', 's_numBSS',
                                's_totalCap', 's_numSchool', 's_numPark', 's_totalParkArea']
        end_station_fields = ['t_staID', 't_tripCt', 't_ctPopn', 't_avgMedianIncome', 't_numBiz', 't_numBSS',
                              't_totalCap', 't_numSchool', 't_numPark', 't_totalParkArea']
        model_data_fields = date_fields + start_station_fields
    elif mode == 5:
        group_by_time_range = ['date', 'hour-period']
        group_by_field = ['s_staID'] + ['t_staID'] + group_by_time_range
        date_fields = group_by_field + ['week of year', 'weekday', 'business_day']
        start_station_fields = ['s_tripCt', 's_ctPopn', 's_avgMedianIncome', 's_numBiz', 's_numBSS', 's_totalCap',
                                's_numSchool', 's_numPark', 's_totalParkArea']
        end_station_fields = ['t_tripCt', 't_ctPopn', 't_avgMedianIncome', 't_numBiz', 't_numBSS', 't_totalCap',
                              't_numSchool', 't_numPark', 't_totalParkArea']
        model_data_fields = date_fields + start_station_fields + + end_station_fields
    elif mode == 6:
        group_by_time_range = ['hour-period', 'date']
        group_by_field = group_by_time_range + ['s_staID']
        date_fields = group_by_field + ['week of year', 'weekday', 'business_day']
        start_station_fields = ['s_tripCt', 's_ctPopn', 's_avgMedianIncome', 's_numBiz', 's_numBSS', 's_totalCap',
                                's_numSchool', 's_numPark', 's_totalParkArea']
        end_station_fields = ['t_staID', 't_tripCt', 't_ctPopn', 't_avgMedianIncome', 't_numBiz', 't_numBSS',
                              't_totalCap', 't_numSchool', 't_numPark', 't_totalParkArea']
        model_data_fields = date_fields + start_station_fields

    trip_station['date-hour-period'] = trip_station['date'].dt.strftime("%Y-%m-%d") + "_" + trip_station['hour-period']
    trip_info_gb = trip_station[['s_staID', 'date-hour-period', 'starttime']].pivot_table(index=['s_staID'],
                                                                                          columns='date-hour-period',
                                                                                          values='starttime',
                                                                                          fill_value=0,
                                                                                          aggfunc='count').stack().to_frame()
    trip_info_gb.columns = ['trip count']
    return trip_info_gb


def merge_trip_station_weather(trip_date_year, station, weather_period_gb):
    trip_station = trip_date_year.merge(station, left_on='from_station_id', right_on='staID', how='inner')

    trip_station = trip_station.drop('starttime_dt', axis=1)

    trip_station.columns = ['starttime', 'stoptime', 'tripduration', 'from_station_id',
                            'to_station_id', 'usertype', 'gender', 'birthyear', 'date', 'hour',
                            'hour-period', 'week of year', 'weekday', 'weekday_hour_period',
                            'business_day', 's_staID', 's_tripCt', 's_ctPopn', 's_avgMedianIncome',
                            's_numBiz', 's_numBSS', 's_totalCap', 's_numSchool','s_numPark',
                            's_totalParkArea']

    trip_station = trip_station.merge(station, left_on='to_station_id', right_on='staID', how='inner')

    trip_station.columns = ['starttime', 'stoptime', 'tripduration', 'from_station_id',
                            'to_station_id', 'usertype', 'gender', 'birthyear', 'date', 'hour',
                            'hour-period', 'week of year', 'weekday', 'weekday_hour_period',
                            'business_day', 's_staID', 's_tripCt', 's_ctPopn', 's_avgMedianIncome',
                            's_numBiz', 's_numBSS', 's_totalCap', 's_numSchool', 's_numPark',
                            's_totalParkArea', 't_staID', 't_tripCt', 't_ctPopn', 't_avgMedianIncome',
                            't_numBiz', 't_numBSS', 't_totalCap', 't_numSchool', 't_numPark',
                            't_totalParkArea']

    trip_info_gb = trip_groupby_aproach(trip_station, mode=6)

    trip_info_gb = trip_info_gb.reset_index()
    trip_info_gb[['date', 'hour-period']] = trip_info_gb['date-hour-period'].str.split("_", expand=True)

    trip_info_gb['date'] = pd.to_datetime(trip_info_gb['date'], format="%Y-$m-%d", infer_datetime_format=True)
    trip_info_gb['week of year'] = pd.to_datetime(trip_info_gb['date']).dt.week
    trip_info_gb['weekday'] = pd.to_datetime(trip_info_gb['date']).dt.weekday

    business_days = get_business_days_in_df(trip_info_gb)
    trip_info_gb['business_day'] = trip_info_gb['date'].isin(business_days)

    data_utils.check_df_na(trip_info_gb)

    weather_period_gb = weather_period_gb.reset_index()
    trip_info_gb = trip_info_gb.reset_index()

    data = trip_info_gb[['date', 'hour-period', 's_staID', 'week of year', 'weekday',
                         'business_day', 'trip count']].merge(weather_period_gb, how='left',
                                                              on=['date', 'hour-period'])

    data = df_utils.add_dummmies_remove_original_col(data, ['hour-period'], remove_original_col=True)
    data = data.fillna(method='bfill')

    return data