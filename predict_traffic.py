# Import necessary libraries
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import date

# holiday feature


def modify_holiday(x):
    if x == 'None':
        return False
    else:
        return True

# rain feature


def modify_rain_1h(x):
    if x == 0:
        return 'no_rain'
    elif x > 0 and x < 100:
        return 'light'
    elif x > 100 and x < 1000:
        return 'moderate'
    elif x > 1000:
        return 'heavy'

# snow feature


def modify_snow_1h(x):
    if x == 0:
        return 'no_snow'
    else:
        return 'snow'

# Feature engineering and Data cleaning


def preprocessing(df):

    # convert kelvin to celsius
    df['temp'] = (df['temp'] - 273.15)

    # Outlier in temp and rain which was detected earlier needs to be removed
    df = df.loc[df.temp > -240]
    df = df.loc[df.rain_1h < 3800]

    # Extracting features from date_time variable
    df['date_time'] = pd.to_datetime(df.date_time)
    df['hour'] = df.date_time.dt.hour
    df['month'] = df.date_time.dt.month
    df['year'] = df.date_time.dt.year
    df['weekday'] = df.date_time.dt.weekday  # Monday is 0
    df['date'] = df.date_time.dt.date

    # categorizing holiday as True and not Holiday as False
    df['holiday'] = df['holiday'].map(
        modify_holiday)

    # categorizing Rain as no rain, linght, moderate and heavy
    df['rain_1h'] = df['rain_1h'].map(
        modify_rain_1h)

    # categorizing Snow as no snow and not snow
    df['snow_1h'] = df['snow_1h'].map(
        modify_snow_1h)

    # encoding weather information as label 0,1,2,3...
    le = LabelEncoder()
    df['weather_main'] = le.fit_transform(
        df['weather_main'])
    df['weather_description'] = le.fit_transform(
        df['weather_description'])

    return df


def exploratory_analyis(df):

    # check if null values - here no null values
    print(df.info())

    # check outlier and other details - here can find temp(min) and
    # rain_1h(max) has outlier
    print(df.describe())
    print(df.describe(include='object'))
    cols = ['clouds_all', 'rain_1h', 'snow_1h', 'temp', 'traffic_volume']
    # sns.pairplot(df[cols])
    # plt.show()

    # correlation between traffic and other variables
    # sns.heatmap(df.corr(), annot=True)
    # plt.show() # plot show no correlation

    return df


def data_preprocessor():
    cat_vars = [
        'weather_description',
        'weather_main',
        'hour',
        'month',
        'year',
        'weekday',
        'holiday',
        'snow_1h',
        'rain_1h']

    num_vars = ['temp', 'clouds_all']

    # Creating pipeline to transform data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('oneHot', OneHotEncoder())])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_vars),
        ('cat', categorical_transformer, cat_vars)])

    return preprocessor


def main():
    # Import dataset - csv to dataframe
    df_traffic_data = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')
    df_traffic_data = exploratory_analyis(df_traffic_data)
    df_traffic_features = preprocessing(df_traffic_data)

    # test-data having the latest 60 days
    # split data in traing and test data
    # as max date is 2018-09-30 so data is splitted at
    data_before = df_traffic_features[(df_traffic_features.date_time < pd.to_datetime(
        date(2018, 8, 1)))].reset_index(drop=True)
    test_index_start = data_before.shape[0]

    df_traffic_features.set_index('date', inplace=True)
    df_traffic_transformed = data_preprocessor(
    ).fit_transform(df_traffic_features).toarray()

    x_train = df_traffic_transformed[:test_index_start]
    x_test = df_traffic_transformed[test_index_start:]
    y_train = df_traffic_features.traffic_volume[:test_index_start]
    y_test = df_traffic_features.traffic_volume[test_index_start:]

    model = RandomForestRegressor()

    parameter = {
        'n_estimators': np.arange(
            1, 10), 'max_depth': np.arange(
            1, 15)}
    grid_search = GridSearchCV(model, parameter, cv=3)
    grid_search.fit(x_train, y_train)

    print(grid_search.best_params_)

    model = RandomForestRegressor(
        n_estimators=grid_search.best_params_['n_estimators'],
        max_depth=grid_search.best_params_['max_depth'])
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('R2: ', round(metrics.r2_score(y_test, pred) * 100, 2), '%')
    print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, pred))
    print('Root Mean Squared Error: ', np.sqrt(
        metrics.mean_squared_error(y_test, pred)))

    with open('model/regression_model.pkl', 'wb') as fid:
        pickle.dump(model, fid)


if __name__ == "__main__":
    main()
