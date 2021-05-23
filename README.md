Metro Interstate Traffic Volume Prediction

## Objective
1) Predict Metro Interstate Traffic Volume

**Output** :
1) This analysis has been done using random forest model. 
2) Analytical tools used in project are python and scikit learn. 
3) Model is exported in Pickle Format
4) Exploratory Analyis result saved in result folder
5) R squared is calculated with other metrics 
6) [R squared is 92.71 %] using Random forest model with ['max_depth': 14, 'n_estimators': 9]

Note: Python code is pep8 compliant

## Tools use 
> Python 3

> Main Libraries Used -
1) Scikit-learn
2) Numpy
3) Pandas
4) matplotlib

**** 

## Installing and Running

> Folders Used -
1) data - having data in csv from https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
2) model - trained and exported model


```sh
$ cd traffic_prediction_data
$ pip install -r requirements.txt
``` 

```
For Running model
```sh
$ python predict_traffic.py
```
****

## Various Steps in approach are -

1) Data is from UCI

2) Using machine learning model, a prediction model is made which will help in prediction of Traffic

3) Following features or parameters are used – 'weather_description', 'weather_main', 'hour', 'month', 'year', 'weekday', 'holiday','snow_1h', 'rain_1h', 'temp', 'clouds_all'

4) Pre-processing and normalization of data is done 

5) dataset is split into train and test with test-data having the latest 60 days (data from 2018-08-01 2018-09-30)

5) Random forest model is trained on normalized data using grid search to find best params

6) Test accuracy is calculated and model is exported