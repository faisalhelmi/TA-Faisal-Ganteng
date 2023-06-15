import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.ndimage import shift
from math import sqrt
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from fastapi import FastAPI, BackgroundTasks

import glob
import os

files = os.path.join(r"C:\Users\faisa\Desktop\PM Data Jogja 2021", "psi-jogja-*.csv")
files = glob.glob(files)
pmbeforepreproc = pd.concat(map(pd.read_csv,files))

cols25 = ['PM10', 'SO2', 'CO', 'O3', 'NO2', 'Max', 'Critical Component', 'Category']
pm25 = pmbeforepreproc
pm25.drop(cols25, axis=1, inplace=True)

pm25['Date'] = pd.to_datetime(pm25['Date'] + ' ' + pm25['Time'])
del pm25['Time']

pm25 = pm25.sort_values('Date')

pm25 = pm25.ffill()

pm25.replace(0, np.nan, inplace=True)

pm25deletenan = pm25.dropna()

def isSeriesStationary(series):
    pValue = adfuller(series)[1]
    if pValue > 0.05:
        return False
    else:
        return True
def isSeriesStationaryAvg(series, delta = 2):
    split = int(len(series)/2)
    split1, split2 = series[:split], series[split:]
    avg1, avg2 = split1.mean(), split2.mean()
    var1, var2 = split1.var(), split2.var()
    if abs(avg1 - avg2) > delta or abs(var1 - var2) > delta**2:
        return False
    else:
        return True
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

isSeriesStationary(pm25deletenan["PM2.5"].values)

isSeriesStationaryAvg(pm25deletenan["PM2.5"].values)

def splitTrainTest(series, testSplit):
    totalData = len(series)
    trainSplit = int(totalData * (1 - testSplit))
    trainSet = series[:trainSplit]
    testSet = series[trainSplit:]
    return trainSet, testSet
trainSet, testSet = splitTrainTest(pm25deletenan["PM2.5"].values, 0.1)
differencedTrainSet = difference(trainSet, 365)
model = ARIMA(differencedTrainSet, order=(7,0,1))
"""Fit model with non constant trend and no displacement"""
model_fit = model.fit()
forecast = model_fit.predict(len(differencedTrainSet), len(differencedTrainSet) + len(testSet))

cred = credentials.Certificate("project-iot-pm-monitoring-firebase-adminsdk-bdbds-75ba00f05b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://project-iot-pm-monitoring-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your database URL
})

app = FastAPI()

def isSeriesStationary(series):
    pValue = adfuller(series)[1]
    if pValue > 0.05:
        return False
    else:
        return True

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

def inverse_difference(history, yhat, interval=1):
    print('yhat')
    print(yhat)
    print('history')
    print(history[-interval])
    return yhat + history[-interval]

def predict_pm25(data):
    # Preprocess the data
    data = data.ffill()
    data.replace(0, np.nan, inplace=True)
    data.dropna(inplace=True)

    # Calculate the length of a day in the data
    day_length = (data.index[1] - data.index[0]).days

    # Perform the prediction
    differenced_train_set = difference(data.values, day_length)
    forecast = model_fit.predict(
        len(differenced_train_set), len(differenced_train_set) + 11  # Forecasting for the next 12 hours
    )
    forecast = inverse_difference(data.values, forecast, day_length)

    # Convert the forecast to a pandas Series with a datetime index
    forecast_series = pd.Series(forecast[:12], index=data.index[-12:])  # Selecting the first 12 hours

    # Convert the forecast Series to a dictionary with string keys
    forecast_dict = {str(key): value for key, value in forecast_series.items()}

    return forecast_dict


@app.post("/predict/6k11jAY1UGXrX6LciXf8oSHddFX2")
async def predict(user_id: str, background_tasks: BackgroundTasks):
    # Fetch data from Firebase Realtime Database using the user_id
    data_ref = db.reference(f"UsersData/6k11jAY1UGXrX6LciXf8oSHddFX2/pm25")
    data = data_ref.get()

    # Convert the data into a pandas DataFrame
    pm25_df = pd.DataFrame.from_dict(data, orient='index', columns=['PM2.5'])
    pm25_df.index = pd.to_datetime(pm25_df.index)
    # Perform the prediction
    forecast = predict_pm25(pm25_df)
    print(forecast)

    # Convert the forecast to a dictionary
    #forecast_dict = forecast.to_dict()
    forecast_dict = forecast

    # Update the Firebase Realtime Database with the forecasted values
    forecast_ref = db.reference('/UsersData/6k11jAY1UGXrX6LciXf8oSHddFX2/pm25pred')
    forecast_ref.set(forecast_dict)

    return {"message": "Prediction stored successfully!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)