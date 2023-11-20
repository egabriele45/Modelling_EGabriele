# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:50:44 2023

@author: EGabriele
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from warnings import simplefilter
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import timedelta
import datetime

# Pandas to read dataset, please change filepath
filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EXData.csv'
# Dayfirst and parse-dates used to properly format datetime strings in index
df = pd.read_csv(filepath, index_col = ['DateTime'], usecols=['DateTime','N2EX'], dayfirst=(True), parse_dates=[0])
df.index = pd.to_datetime(df.index)

"""
for i in range(1,7):
    plt.subplot(i+320)
    _ = pd.plotting.lag_plot(df.N2EX, lag=i)
plt.show()
"""

def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)


X = make_lags(df.N2EX, lags=6)
X = X.fillna(0.0)

# Create target series and data splits
y = df.N2EX.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, shuffle=False)

# Fit and predict
model = LinearRegression()  # `fit_intercept=True` since we didn't use DeterministicProcess
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)

"""
ax = y_train.plot(label='N2EX_Train')
ax = y_test.plot(label = "N2EX_Test")
ax = y_pred.plot(ax=ax, label="N2EX_Pred")
_ = y_fore.plot(ax=ax, color='C3', label="N2EX_Forecast")
d_4 = datetime.date.today()-timedelta(days=4)
d_1 = datetime.date.today()+timedelta(days=1)
plt.xlabel("Date")
plt.ylabel("N2EX Price")
plt.xlim(d_4,d_1)
plt.title("N2EX Forecast")
plt.legend()
plt.ylim(0,200)
print(y_pred)
"""

dp = DeterministicProcess(index=df.index, constant=True, order=1, drop=True)               

X = dp.in_sample() 

y = df['N2EX']

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y_train.plot(label='N2EX_Train')
ax = y_test.plot(label = "N2EX_Test")
_ = y_fore.plot(ax=ax, color='C3', label="N2EX_Forecast")
d_4 = datetime.date.today()-timedelta(days=4)
d_1 = datetime.date.today()+timedelta(days=2)
plt.xlabel("Date")
plt.ylabel("N2EX Price")
plt.xlim(d_4,d_1)
plt.title("N2EX Forecast")
plt.legend()
plt.ylim(0,200)
print(y_fore)