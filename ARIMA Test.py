# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:43:27 2023

@author: edthe
"""

from pmdarima.model_selection import train_test_split
import pmdarima as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EXData.csv'
df = pd.read_csv(filepath, index_col=['DateTime'], usecols=['DateTime','N2EX'],parse_dates=[0], dayfirst=True)
df.index = pd.to_datetime(df.index)

train, test = train_test_split(df,train_size=17472)
model = pm.auto_arima(train, seasonal=True, m=12)
forecasts = model.predict(test.shape[0])
x = np.arange(df.shape[0])

plt.plot(x[:17472], train, c='blue')
plt.plot(x[17472:], forecasts, c='green')
plt.show()




