# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:21:23 2023

@author: edthe
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EXData.csv'
df = pd.read_csv(filepath, parse_dates=[0], index_col=['DateTime'], dayfirst=True)
df.index = pd.to_datetime(df.index) 
df = df.loc[:, df.columns!=('Peak/Off Peak')] 
df = df.loc[:, df.columns!=('Weekday')] 

y = df.N2EX



df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df = df.astype('float')
df1 = df
features = ['hour','dayofweek','month','Block', 'UK D+1 Power Demand',
       'UK D+1 Wind Forecast', 'Solar Forecast D+1', 'UK D+1 Generation',
       'Forecast Imbalance Volume', 'French Forecast Gen D+1',
       'WA Price Last Trading Day Average', 'DA Temp Forecast',
       'DA Forecast Gas Supply', 'DA Forecast Gas Demand',
       'DA Forecast Gas Sup/Dem Spread', 'DA CCGT SRMC']
X = df[features]
#X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2)

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy='constant')),
                              ('model',RandomForestRegressor(n_estimators=100,
                                                             random_state=0))])

scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores :\n", scores)
print(scores.mean())






                              