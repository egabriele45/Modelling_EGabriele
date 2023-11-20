# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:08:01 2023

@author: edthe
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

filepath = r'C:\Users\edthe\OneDrive\Documents\Python Scripts\Spot Data.csv'
df = pd.read_csv(filepath, index_col= ['DateTime'],parse_dates=[0])
df.index = pd.to_datetime(df.index)
y = df.N2EX
N2EX_features = ['Total UK Generation', 'Wind Generation', 'Gas Generation', 
                 'Power Demand', 'Interconnector flow']

X = df[N2EX_features]
model = DecisionTreeRegressor(random_state=1)

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    
    return df

df = create_features(df)

for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                'Total UK Generation', 'Wind Generation', 'Gas Generation', 
                                 'Power Demand', 'Interconnector flow']
    
    TARGET = 'N2EX'
    
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]


model.fit(X_train,y_train)
score = model.score(X_train,y_train)
ypred = model.predict(X_test)
mse = mean_squared_error(y_test, ypred)

print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))

x_ax = range(len(X_test))
plt.plot(x_ax, y_test,label='original', color = 'blue')
plt.plot(x_ax, ypred, label = 'predicted',color = 'red')
plt.show()









