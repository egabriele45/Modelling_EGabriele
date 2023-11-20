# -*- coding: utf-8 -*-

"""
Created on Thu Aug 31 15:28:13 2023

@author: edthe
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np

filepath = r'C:\Users\edthe\OneDrive\Documents\Python Scripts\Spot Data.csv'
df = pd.read_csv(filepath, index_col= ['DateTime'],parse_dates=[0])
df.index = pd.to_datetime(df.index)

y = df['N2EX']

df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
    
features = ['Total UK Generation', 'Wind Generation', 'Gas Generation', 
                 'Power Demand', 'Interconnector flow','hour','dayofweek','month']


X = df[features]
model = DecisionTreeRegressor(random_state=1)
model.fit(X,y)
"""
ypred = model.predict(X)
print(mean_squared_error(y, ypred)) 


plt.figure(figsize=(15,6))
plt.plot(df.index, y,label='original', color = 'red')
plt.plot(df.index, ypred, label = 'predicted', color = 'blue')
plt.legend()
plt.show()
"""
pred = []
y_pred = model.predict(X)
pred.append(y_pred)
future = pd.date_range('2023-01-01','2023-01-06', freq='1h')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False
df_and_future = pd.concat([df, future_df])
df_and_future[features] = df[features]
#df_and_future = create_features(df_and_future)
#df_and_future = add_lags(df_and_future)

future_w_features = df_and_future.query('isFuture').copy()

future_w_features['pred'] = model.predict(future_w_features[features])

future_w_features['pred'].plot(figsize=(10, 5),color='blue',
                               ms=1,
                               lw=1,
                               title='Future Predictions')
plt.show()   
