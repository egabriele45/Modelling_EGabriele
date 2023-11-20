# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:42:16 2023

@author: edthe
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path  


filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EXData.csv'

df = pd.read_csv(filepath, parse_dates=[0], index_col=['DateTime'], dayfirst=True)
df.index = pd.to_datetime(df.index)                  
y = df.N2EX
del df['Peak/Off Peak']
del df['Weekday']


df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df = df.astype('float')
df1 = df
features = ['hour','dayofweek','month','UK D+1 Power Demand','UK D+1 Wind Forecast','Solar Forecast D+1','UK D+1 Generation',	'Forecast Imbalance Volume',	'French Forecast Gen D+1', 'WA Price Last Trading Day Average','DA Temp Forecast','DA Forecast Gas Supply','DA Forecast Gas Demand','DA Forecast Gas Sup/Dem Spread','DA CCGT SRMC']
X = df[features]


X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2)

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

def add_lags(df):
    target_map = df.N2EX.to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    return df

start_date = df.index.max()+timedelta(days=1)
end_date = start_date + relativedelta(months=1)

future = pd.date_range(start_date,end_date,freq='30T')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False

df_and_future = pd.concat([df, future_df])
df_and_future = create_features(df_and_future)
df_and_future = add_lags(df_and_future)

model1 = XGBRegressor(n_estimators=1000, learning_rate = 0.05)
model1.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_valid,y_valid)],verbose=100)

future_w_features = df_and_future.query('isFuture').copy()
future_w_features['pred'] = model1.predict(future_w_features[features])
future_w_features['pred'].plot(figsize = (10,5),ms=1,lw=1)
xticks = pd.date_range(start_date,end_date,freq='7D')
plt.xticks(xticks)
plt.ylabel("DA-Price (£/MWh)")
plt.title("Forecast DA Price (£/MWh)")
plt.show()

end_date = end_date.strftime('%Y-%m-%d')
path_forecast = Path(rf'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EX_Forecast_{end_date}.csv')
future_w_features['pred'].to_csv(path_or_buf=path_forecast)

for est in range(500,3001,500):
    model = XGBRegressor(n_estimators=est, learning_rate = 0.05)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],verbose=False)
    pred = model.predict(X_valid)
    print("MAE for %d:" %est, str(mean_absolute_error(pred,y_valid)))

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy='mean')),
                              ('model',XGBRegressor(n_estimators=500))])

scores = -1 * cross_val_score(my_pipeline, X, y, cv=2, scoring='neg_mean_absolute_error')
print(scores,"\n",scores.mean())


                          