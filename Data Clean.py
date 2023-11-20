# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:04:44 2023

@author: edthe
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

filepath = r'C:\Users\edthe\OneDrive\Documents\Python Scripts\SpotData.csv'
df = pd.read_csv(filepath, parse_dates=[0], index_col=['DateTime'], dayfirst=True)
df.index = pd.to_datetime(df.index)
                          
y = df.N2EX

df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df = df.astype('float')
df1 = df
features = ['hour','dayofweek','month','Total UK Generation','Wind Generation','Gas Generation','Power Demand','Interconnector flow','SIP','APX']
X = df[features]
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2)

"Removing colums in X_train that have no values"

X_tr = X_train.copy()
X_va = X_valid.copy()

cols_with_miss = [col for col in X_tr.columns if X_tr[col].isnull().any()]
print("Columns removed:",cols_with_miss)

reduced_X_train = X_train.drop(cols_with_miss, axis=1)
reduced_X_valid = X_valid.drop(cols_with_miss, axis=1)

model = RandomForestRegressor(n_estimators = 100, random_state=0)
model.fit(reduced_X_train,y_train)
preds = model.predict(reduced_X_valid)
mae = mean_absolute_error(y_valid, preds)
print("Mean Absolute Error with removing columns: ",mae)

"Imputation, so adding mean values to missing values in X_train columns"

X_t = X_train.copy()
X_v = X_valid.copy()

imp = SimpleImputer(strategy='constant')
imp_X_train = pd.DataFrame(imp.fit_transform(X_t))
imp_X_valid = pd.DataFrame(imp.transform(X_v))

imp_X_train.columns = X_t.columns
imp_X_valid.columns = X_v.columns

model = RandomForestRegressor(n_estimators = 100, random_state=0)
model.fit(imp_X_train,y_train)
preds = model.predict(imp_X_valid)
mae = mean_absolute_error(y_valid, preds)
print("Mean absolute error for imputing data: ",mae)

"Imputation plus applies mean value to missing values in X_train columns and creates a extra column with True/False to say if a value was added in or not."""

X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

my_imputer = SimpleImputer(strategy='constant')
imp_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imp_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

imp_X_train_plus.columns = X_train_plus.columns
imp_X_valid_plus.columns = X_valid_plus.columns

model = RandomForestRegressor(n_estimators = 100, random_state=0)
model.fit(imp_X_train_plus,y_train)
preds = model.predict(imp_X_valid_plus)
mae = mean_absolute_error(y_valid, preds)
print("Mean absolute error for imputing plus creating True/False column: ",mae)
