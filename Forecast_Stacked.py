# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:13:44 2023

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
from xgboost import XGBRegressor

# Pandas to read dataset, please change filepath
filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EXData.csv'
# Dayfirst and parse-dates used to properly format datetime strings in index
df = pd.read_csv(filepath, index_col = ['DateTime'], usecols=['DateTime','N2EX'], dayfirst=(True), parse_dates=[0])
df.index = pd.to_datetime(df.index)

y = df.copy()

# Create trend features
dp = DeterministicProcess(
    index=y.index,  # dates from the training data
    constant=True,  # the intercept
    order=2,        # quadratic trend
    drop=True,      # drop terms to avoid collinearity
)
X = dp.in_sample()  # features for the training data

"""
# Test on the years 2016-2019. It will be easier for us later if we
# split the date index instead of the dataframe directly.
idx_train, idx_test = train_test_split(
    y.index, test_size=12 * 4, shuffle=False,
)
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

# Fit trend model
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

# Make predictions
y_fit = pd.DataFrame(
    model.predict(X_train),
    index=y_train.index,
    columns=y_train.columns,
)
y_pred = pd.DataFrame(
    model.predict(X_test),
    index=y_test.index,
    columns=y_test.columns,
)

d_4 = datetime.date.today()-timedelta(days=4)
d_1 = datetime.date.today()+timedelta(days=2)


# Plot
axs = y_train.plot(label='Y_train',color='0.25', subplots=True, sharex=True)
axs = y_test.plot(label = 'Y_test', color='0.25', subplots=True, sharex=True, ax=axs)
axs = y_fit.plot(label = 'Y_fit',color='C0', subplots=True, sharex=True, ax=axs)
axs = y_pred.plot(label = 'Y_pred', color='C3', subplots=True, sharex=True, ax=axs)
plt.legend()
plt.xlim(d_4,d_1)
plt.ylim(0,200)
for ax in axs: ax.legend([])
_ = plt.suptitle("Trends")
"""
idx_train, idx_test = train_test_split(
    y.index, test_size=12 * 4, shuffle=False,)
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

# Make predictions
y_fit = pd.DataFrame(
    model.predict(X_train),
    index=y_train.index,
    columns=y_train.columns,
)
y_pred = pd.DataFrame(
    model.predict(X_test),
    index=y_test.index,
    columns=y_test.columns,
)

# Pivot wide to long (stack) and convert DataFrame to Series (squeeze)
y_fit = y_fit.stack().squeeze()    # trend from training set
y_pred = y_pred.stack().squeeze()  # trend from test set

# Create residuals (the collection of detrended series) from the training set
y_resid = y_train - y_fit

# Train XGBoost on the residuals
xgb = XGBRegressor()
xgb.fit(X_train, y_resid)

# Add the predicted residuals onto the predicted trends
y_fit_boosted = xgb.predict(X_train) + y_fit
y_pred_boosted = xgb.predict(X_test) + y_pred