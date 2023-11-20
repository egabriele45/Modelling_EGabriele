# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# 1. Reading and exploring dataset
import pandas as pd
filepath1 ='xxx.csv'
df = pd.read_csv(filepath1)
#Info function just lets us know how many entries there are and num of columns
df.info()
#Plots the data
df.plot()
import numpy as np
#Log of variable df, not needed
df = np.log(df)
df.plot()
#To get rid of Log of df and put back to original
df = np.exp(df)
#Reading all rows except for bottom 30
msk = (df.index < len(df)-30)
df_train = df[msk].copy()
# '~' operator is a bitwise operator, 'NOT', so inverts msk. 'NOT' like 'AND,'OR', 'XOR'.
df_test = df[~msk].copy()

# 2. Check data for stationarity
# ACF is autocorrelation plot, PACF is partial autocorrelation plot, it looks at difference values
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#If ACF gradually decreases and goes to 0 then data is not stationary so AR model fine
# If PACF gradually decreases and goes to 0 then data is not stationary so MA model is fine
acf_original = plot_acf(df_train)
pacf_original = plot_pacf(df_train)

# 3. ADF Test
from statsmodels.tsa.stattools import adfuller
#Augmented-Dickey Fuller test
adf_test = adfuller(df_train)
#[1] signifies looking at first difference of time series
# p-Value will tell us if the series is stationary or not.
# If large this will mean reject H0 so series is non stationary
print(f'p-value: {adf_test[1]}')
#Dropna function removes rows that contain NULL values
#diff() function differences data
df_train_diff = df_train.diff().dropna()
#plot differenced data
df_train_diff.plot()
#plot ACF and PACF of differenced data
acf_diff = plot_acf(df_train_diff)
pacf_diff = plot_pacf(df_train_diff)
#Perform ADFuller test for stationarity
#If p-Value is smaller and below Confidence level then accpet H0, so Stationary
adf_test = adfuller(df_train_diff)
print(f'p-value: {adf_test[1]}')
# Depending on which difference is stationary, if first, set d = 1 in (p,d,q)

# 4. Setting p and q values in ARIMA(p,d,q)
# If PACF plot for differenced[1] has spike at lag[p],but not beyond.
# And ACF plot decays more gradually, this suggests ARIMA(p,d,0)
# If ACF has spike at lag[q] but not beyond.
# And PACF plot decays more gradually then this suggests  ARIMA(0,d,q).

# 5. For ARIMA Model
from statsmodels.tsa.arima.model import ARIMA
#Use p,d,q values obtained in previous part
model = ARIMA(df_train, order=(p,d,q))
model_fit = model.fit()
#Summary functions gives use data about tests, kurtosis, etc.
print(model_fit.summary())

# 7. Check Stationarity
import matplotlib.pyplot as plt
#Look at residuals plot, if residuals plot is like white-noise then model is good
residuals = model_fit.resid[1:]
# Subplot function describes plots, (num rows, num columns, num of plots)
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()
# ACF and PACF plot of residuals
acf_res = plot_acf(residuals)
pacf_res = plot_pacf(residuals)

# 8. Make Time Series predictions
forecast_test = model_fit.forecast(len(df_test))
df['forecast_manual'] = [None]*len(df_train) + list(forecast_test)
df.plot()

# 9. Alternative, Auto-fit ARIMA model in Python
import pmdarima as pm
# pmdarima module computes and fits ARIMA model automatically
# This uses KPSS unit root rest to identify value of d
# Uses AIC information criteria to determine values of p and q
auto_arima = pm.auto_arima(df_train, stepwise=False, seasonal=False)
# Summary function will tell you about p,d,q
auto_arima.summary()
# Uses Auto_ARIMA fitted model
forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
df['forecast_auto'] = [None]*len(df_train) + list(forecast_test_auto)
df.plot()

# 10. Testing how good each model is by looking at errors
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

mae = mean_absolute_error(df_test, forecast_test)
mape = mean_absolute_percentage_error(df_test, forecast_test)
rmse = np.sqrt(mean_squared_error(df_test, forecast_test))

print(f'mae - manual: {mae}')
print(f'mape - manual: {mape}')
print(f'rmse - manual: {rmse}')

mae = mean_absolute_error(df_test, forecast_test_auto)
mape = mean_absolute_percentage_error(df_test, forecast_test_auto)
rmse = np.sqrt(mean_squared_error(df_test, forecast_test_auto))

print(f'mae - auto: {mae}')
print(f'mape - auto: {mape}')
print(f'rmse - auto: {rmse}')