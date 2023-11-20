# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:17:20 2023

@author: EGabriele
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import statistics as sts
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess


# Pandas to read dataset, please change filepath
filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EXData.csv'
# Dayfirst and parse-dates used to properly format datetime strings in index
df = pd.read_csv(filepath, index_col = ['DateTime'], usecols=['DateTime','N2EX'], dayfirst=(True), parse_dates=[0])
df.index = pd.to_datetime(df.index)


df['Time'] = np.arange(start=0,stop=len(df))
df['Lag_1'] = df['N2EX'].shift(1)

"""
fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='N2EX', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot N2EX Pricing')
plt.show()
"""

# Training data
X = df.loc[:, ['Time']]  # features
y = df.loc[:, 'N2EX']  # target

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)
y = df.loc[:, 'N2EX']
y,X = y.align(X, join='inner')

model = LinearRegression()
model.fit(X,y)

y_pred = pd.Series(model.predict(X), index=X.index)

"""
fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('N2EX')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of N2EX')
plt.show()
"""

df2 = df.copy()


dp = DeterministicProcess(index=df2.index, constant=True, order=1, drop=True)
X = dp.in_sample()
y = df2['N2EX']
model = LinearRegression(fit_intercept=False)
model.fit(X,y)
y_pred = pd.Series(model.predict(X), index=X.index)

"""
ax = df2.plot(style=".", color="0.5", title="N2EX - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label="Trend")
plt.show()
"""

X = dp.out_of_sample(steps=30)

y_fore = pd.Series(model.predict(X), index=X.index)

ax = df2["2023-01":].plot(title="N2EX - Linear Trend Forecast")
ax = y_pred["2023-01":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()
plt.show()


