# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:28:37 2023

@author: EGabriele
"""

import pandas as pd
import numpy as np

# Pandas to read dataset, please change filepath
filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EXData.csv'
# Dayfirst and parse-dates used to properly format datetime strings in index
df = pd.read_csv(filepath, index_col = ['DateTime'], usecols=['DateTime','N2EX'], dayfirst=(True), parse_dates=[0])
df.index = pd.to_datetime(df.index)

print(df)
df = df.resample('15min').mean()
df = df.fillna(method='bfill')
print(df)