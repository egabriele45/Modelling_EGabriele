# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:56:19 2023

@author: edthe
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller as adf

filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EXData.csv'
df = pd.read_csv(filepath, index_col=['DateTime'], parse_dates=[0], dayfirst=True)
df.index = pd.to_datetime(df.index)

generation = df['UK D+1 Generation']


def difference(data,interval=1):
    diff = []
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return pd.Series(diff)
    
generation = generation.values
gen_diff = difference(generation)
print(adf(gen_diff))
print(adf(generation))





