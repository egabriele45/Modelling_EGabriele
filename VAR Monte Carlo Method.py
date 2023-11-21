# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:47:05 2023

@author: EGabriele
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sts
from scipy.stats import norm
import seaborn as sns
from random import random


# Pandas to read dataset, please change filepath
filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\VAR\RawData.csv'
# Dayfirst and parse-dates used to properly format datetime strings in index
df = pd.read_csv(filepath, index_col = ['Date'], dayfirst=(True), parse_dates=[0])
df.index = pd.to_datetime(df.index)

#Returns of each price
df['return_N2EX'] = (df['N2EX']/df['N2EX'].shift(1))-1
df['return_APX'] = (df['APX']/df['APX'].shift(1))-1
df['return_SIP'] = (df['SIP']/df['SIP'].shift(1))-1

#Remove top row
df = df.iloc[1:]

#Average Daily Return
N2EX_Average = sts.mean(df['return_N2EX'])
APX_Average = sts.mean(df['return_APX'])
SIP_Average = sts.mean(df['return_SIP'])

#Daily Standard Deviation
N2EX_Std = sts.stdev(df['return_N2EX'])
APX_Std = sts.stdev(df['return_APX'])
SIP_Std = sts.stdev(df['return_SIP'])

#Weights
Weights = 0.85, 0.125, 0.025

#Portfolio Value
Portfolio = 10000

#Create array for weights
Weight_Array = np.array(Weights)

#Transpose Weight Matrix
WA_Tr = Weight_Array.transpose()

#Create Array for Cov Matrx
Array = np.array([df['return_N2EX'],df['return_APX'],df['return_SIP']])

#Covariance matrix of retuns
cov_matrix = np.cov(Array, bias=True)

#Portfolio Variance
Portfolio_Var = np.matmul(Weight_Array,np.matmul(cov_matrix,WA_Tr))

#Portfolio Standard Deviation
Stdev = np.sqrt(Portfolio_Var)

#Returns k-th percentile of values in range
def quantile_exc(ser, q):
    ser_sorted = ser.sort_values()
    rank = q * (len(ser) + 1) - 1
    assert rank > 0, 'quantile is too small'
    rank_l = int(rank)
    return ser_sorted.iat[rank_l] + (ser_sorted.iat[rank_l + 1] - 
                                     ser_sorted.iat[rank_l]) * (rank - rank_l)

#Setting number of days to 20
period = 20

#Setting confidence level
Conf_Lev = 0.05

#Expected return
expected_return = ((Portfolio*((N2EX_Average*Weight_Array[0])
                              +(APX_Average*Weight_Array[1])
                              +(SIP_Average*Weight_Array[2])))
                               *(period/len(df['return_N2EX'])))

#Inferred confidence level of 5%
CI = norm.ppf(Conf_Lev)

#Calculating VaR from Inferred confidence level
computed_VaR = expected_return - (Portfolio*Stdev*CI*((period/(len(df['return_N2EX'])**0.5))))

#Monte Carlo VaR for 20 days
for days in range(1,21):

    #Create empty list for Monte Carlo data to go in
    VaR_list = []
    
    #Creating for loop for 10,000 Random VaR generators
    for sim in range(1,10001):
        
        #Random confidence interval, random() provides random float numbers between 0 and 1
        confidence_interval = norm.ppf(random())
            
        #VaR for portfolio
        VaR = expected_return - (Portfolio*Stdev*confidence_interval*((days/(len(df['return_N2EX'])**0.5)))) 
        
        #Append VaR to empty list
        VaR_list.append(VaR)
        
    #Create pandas dataframe from Random VaR
    df_VaR = pd.DataFrame({'Random VaR': VaR_list})
        
    #Inputted Confidence Interval
    conf_interval = 1 - Conf_Lev
    
    #Returns Monte-Carlo VaR using quantile function
    Monte_Carlo_VaR = quantile_exc(df_VaR['Random VaR'], conf_interval)

#Print Calculated VaR and Monte Carlo VaR
print(f"For a confidence level of  {Conf_Lev}, the computed VaR is \n {computed_VaR}")
print(f"For a confidence level of  {Conf_Lev}, the Monte Carlo VaR is \n {Monte_Carlo_VaR}")

#Plot histogram of Monte-Carlo VaR for 20 days
plt.figure(figsize=(10,6))
plt.hist(x=df_VaR['Random VaR'], color = 'green', ec='black')
plt.xlabel("Monte-Carlo VaR")
plt.ylabel("Number of Simulations")
plt.title("Monte Carlo VaR")
plt.xlim((min(df_VaR['Random VaR'])),(max(df_VaR['Random VaR'])))
plt.show()

