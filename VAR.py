# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:57:41 2023

@author: EGabriele
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sts
from scipy.stats import norm
import seaborn as sns


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

#Portfolio Retunrs
df['Portfolio'] = ((df['return_N2EX'] + df['return_APX'] + df['return_SIP']) / 3)

#Weights
Weights = 0.85, 0.125, 0.025

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

#Average Return of Portfolio
Average_Return = sts.mean(df['Portfolio'])

#Portfolio Total Value
Portfolio = 20000

#Returns k-th percentile of values in range
def quantile_exc(ser, q):
    ser_sorted = ser.sort_values()
    rank = q * (len(ser) + 1) - 1
    assert rank > 0, 'quantile is too small'
    rank_l = int(rank)
    return ser_sorted.iat[rank_l] + (ser_sorted.iat[rank_l + 1] - 
                                     ser_sorted.iat[rank_l]) * (rank - rank_l)

#Creating list for z-stats and confidence levels to go into
z_stat = []
confidence_list = []
confidence_index = (0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)

#Returning Z-stats from confidence levels and appending values into list
for confidence in confidence_index:
    z_stat_ind = (norm.ppf(1-confidence))
    z_stat.append(z_stat_ind)
    confidence_list.append(confidence)

#Creating second pandas dataframe to store VAR data
df_VAR = pd.DataFrame({'Z_Stat' : z_stat, 'Confidence' : confidence_list})
df_VAR['VCV_VAR_Percent'] = -(Average_Return + df_VAR['Z_Stat']*Stdev)

#Empy list to input Historical VAR data
HS_list = []

#Using Quantile_Exc function, returning Historical VAR data
for i in confidence_index:
    HS_VAR = -quantile_exc(df['Portfolio'], 1-i)
    HS_list.append(HS_VAR)
df_VAR['HS_VAR_Percent'] = np.array(HS_list)

#Calculating VCV and HS VAR data as values
df_VAR['VCV_VAR_Value'] = df_VAR['VCV_VAR_Percent'] * Portfolio
df_VAR['HS_VAR_Value'] = df_VAR['HS_VAR_Percent'] * Portfolio

df_VAR2 = df_VAR.copy()
df_VAR2.drop(columns=['Z_Stat','HS_VAR_Percent','VCV_VAR_Percent'], inplace=True)

columns = ["Confidence", "VCV_VAR_Value", "HS_VAR_Value"]
test_data = df_VAR2

test_data_melted = pd.melt(test_data, id_vars=columns[0],\
                           var_name="source", value_name="value_numbers")
    
# Scale the data, just a simple example of how you might determine the scaling
mask = test_data_melted.source.isin(["VCV_VAR_Value", "HS_VAR_Value"])
scale = 1
test_data_melted.loc[mask, 'value_numbers'] = test_data_melted.loc[mask, 'value_numbers']*scale

# Plot
fig, ax1 = plt.subplots(figsize=(10,6))
g = sns.barplot(x=columns[0], y="value_numbers", hue="source",\
                data=test_data_melted, ax=ax1)

# Create a second y-axis with the scaled ticks
ax1.set_ylabel('Value-at-Risk (£)')
plt.title("VCV and Historical Simulations Value at Risk (£)")
plt.show()

# Back testing using Standard Coverage Test
Sample_size = len(df['Portfolio'])

confidence_levels = []
p_values = []
# Confidence Alpha
for confidence_backtest in np.arange(0.01, 0.2, 0.01):
    
    confidence_backtest = round(confidence_backtest,2)
    confidence_levels.append(confidence_backtest)
    
    # Portfolio Returns
    if (1+df['Portfolio']).product() < 0:
        returns = -1*(((abs((1+df['Portfolio']).product())))**(1/Sample_size)-1)
    else:
        returns = (((1+df['Portfolio']).product())**(1/Sample_size)-1)
       
    # Volatility of returns
    backtest_vol = sts.stdev(df['Portfolio'])
    
    # Variance Co-Variance VAR of returns 95% confidence
    VCV_VAR_backtest =returns + ((norm.ppf(confidence_backtest))*backtest_vol)
    
    # Creating Violations dataframe column
    df['Violations'] = np.where(df['Portfolio'] < VCV_VAR_backtest, 1, 0)
         
    # Total number of violations
    num_violations = df['Violations'].sum()
    
    # Proportion of Violations
    prop_violations = num_violations / Sample_size
    
    # Difference between confidence levels and proportion of violations
    conf_prop_diff = confidence_backtest - prop_violations
    
    # Standard errors from sample and confidence
    standard_error = np.sqrt(((confidence_backtest * (1 - confidence_backtest))/Sample_size))
    
    # Z-stat from standard error and confidence/proportion difference
    z_stat = conf_prop_diff / standard_error
    
    # Return p-value from Standard-coverage test
    p_value = (2 * (1-(norm.cdf(abs(z_stat),False))))
    
    # Appending p-value to empty list
    p_values.append(p_value)
    
    # Results from Standard coverage test
    if p_value < confidence_backtest:
        print(f"Significant result, p-value is less than Confidence level {confidence_backtest}")
    else:
        print(f"Unsignificant result, p-value is more than Confidence level {confidence_backtest}")
    

# Create new pandas dataframe from confidence levels and p-values
df3 = pd.DataFrame({'Confidence': confidence_levels, 'P_values': p_values})

# Plot p-value versus confidence levels
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=df3['Confidence'], y=df3['P_values'], ax=ax)

# Create a second y-axis with the scaled ticks
ax.set_ylabel("P-Value (%)")
plt.title("P-Value versus Confidence Level (%)")
plt.show()

   
  




