# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:08:22 2023

@author: EGabriele
"""

import pyodbc as odbc
import pandas as pd
import numpy as np

server = 'adwhl-vwm-sql01.database.windows.net'
database = 'dwhVwmL' 
username = 'BIandData_svc' 
password = 'uQGYTdGGqHukK5ss'
driver = '{ODBC Driver 17 for SQL Server}'


cnxn = odbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
query = "SELECT [Assessment_Date], AVG([Value]) As Power_Curve FROM [rpt].[Marex_Power_UK_Curve] Where Type = 'UK Bsld' AND CONTRACT = 'Sum-24' AND Assessment_Date > '2022-04-01' GROUP BY Assessment_Date ORDER BY Assessment_Date ASC;"
df_power = pd.read_sql(query, cnxn, index_col=['Assessment_Date'])

query2 = "SELECT [Assessment_Date], AVG([Value]) As Gas_Curve FROM [rpt].[Marex_Gas_NBP_Curve] Where Delivery_Point = 'NBP' AND Contract = 'Sum-24' AND Assessment_Date > '2022-04-01' GROUP BY Assessment_Date ORDER BY Assessment_Date ASC;"
df_gas =pd.read_sql(query2,cnxn, index_col=['Assessment_Date'])

df = pd.concat([df_power,df_gas], axis=1, join='inner')

df['Power_Returns'] = (df['Power_Curve'] / df['Power_Curve'].shift(1)) - 1
df['Gas_Returns'] = (df['Gas_Curve'] / df['Gas_Curve'].shift(1)) - 1

#Remove top row
df = df.iloc[1:]

#Linest function, least squares method to calculate straight line of best fit for data
matrix = np.polyfit(df['Power_Returns'],df['Gas_Returns'],1)

#Expected Retruns
df['Exp_Returns'] = (df['Gas_Returns'] * matrix[0]) + matrix[1]

#Residuals
df['Residuals'] = df['Power_Returns'] - df['Exp_Returns'] 

#Residuals Squared
df['Residual_Sq'] = df['Residuals']**2

#Differences Squared
df['Squared_Diff'] = (df['Residuals'].shift(1)-df['Residuals'])**2


Sum_Sq_Diff = df['Squared_Diff'].sum()

Sum_Resi = df['Residual_Sq'].sum()

#Dubrin Watson Cal - 0 to 4 (0 positive autocorrelation, 2 no autocorrelation, 4 negative autocorrelation)
Durbin_Watson = Sum_Sq_Diff/Sum_Resi
print(Durbin_Watson)