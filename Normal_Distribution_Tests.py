# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:27:59 2023

@author: EGabriele
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sts
from scipy import stats



# Pandas to read dataset, please change filepath
filepath = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\NormDistrTestData.csv'
# Dayfirst and parse-dates used to properly format datetime strings in index
df = pd.read_csv(filepath, index_col = ['Date'], dayfirst=(True), parse_dates=[0])
df.index = pd.to_datetime(df.index)

class norm_dist:
    def returns(df):
        df['Returns'] = (df['N2EX Price']/df['N2EX Price'].shift(1))-1
        df = df.iloc[1:]
        global count, mean, Stdev
        count = len(df)
        mean = sts.mean(df['Returns'])
        Stdev = sts.stdev(df['Returns'])
        return df['Returns']
   
    
    def ranked(returns):
        Ranks = np.arange(0,len(returns),1)
        Ranked = sorted(returns.copy())
        global df_ranked
        df_ranked = pd.DataFrame({'Rank':Ranks,'Ranked':Ranked})
        df_ranked = df_ranked.iloc[1:]
        df_ranked['Empirical_Distribution'] = df_ranked['Rank'] / len(df_ranked)
        df_ranked['Theoretical_Distribution'] = stats.norm.cdf(df_ranked['Ranked'],mean,Stdev)
        df_ranked['Rank'] = df_ranked['Rank'].astype(float)
        return df_ranked
        
    
    def Kol_Smirov(rkd):
        difference = abs(rkd['Empirical_Distribution'] - rkd['Theoretical_Distribution'])
        supernum = max(difference)
        KS_Stat = supernum*np.sqrt(count)
        KS_critical_val = 1.517
        p_val = np.exp(-(supernum**2)*count)
        if KS_Stat > KS_critical_val:
            print("KS Test Result: Not normally distributed")
        else:
            print("KS Test Result: Normally distributed")
        print(f"There is {p_val}"+"% of data being normally distributed using Kolmogorov-Smirnov")
        
    def And_Darling(rkd):
        df_ranked['Theoretical_backwards'] = sorted(df_ranked['Theoretical_Distribution'],reverse=True) 
                          
        df_ranked['S'] = (((2*df_ranked['Rank'])-1)/count)*((np.log(df_ranked['Theoretical_Distribution'])+(np.log(1-df_ranked['Theoretical_backwards']))))
        df_ranked.at[1,'S'] = (((2*df_ranked['Rank'][1])-1)/count)*np.log(df_ranked['Theoretical_Distribution'][1])
        df_ranked.at[2,'S'] = (((2*df_ranked['Rank'][2])-1)/count)*np.log(df_ranked['Theoretical_Distribution'][2])
        #df_ranked.to_csv(r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\NormTest.csv')
        S = df_ranked['S'].sum()
        And_Darling_Stat = np.sqrt((-1*count)-S)
        exponent = 1.2937 - (5.709*And_Darling_Stat)+(0.0186*(And_Darling_Stat**2))
        p_Value = np.exp(exponent)
        return p_Value

if __name__ == "__main__":
    norm_dist.returns(df)
    norm_dist.ranked(df['Returns'])
    norm_dist.Kol_Smirov(df_ranked)
    norm_dist.And_Darling(df_ranked)

    

    










