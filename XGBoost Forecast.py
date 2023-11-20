# -*- coding: utf-8 -*-
"""
Spyder Editor


This is a temporary script f
"""
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import seaborn as sns
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

spot_data_path = r'C:\Users\EGabriele\OneDrive - Viridor\Documents\Personal\Industry Knowledge\Technical\Forecasting Models\Python work\N2EXData.csv'
df = pd.read_csv(spot_data_path, index_col=['DateTime'], usecols= ["DateTime","N2EX"], parse_dates=[0], dayfirst=True)
df.set_index = pd.to_datetime(df.index)

train = df.loc[df.index < '01-01-2022']
test = df.loc[df.index >= '01-01-2022']


tss = TimeSeriesSplit(n_splits=3, test_size=24*365*1, gap=24)
df = df.sort_index()
fig, axs = plt.subplots(3,1,figsize=(15,5),sharex=True)
plt.show()


fold =0
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    train['N2EX'].plot(ax=axs[fold], label = 'Training Set', title = f'Data Train/Split Fold {fold}')
    test['N2EX'].plot(ax=axs[fold], label='Test Set')
    axs[fold].axvline(test.index.min(),color='black',ls='--')
    fold += 1
#plt.show()

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year

df = create_features(df)

fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data=df, x=df['hour'], y='N2EX')
ax.set_title('MW by Hours')
plt.show()



def add_lags(df):
    target_map = df['N2EX'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('1 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('3 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('5 days')).map(target_map)
    return df


df = add_lags(df)
tss = TimeSeriesSplit(n_splits=2, test_size=24*365*1, gap=24)
df = df.sort_index()


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    train = create_features(train)
    test = create_features(test)
    
    FEATURES = ['hour', 'dayofweek', 'month', 'year', 'lag1', 'lag2', 'lag3']
    TARGET = 'N2EX'
    
    X_train = train[FEATURES]
    y_train = train[TARGET]
    
    X_test = test[FEATURES]
    y_test = test[TARGET]
    
    reg = xgb.XGBRegressor(base_score = 0.5, booster = 'gbtree',
                           n_estimators = 1000, early_stopping_rounds = 50,
                           objective = 'reg:linear', max_depth = 3,
                           learning_rate = 0.01)
    reg.fit(X_train, y_train, eval_set=[(X_train,y_train), (X_test, y_test)],
            verbose = 100)
    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')




    
    



