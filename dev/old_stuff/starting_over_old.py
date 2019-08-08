import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)
from functions import *
from sklearn.model_selection import train_test_split

datetime_cols = ['last_trip_date', 'signup_date']
churn = pd.read_csv('data/churn.csv', parse_dates=datetime_cols)
train_data = pd.read_csv('data/churn_train.csv', parse_dates=datetime_cols)

train_data['num_days_since'] = ((np.max(train_data['last_trip_date'])
                                 -train_data['last_trip_date'])).astype('timedelta64[D]')
train_data['churn'] = (train_data['num_days_since'] > 30)   
train_data['churn'] = train_data['churn'].astype(int) 

ohe = OneHotEncoder()
ohe.fit(train_data['city'])
ohe_city = ohe.transform(train_data['city'])

rating_by_is_nan = train_data['avg_rating_by_driver'].isna().astype(int)
rating_by_is_nan.rename('rating_by_is_nan', inplace=True)
train_data['avg_rating_by_driver'].fillna(
    np.mean(train_data['avg_rating_by_driver']), inplace=True)

rating_of_is_nan = train_data['avg_rating_of_driver'].isna().astype(int)
rating_of_is_nan.rename('rating_of_is_nan', inplace=True)
train_data['avg_rating_of_driver'].fillna(
    np.mean(train_data['avg_rating_by_driver']), inplace=True)

phone_is_iphone = (train_data['phone'] == 'iPhone').astype(int)
phone_is_iphone.rename('is_iphone', inplace=True)
phone_is_android = (train_data['phone'] == 'Android').astype(int)
phone_is_android.rename('is_android', inplace=True)

train_data['luxury_car_user'] = train_data['luxury_car_user'].astype(int)

train_data['miles_per_month'] = train_data['avg_dist'] * train_data['trips_in_first_30_days']

cols_to_use = ['avg_dist',
               'avg_rating_by_driver',
               'avg_rating_of_driver',
               'avg_surge',
               'surge_pct',
               'trips_in_first_30_days',
               'luxury_car_user',
               'weekday_pct',
               'miles_per_month']

frames = [train_data[cols_to_use],
          ohe_city,
          rating_by_is_nan,
          rating_of_is_nan,
          phone_is_android,
          phone_is_iphone]
X = pd.concat(frames, axis=1)
y = train_data['churn']

lr_model = LogisticRegression()
rfc_model = RandomForestClassifier(n_estimators=100,
                                  max_depth=5)
gbc_model = GradientBoostingClassifier(learning_rate=0.1,
                                      n_estimators=100,
                                      subsample=0.5,
                                      max_depth=5)

kf = KFold(n_splits=10, shuffle=True)
lr_logloss = np.zeros((10,2))
rfc_logloss = np.zeros((10,2))
gbc_logloss = np.zeros((10,2))
for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train = X.iloc[train_idx,:]
    X_test = X.iloc[test_idx,:]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    lr_model.fit(X_train, y_train)
    pred_probs_train = lr_model.predict_proba(X_train)
    lr_logloss[i,0] = log_loss(y_train, pred_probs_train)
    pred_probs_test = lr_model.predict_proba(X_test)                           
    lr_logloss[i,1] = log_loss(y_test, pred_probs_test)
    
    rfc_model.fit(X_train, y_train)
    pred_probs_train = rfc_model.predict_proba(X_train)
    rfc_logloss[i,0] = log_loss(y_train, pred_probs_train)
    pred_probs_test = rfc_model.predict_proba(X_test)                           
    rfc_logloss[i,1] = log_loss(y_test, pred_probs_test)
    
    gbc_model.fit(X_train, y_train)
    pred_probs_train = gbc_model.predict_proba(X_train)
    gbc_logloss[i,0] = log_loss(y_train, pred_probs_train)
    pred_probs_test = gbc_model.predict_proba(X_test)                           
    gbc_logloss[i,1] = log_loss(y_test, pred_probs_test)

lr_logloss_mean = np.mean(lr_logloss, axis=0)
rfc_logloss_mean = np.mean(rfc_logloss, axis=0)
gbc_logloss_mean = np.mean(gbc_logloss, axis=0)

print('Mean Log Loss (10-fold cross-validation)')
print('Model                     | Train  | Test')
print('--------------------------+--------+-------')
print('LogisticRegression        | {0:6.4f} | {1:6.4f}'.format(lr_logloss_mean[0], lr_logloss_mean[1]))
print('RandomForestClassifier    | {0:6.4f} | {1:6.4f}'.format(rfc_logloss_mean[0], rfc_logloss_mean[1]))
print('GradientBoostingClassifier| {0:6.4f} | {1:6.4f}'.format(gbc_logloss_mean[0], gbc_logloss_mean[1]))