import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_load(filepath):
    '''
    Action: Load, clean and transform raw data
      Take: Path to dataset
    Return: Clean dataframe 'df' and series 'target'
    '''

    # Load data
    dat = pd.read_csv(filepath, parse_dates=['last_trip_date', 'signup_date'])

    # Grab features needing no transform
    df = dat[['avg_dist', 'avg_surge', 'surge_pct', 'trips_in_first_30_days',
             'weekday_pct']].copy()
    df.columns = ['avg_dist', 'avg_surge', 'pct_surge', 'first_30',
                  'weekday_pct']
    target = pd.DataFrame()

    # Fill NaNs in avg_rating_by_driver
    mean_by = dat['avg_rating_by_driver'].mean()
    df['rider_score'] = dat['avg_rating_by_driver'].fillna(mean_by)

    # Fill NaNs in avg_rating_of_driver
    mean_of = dat['avg_rating_of_driver'].mean()
    df['driver_score'] = dat['avg_rating_of_driver'].fillna(mean_of)

    # Code Phone and fill NaNs (NaN = iPhone)
    df['iphone'] = (dat['phone'] != 'Android').astype(int)

    # Make luxury_car_user a 1-0
    df['luxury'] = dat['luxury_car_user'].astype(int)

    # Convert last_trip_date into days since measure
    last_trip_delta = max(dat['last_trip_date']) - dat['last_trip_date']
    target['last_trip'] = last_trip_delta.apply(lambda x: x.days)

    # Convert signup_date to day integer
    signup_delta = max(dat['signup_date']) - dat['signup_date']
    df['signup'] = signup_delta.apply(lambda x: x.days)

    # Code cities
    df['city_wint'] = (dat['city'] == 'Winterfell').astype(int)
    df['city_asta'] = (dat['city'] == 'Astapor').astype(int)
    df['city_king'] = (dat['city'] == 'King\'s Landing').astype(int)

    return df, target

def feature_peek(df, column_list):
    for c in column_list:
        cn = df[c][~df[c].isna()]
        print(c)
        print(str(df[c].dtype) + ' | ' +
              str(sum(df[c].isna())) + ' NaNs | ' +
              str(len(cn.unique())) + ' unique')
        if len(cn.unique()) < 10:
            print('Values: ' + str(list(cn.unique())))
        elif df[c].dtype != 'object':
            print('min ' + str(df[c].min()) +
                  ' | med ' + str(df[c].median()) +
                  ' | max ' + str(df[c].max()))
        else:
            print('Sample: ' + str(list(cn.sample(5))))
        if c != column_list[-1]:
            print()

def roc_plot(proba, y):
    '''
    Action: Build fpr and tpr arrays and plot the ROC
        In: The probability array and target labels from a fit model
       Out: The fpr, tpr arrays and the set of threshholds used
    '''
    thresh = np.argsort(proba)
    fpr, tpr = [], []
    for t in thresh:
        predict = np.array(proba > proba[t]).astype(int)
        fpr.append(np.sum(predict * (1 - y)) / np.sum(1 - y))
        tpr.append(np.sum(predict * y) / np.sum(y))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], c='gray', linewidth=0.6)
    ax.set(xlabel='False Positive Rate (1 - Specificity)',
           ylabel='True Positive Rate (Sensitivity, Recall)',
           title='ROC Plot')

    return fpr, tpr, proba[thresh]
