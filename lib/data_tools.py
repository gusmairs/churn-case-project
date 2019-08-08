import pandas as pd
from os import path

def data_load(data_set):
    '''
    Action: Load, clean and transform raw data
      Take: Path to dataset
    Return: Clean dataframe 'df' and series 'target'
    '''

    data_path = '~/OneDrive/DS_Study/churn-project/data'

    # Load data
    dat = pd.read_csv(
        path.join(data_path, data_set),
        parse_dates=['last_trip_date', 'signup_date']
    )

    # Grab features needing no transform
    df = dat.loc[:, [
        'avg_dist', 'avg_surge', 'surge_pct', 'trips_in_first_30_days',
        'weekday_pct'
    ]].copy()

    # df.columns = ['avg_dist', 'avg_surge', 'pct_surge', 'first_30',
    #               'weekday_pct']
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
    # Create target churn measure
    last_trip_delta = max(dat['last_trip_date']) - dat['last_trip_date']
    churn_days = last_trip_delta.apply(lambda x: x.days)
    target['churn'] = (churn_days > 30).astype(int)

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
