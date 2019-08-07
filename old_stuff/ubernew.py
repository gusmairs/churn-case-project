import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_peek import feature_peek
from functions import PhoneNan

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer



dat = pd.read_csv('data/churn.csv',
                  parse_dates=['last_trip_date', 'signup_date'])
dat.head()

# Trying out column transformer
#
dat_na = dat.dropna(axis=0)
preprocess = make_column_transformer(
    (OneHotEncoder(), ['phone'])
)

dat_features = dat_na[['phone']]
preprocess.fit_transform(dat_features)[:5]
#
#

ph = PhoneNan()
ph.fit()
n, android, apple = ph.transform(dat['phone'])
apple.value_counts()

dat['phone'].value_counts()
dat['phone_is_nan'].value_counts()

dat['phone_is_nan'] = dat['phone'].isna().astype(int)
dat['phone_is_android'] = (dat['phone'] == 'Android')
dat['phone_is_apple'] = (dat['phone'] == 'iPhone').astype(int)

dat['is_apple'].value_counts()

dat['phone_is_nan'].value_counts()

lb = LuxuryBoolean()
lb.fit()
lx = lb.transform(dat['luxury_car_user'])
lx[0:10]


# %%
# Testing the days_ago transformer
ago = DaysAgo()
ago.fit()
a = ago.transform(dat['last_trip_date'])
a[0:10]
# Testing the churned transformer
ch = Churned()
ch.fit()
c = ch.transform(dat['last_trip_date'])
c[0:10]
# %%


feature_peek(dat, ['avg_surge'])
dat['avg_rating_by_driver'].value_counts()

# Designing the days_ago and churned transformers
dat['ago'] = pd.to_datetime(max(dat['last_trip_date'])) - pd.to_datetime(dat['last_trip_date'])
dat['ago'] = dat['ago'].apply(lambda x: x.days)
dat['churned'] = (dat['ago'] > 30).astype(int)
dat['luxury_car_user'].value_counts()
sns.distplot(dat['ago'])

fig, ax = plt.subplots(1, 1, figsize=(4, 10))
sns.countplot(y=dat['luxury_car_user'])

sns.distplot(dat['weekday_pct'])
