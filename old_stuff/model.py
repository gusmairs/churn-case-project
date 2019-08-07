# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.pipeline import Pipeline
<<<<<<< HEAD
from sklearn.preprocessing import StandardScaler
from functions import (
    DaysAgo, Churned, LuxuryBoolean, OneHotEncoder, MilesPer30Days)
=======
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from functions.py import DaysAgo, Churned, log_loss_score
>>>>>>> d3bc9c4d6dafeb50e5a0f15786d54cc71ff5762d
from sklearn.model_selection import train_test_split

from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


'''
log_model = LogisticRegression(    )
tree_model = DecisionTreeClassifier(   )
rf_model = RandomForestClassifier( n_estimators = 1000   )
gb_model = GradientBoostingClassifier(   )
reg_model = LinearRegression(fit_intercept = False, normalize = True)
'''

# Define functions for pipeline creation
'''
def continuous_specification(col_name):
    select_name = "{}_select".format(col_name)
    return Pipeline([
        (select_name, ColumnSelector(name=col_name))
        ])

def categorical_specification(col_name):
    select_name = "{}_select".format(col_name)
    ohe_name = "{}_ohe".format(col_name)
    return Pipeline([
        (select_name, ColumnSelector(name=col_name)),
        (ohe_name, OneHotEncoder())
        ])
'''

# Load data
datetime_cols = ['last_trip_date', 'signup_date']
train_data = pd.read_csv('data/churn_train.csv', parse_dates=datetime_cols)
'''
from datetime import date
train_data['num_days_since'] = (np.max(train_data['last_trip_date']) - train_data['last_trip_date']).astype('timedelta64[D]')
train_data['churn'] = ( train_data['num_days_since'] > 30)   
train_data['churn'] = train_data['churn'].astype('bool')   


#######    Get X and y for base model 
y = train_data['churn']
xvals = ['avg_dist', 'num_days_since']
X = train_data[xvals]


###### Base Tree ##########
tree_model.fit(X, y)
tree_y_hat = tree_model.predict(X)
tree_y_hat
log_loss_score( tree_y_hat, y)

######     Base logistic    ##########
log_model = LogisticRegression(    )
log_model.fit(X, y)
log_y_hat = log_model.predict(X)
log_y_hat
log_loss_score( log_y_hat, y)

######     Base Regression     ##########
gb_model = GradientBoostingClassifier(   )
gb_model.fit(X, y)
gb_y_hat = gb_model.predict(X)
gb_y_hat
<<<<<<< HEAD
'''
=======
log_loss_score( gb_y_hat, y)
>>>>>>> d3bc9c4d6dafeb50e5a0f15786d54cc71ff5762d

# Create transformer pipeline for each feature
avg_dist_spec = ColumnSelector(name='avg_dist')
'''
rating_by_spec = Pipeline([
    ('rating_by_select', ColumnSelector(name='avg_rating_by_driver')),
    ('rating_by_is_missing', )
    ('rating_by_spline', )
])

rating_of_spec = Pipeline([
    ('rating_of_select', ColumnSelector(name='avg_rating_of_driver')),
    ('rating_of_is_missing', )
    ('rating_of_spline', )
])
'''
avg_surge_spec = ColumnSelector(name='avg_surge')

city_spec = Pipeline([
    ('city_selector', ColumnSelector(name='city')),
    ('city_ohe', OneHotEncoder())
])

days_ago_spec = Pipeline([
    ('last_trip_select', ColumnSelector(name='last_trip_date')),
    ('days_ago', DaysAgo())
])

phone_spec = Pipeline([
    ('phone_selector', ColumnSelector(name='phone')),
    ('phone_ohe', OneHotEncoder())
])

surge_pct_spec = ColumnSelector(name='surge_pct')

first_30d_spec = ColumnSelector(name='trips_in_first_30_days')

lux_car_spec = Pipeline([
    ('lux_car_selector', ColumnSelector(name='luxury_car_user')),
    ('lux_car_bool', LuxuryBoolean())
])

weekday_pct_spec = ColumnSelector(name='weekday_pct')

churn_spec = Pipeline([
    ('last_trip_select', ColumnSelector(name='last_trip_date')),
    ('churn', Churned())
])


# Create target vector
targets = churn_spec.transform(train_data)

# Create FeatureUnion to join transformers
feature_pipeline = FeatureUnion([
    ('avg_dist_spec', avg_dist_spec),
#    ('rating_by_spec', rating_by_spec),
#    ('rating_of_spec', rating_of_spec),
    ('avg_surge_spec', avg_surge_spec),
    ('city_spec', city_spec),
    ('days_ago_spec', days_ago_spec),
    ('phone_spec', phone_spec),
    ('surge_pct_spec', surge_pct_spec),
    ('first_30d_spec', first_30d_spec),
    ('lux_car_spec', lux_car_spec),
    ('weekday_pct_spec', weekday_pct_spec)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(train_data, targets, test_size=0.3)

# KFolds 

# Fit FeatureUnion with training set and transform features
# feature_pipeline.fit(X_train)
train_features = feature_pipeline.transform(X_train)

# Instantiate model (LogisticRegression, RandomForest, etc. from sklearn)
model = LogisticRegression()

# Fit model with transformed features
model.fit(train_features, y_train)

# Predict train and test
pred_train = model.predict(train_features)
test_features = feature_pipeline.transform(X_test)
pred_test = model.predict(test_features)

# Compute log-loss score


# Evaluate 