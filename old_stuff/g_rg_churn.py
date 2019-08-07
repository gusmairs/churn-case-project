import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from statsmodels.discrete.discrete_model import Logit
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence

from g_rg_churn_functions import data_load, feature_peek
from roc_plot import roc

path = 'data/churn_train.csv'
X_df, y_df = data_load(path)
features_sub = ['rider_score', 'avg_surge', 'weekday_pct']
features_sub = ['avg_dist', 'avg_surge', 'pct_surge', 'first_30', 'weekday_pct',
                'rider_score', 'driver_score', 'iphone', 'luxury', 'signup',
                'city_wint', 'city_asta']
X_df = X_df[features_sub].copy()
X_df['constant'] = 1
X = X_df.to_numpy()
y = y_df['last_trip'].to_numpy()
n = X_df.shape[0]
#
# %% Tree regression with sklearn
#
tree = DecisionTreeRegressor(max_depth=20)
tree.fit(X, y)
tree.score(X, y)
pred = tree.predict(X).astype(int)
pred[0:10]
resid = pred - y
sns.distplot(resid)
x = X_df['rider_score'].to_numpy()
sns.scatterplot(x, resid, alpha=0.1).set(xlim=(3, 5))
# %%
#
# %% RF regressor w sklearn
#
rf = RandomForestRegressor(max_depth=20)
rf.fit(X, y)
pred = tree.predict(X).astype(int)
resid = pred - y
sns.distplot(resid)
x = X_df['rider_score'].to_numpy()
sns.scatterplot(x, resid, alpha=0.1).set(xlim=(3, 5))

# %% Logistic regression with statsmodels
#
model = Logit(y_df, X_df)
model = model.fit()
model.summary()
proba = np.array(model.predict(X_df))
y_hat = np.array((proba > 0.5).astype(int))
sum(y_hat == y) / n
idx = np.random.randint(n, size=1000)
fpr, tpr, thresholds = roc(proba[idx], y[idx])
log_loss(y, proba)
# %%
#
# %% Logistic regression with sklearn
log_mod = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X, y)
log_mod.coef_, log_mod.intercept_
proba = log_mod.predict_proba(X)[:, 1]
y_hat = np.array((proba > 0.5).astype(int))
sum(y_hat == y) / n
idx = np.random.randint(n, size=1000)
fpr, tpr, thresholds = roc(proba[idx], y[idx])
log_loss(y, proba)
# %%
#
# %% Random forest with sklearn
#
tree_mod = DecisionTreeClassifier(criterion='entropy', max_depth=10)
tree_mod.fit(X, y)
proba = tree_mod.predict_proba(X)[:, 1]
proba
y_hat = np.array((proba > 0.5).astype(int))
sum(y_hat == y) / n
idx = np.random.randint(n, size=1000)
fpr, tpr, thresholds = roc(proba[idx], y[idx])
log_loss(y, proba)
# %%
#
# %% Random forest with sklearn
rf_mod = RandomForestClassifier(criterion='entropy', max_depth=10)
rf_mod.fit(X, y)
proba = rf_mod.predict_proba(X)[:, 1]
proba
y_hat = np.array((proba > 0.5).astype(int))
sum(y_hat == y) / n
idx = np.random.randint(n, size=1000)
fpr, tpr, thresholds = roc(proba[idx], y[idx])
log_loss(y, proba)
impt = rf_mod.feature_importances_
idx = (-impt).argsort()
sns.barplot(y=X_df.columns[idx], x=impt[idx], orient='h')
# %%
#
# %% Gradient boost with sklearn
gb_mod = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, subsample=0.8)
gb_mod.fit(X, y)
proba = gb_mod.predict_proba(X)[:, 1]
proba
y_hat = np.array((proba > 0.5).astype(int))
sum(y_hat == y) / n
idx = np.random.randint(n, size=1000)
fpr, tpr, thresholds = roc(proba[idx], y[idx])
log_loss(y, proba)
impt = gb_mod.feature_importances_
idx = (-impt).argsort()
sns.barplot(y=X_df.columns[idx], x=impt[idx], orient='h')
fig, ax = plot_partial_dependence(gb_mod, X,
                                  features=[2],
                                  feature_names=X_df.columns)
plt.tight_layout()

# Old stuff
feature_peek(X_df, X_df.columns)
sns.distplot(dat['ago'])

fig, ax = plt.subplots(1, 1, figsize=(4, 10))
sns.countplot(y=dat['luxury_car_user'])

sns.distplot(dat['weekday_pct'])


x = np.array([3, 2, 4, 6, 5])
x, (-x).argsort()
x
