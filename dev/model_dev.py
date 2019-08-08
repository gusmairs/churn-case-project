# Model development code for churn case project

# %% Load packages
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from lib.data_tools import data_load, feature_peek
from lib.model_tools import roc_plot

# %% Load and transform data, create model matrix
#
train = 'churn_train.csv'
X_df, y_df = data_load(train)
grp_1 = [
    'rider_score', 'avg_surge', 'weekday_pct'
]
X = X_df.to_numpy()
y = y_df['churn'].to_numpy()
n = X_df.shape[0]

# %% Logistic regression with statsmodels
#
X_df['intercept'] = 1
model = Logit(y_df, X_df)
model = model.fit()
model.summary()
proba = np.array(model.predict(X_df))
y_hat = np.array((proba > 0.5).astype(int))
sum(y_hat == y) / n
idx = np.random.randint(n, size=1000)
fpr, tpr, thresholds = roc_plot(proba, y)
log_loss(y, proba)

# %% Logistic regression with sklearn
#
log_mod = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X, y)
log_mod.coef_, log_mod.intercept_
proba = log_mod.predict_proba(X)[:, 1]
y_hat = np.array((proba > 0.5).astype(int))
sum(y_hat == y) / n
idx = np.random.randint(n, size=1000)
fpr, tpr, thresholds = roc_plot(proba[idx], y[idx])
log_loss(y, proba)
log_mod.score(X, y)
# %%
#
# %% Random forest with sklearn
#
tree_mod = DecisionTreeClassifier(criterion='entropy', max_depth=10)
tree_mod.fit(X, y)
proba = tree_mod.predict_proba(X)[:, 1]
y_hat = np.array((proba > 0.5).astype(int))
sum(y_hat == y) / n
idx = np.random.randint(n, size=1000)
fpr, tpr, thresholds = roc_plot(proba[idx], y[idx])
log_loss(y, proba)
# %%
#
# %% Random forest with sklearn
rf_mod = RandomForestClassifier(criterion='entropy', max_depth=10)
rf_mod.fit(X, y)
proba = rf_mod.predict_proba(X)[:, 1]
y_hat = np.array((proba > 0.5).astype(int))
sum(y_hat == y) / n
idx = np.random.randint(n, size=1000)
fpr, tpr, thresholds = roc_plot(proba[idx], y[idx])
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
fpr, tpr, thresholds = roc_plot(proba[idx], y[idx])
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
