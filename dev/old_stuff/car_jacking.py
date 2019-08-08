import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import r2_score as r2
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from basis_expansions.basis_expansions import (
    Polynomial, LinearSpline)
from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)
from regression_tools.plotting_tools import (
    plot_univariate_smooth,
    bootstrap_train,
    display_coef,
    plot_bootstrap_coefs,
    plot_partial_depenence,
    plot_partial_dependences,
    predicteds_vs_actuals)
from sklearn.preprocessing import StandardScaler
from regression_tools.dftransformers import (
    ColumnSelector, Identity,
    FeatureUnion, MapFeature,
    StandardScaler)
###   %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
from math import ceil
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from basis_expansions.basis_expansions import NaturalCubicSpline
np.random.seed(137)
import sys
from performotron import Comparer
import os 

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline





os.getcwd()
raw = pd.read_csv("./data/churn_train.csv")
df = raw.copy()
df.head()
df.describe()
df.info()
df.head()
df.shape






###############   Start with all columns ##########################
Xfeatures = df.columns
Xfeatures
################  All columns X.features   #########################




#####       log loss Scoring Function for all models     #########
def log_loss_score( y_hats, y_act):
    return np.sum(  -( ((y_act) * np.log (y_hat)) + (( 1 - y_act) * np.log (1 - y_hat)   )  ))


####################################################################





####   Transform date last used into boolean  #########
### Find the y values using last 30 days:  0 = not churn    or    1 = churn    #### 
y = 




###   This is the model that will first be used   #######
log_model = LogisticRegression(    )
tree_model = DecisionTreeClassifier( n_estimators = 1000   )
rf_model = RandomForestClassifier(   )
gb_model = GradientBoostingClassifier(   )
reg_model = LinearRegression(fit_intercept = False, normalize = True)


#########################################################


  #### ## Plot charts for each item in X.features       ######
fig, axs = plt.subplots(2, 2 , figsize=(20, 15))
for i, ax in enumerate(axs.flatten()):
    ax.scatter(df[Xfeatures[i]], y, alpha = .2)
    ax.set_title("Scatter plot of: " + Xfeatures[i], fontsize = 10)
    ax.set_xlabel(Xfeatures[i] , fontsize = 6)
    ax.set_ylabel(str(y),  fontsize = 6)
plt.show()

############   


#####      Transform catagorical (if necessary)     ########




#####     
### Plot scatter matrix for variables you chose #######

_ = scatter_matrix(df[Xfeatures], alpha=0.2, figsize=(20, 20), diagonal='kde')
plt.show()





### This will show a description for all of the columns in the data.
for col in df.columns:
    print (df[col].describe(include = 'all'))
##    Show description of all columns   ########





#       This will print the info for columns in Xfeatures...
Xfeatures
df[Xfeatures].info()
df[Xfeatures].describe()



#      Info for all columns    #####   
df.head()
df.describe()

###   This is the model that will first be used   #######

model = LinearRegression(fit_intercept = True, normalize = True)



## Choose the variables and set value for X training data
X = df[Xfeatures[:4]]
y = np.log(df[Xfeatures[4]])   # this is the log of y  




###    This will plot everything. Even the things that make no sense  #### 

for num in range(1,101):
    y_hat[y_hat.Unit == num].plot.scatter( y , , alpha = .3) 
    plt.show()



## Choose the variables for the test data  #####
test_y = test.remaining_life
y_hat['idx'] = list(range(0,len(y_hat)))

#############   


###### The values for X and y are now set ########
def residual_plot_scatter(ax, y, y_hat):
    ax.axhline(0, color="black", linestyle="--")
    ax.scatter(np.e**y, y_hat, color="grey", alpha=0.5)
    ax.set_ylabel("Residuals ($y - \hat y$)")
#############   


### Try model with all variables ###
Xall = df.copy()
Xall.head()
Xall = Xall.drop(['Op_Set_1', 'Op_Set_2', 'Op_Set_3', 'T2_Inlet', 'Unit', 'time_cycles', 'cycles_to_fail', 'W32_LPT_cool_bl_min', 'T50_LPT_max',
       'htBleed_Enthalpy_max', 'EPR_P50/P2', 'nf_DMD_dem_fan_sp', 'T50_stddev', 'W32_stddev', 'HTP_Bleed_var'] , axis = 1)
Xall = Xall.drop(Xall.columns[0], axis = 1)

col = [ 'T24_LPC', 'T30_HPC',
       'T50_LPT', 'P2_FIP', 'P15_PBY', 'P30_HPC', 'Nf_Fan_Speed',
       'Nc_Core_Speed',  'Ps30_Sta_Pres', 'phi_fp_Ps30',
       'NRf_cor_fan_sp', 'NRc_cor_core_sp', 'BPR_Bypass_rat', 'farB_f_air_rat',
       'htBleed_Enthalpy',  'PCNfR_dmd', 'W31_HPT_cool_bl',
       'W32_LPT_cool_bl' ]



Xall.head()
### Fit baseline  the model  ####
model.fit(Xall, y)
####          ####   

x = range(0,20631)     # create a range for plotting
y_hat_raw = model.predict(Xall)   # create the first round of y_hats # these will be logs 
y_hat = np.e**y_hat_raw   ### Convert y_hat back to # of cycles remaining

fig, ax = plt.subplots(figsize=(14, 4))
residual_plot(ax, x, y, y_hat)
plt.show()


fig, ax = plt.subplots(figsize=(14, 4))
residual_plot_scatter(ax, y, y_hat_raw)
plt.show()
#########      Score     ###  
score_er(y_hat_raw, y)                    # 0.4973549654394218  this is the other score ##  2.801590390955396
                                        # best score with y as a log and y_hat as a log is 0.14387745775561073   # but is this a supprise? 
##     0.13788048628   using only select features    
##     0.12711778739737728   score 
# col = [ 'T24_LPC', 'T30_HPC',
#        'T50_LPT', 'P2_FIP', 'P15_PBY', 'P30_HPC', 'Nf_Fan_Speed',
#        'Nc_Core_Speed',  'Ps30_Sta_Pres', 'phi_fp_Ps30',
#        'NRf_cor_fan_sp', 'NRc_cor_core_sp', 'BPR_Bypass_rat', 'farB_f_air_rat',
#        'htBleed_Enthalpy',  'PCNfR_dmd', 'W31_HPT_cool_bl',
#        'W32_LPT_cool_bl' ]

#### Show the model coefficients   #######
coef_dict = {}
for coef, feat in zip(model.coef_,Xall.columns):
    coef_dict[feat] = coef

coef_dict

################################################

###   Run this on the test data set    #########



#### #### #### ### #### ### #### #### ##### #### ##### ##

def plot_one_univariate(ax, var_name, mask=None, bootstrap=100):
    if mask is None:
        plot_univariate_smooth(
            ax,
            X[var_name].values.reshape(-1, 1), 
            X['cycles_to_fail'],
            bootstrap=bootstrap)
    else:
        plot_univariate_smooth(
            ax,
            X[var_name].values.reshape(-1, 1), 
            X['cycles_to_fail'],
            mask=mask,
            bootstrap=bootstrap)

#################   Plot the univariate effect ###########
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
univariate_plot_names = Xfeatures

for name, ax in zip(univariate_plot_names, axs.flatten()):
    plot_univariate_smooth(ax,
                           df[["Unit#" == 1][df[name]].values.reshape(-1, 1),
                           df['cycles_to_fail'],
                           bootstrap=100)
    ax.set_title(name)
plt.show()


#######################     Plot one item    #######
Xall.columns

for col in Xall.columns:
    fig, ax = plt.subplots(figsize=(12, 3))
    plt.scatter(y, Xall[col], alpha = .1)
    ax.set_title(col)
    plt.show()

# Pull out the columns that are numeric and will be standardized in the model......
 
 ####    Linear spline #######
for col in Xall.columns:
    linear_spline_transformer = LinearSpline(knots=[0, .70 , 1.5, 2, 2.5 , 3, 4, 5])
    linear_spline_transformer.transform(Xall[col]).head()

col = ['T24_LPC', 'T30_HPC', 'T50_LPT', 'P2_FIP', 'P15_PBY', 'P30_HPC',
       'Nf_Fan_Speed', 'Nc_Core_Speed', 'Ps30_Sta_Pres', 'phi_fp_Ps30',
       'NRf_cor_fan_sp', 'NRc_cor_core_sp', 'BPR_Bypass_rat', 'farB_f_air_rat',
       'htBleed_Enthalpy', 'PCNfR_dmd', 'W31_HPT_cool_bl', 'W32_LPT_cool_bl']

####    This will be the transformer for all.... 
# 
#    
#### this will name all of the variables and create the splines for all the models  ####### 
for col in Xall.columns:
    locals()[col + '_fit'] = Pipeline([
        (col, ColumnSelector(name=col)),
        (col + '_spline', LinearSpline(knots=[0, .70 , 1.5, 2, 2.5 , 3, 4, 5]))
    ])

#### This is the plot for the W32 variable #####

fig, ax = plt.subplots(figsize=(12, 3))
plt.scatter(y, X["W32_LPT_cool_bl"], alpha = .1)
ax.set_title("W32_LPT_cool_bl")
plt.show()

#########

linear_spline_transformer = LinearSpline(knots=[0, .70 , 1.5, 2, 2.5 , 3, 4, 5])

linear_spline_transformer.transform(X['W32_LPT_cool_bl']).head()

####    This will be the transformer for the t50_LPT    
W32_fit = Pipeline([
    ('W32_LPT_cool_bl', ColumnSelector(name='W32_LPT_cool_bl')),
    ('W32_spline', LinearSpline(knots=[0, .70 , 1.5, 2, 2.5 , 3, 4, 5]))
    ])

fig, ax = plt.subplots(figsize=(12, 3))
plt.scatter(y, X["BPR_Bypass_rat"], alpha = .1)
ax.set_title("BPR_Bypass_rat")
plt.show()

#########

linear_spline_transformer = LinearSpline(knots=[0, .70 , 1.5, 2, 2.5 , 3, 4, 5])

linear_spline_transformer.transform(X['BPR_Bypass_rat']).head()

####    This will be the transformer for the BPR     
BPR_fit = Pipeline([
    ('BPR_Bypass_rat', ColumnSelector(name='BPR_Bypass_rat')),
    ('BPR_Bypass_spline', LinearSpline(knots=[0, .70 , 1.5, 2, 2.5 , 3, 4, 5]))
    ])
######             ########           ######
linear_spline_transformer.transform(X['BPR_Bypass_rat']).head()




       ####### Start here    ########
fig, ax = plt.subplots(figsize=(12, 3))
plt.scatter(y, X["P30_HPC"], alpha = .1)
ax.set_title("P30_HPC")
plt.show()

#########

linear_spline_transformer = LinearSpline(knots=[0, .70 , 1.5, 2, 2.5 , 3, 4, 5])

linear_spline_transformer.transform(X['P30_HPC']).head()

####    This will be the transformer for the BPR     
P30_fit = Pipeline([
    ('P30_HPC', ColumnSelector(name='P30_HPC')),
    ('P30_HPC_spline', LinearSpline(knots=[0, .70 , 1.5, 2, 2.5 , 3, 4, 5]))
    ])
######             ########           ######



####   End feature selection and tuning  ######
for col in Xall.columns:
    linear_spline_transformer.transform(Xall[col]).head()

# ######            PIPLINE FOR EACH CONT. FEATURE      #####

col =  [ 'T24_LPC', 'T30_HPC', 'T50_LPT', 'P2_FIP', 'P15_PBY', 'P30_HPC',
       'Nf_Fan_Speed', 'Nc_Core_Speed', 'Ps30_Sta_Pres', 'phi_fp_Ps30',
       'NRf_cor_fan_sp', 'NRc_cor_core_sp', 'BPR_Bypass_rat', 'farB_f_air_rat',
       'htBleed_Enthalpy', 'PCNfR_dmd', 'W31_HPT_cool_bl', 'W32_LPT_cool_bl']



# FEAUTRE UNTION 
feature_pipeline_scaled = Pipeline([
    ('continuous_features', FeatureUnion([
    ("T50_LPT", T50_LPT_fit),
    ("T30_HPC", T30_HPC_fit),
    ("P2_FIP", P2_FIP_fit),
    ("T24_LPC" , T24_LPC_fit),
    ("P15_PBY" , P15_PBY_fit),  
    ("P30_HPC", P30_HPC_fit),
    ("Nf_Fan_Speed", Nf_Fan_Speed_fit),
    ("Nc_Core_Speed", Nc_Core_Speed_fit),
    ("Ps30_Sta_Pres" , Ps30_Sta_Pres_fit),
    ("phi_fp_Ps30", phi_fp_Ps30_fit),
    ("NRf_cor_fan_sp", NRf_cor_fan_sp_fit),
    ("BPR_Bypass_rat", BPR_Bypass_rat_fit),
    ("farB_f_air_rat" , farB_f_air_rat_fit),
    ("htBleed_Enthalpy" , htBleed_Enthalpy_fit),
    ("PCNfR_dmd", PCNfR_dmd_fit),
    ("W31_HPT_cool_bl", W31_HPT_cool_bl_fit),
    ("W32_LPT_cool_bl", W32_LPT_cool_bl_fit)])),
    ('standardizer', StandardScaler() )
])

########################################################
####### Make these work ################################
feature_pipeline_scaled.fit(Xall)
features = feature_pipeline_scaled.transform(Xall)
features.values
model = LinearRegression(fit_intercept=False)
model.fit(features.values, y=concrete['compressive_strength'])
model.predict(features.values)
display_coef(model, features.columns)


##########################################################
##########################################################

####     This might fit the model    #####   
feature_pipeline_scaled.fit(Xall)
X_features = feature_pipeline_scaled.transform(Xall)
X_features.shape

display_coef(model, Xall.columns)
