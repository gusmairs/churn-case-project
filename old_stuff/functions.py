import numpy as np
import pandas as pd

#This is the start of our collection of fuctions


#Transform feature Classes

    
    
    #target
    
    
    #sub populations
    
    
    #total miles
    
    
    
    #NaNs
    
    
#feature Union    

    
    
#Models
from sklearn.preprocessing import OneHotEncoder as onehot

# This is the start of our collection of functions and classes

#
# Transformer Classes
#
# Target -- Create both DaysAgo and Churned features
#
class DaysAgo():
    '''
    Transformer creates a days_ago feature for the last trip taken
    Start date is 7/1/2014, the day data was pulled
    In: The 'last_trip_date' feature as a datetime object
    Out: New feature 'days_ago' as an integer
    '''

    def __init__(self):
        
        pass

    def fit(self):
        '''
        Does this need to do anything?
        '''
        return self

    def transform(self, dates):
        '''Create the new feature'''
        days_ago = max(dates) - dates
        days_ago = days_ago.apply(lambda x: x.days)
        return days_ago

class Churned():
    '''
    Transformer creates the target feature
    In: The 'last_trip_date' feature as a datetime object
    Out: New feature 'churned' as 1-0 data
    '''

    def __init__(self):
        pass

    def fit(self):
        '''
        Does this need to do anything?
        '''
        return self

    def transform(self, dates):
        '''Create the new feature'''
        days_ago = max(dates) - dates
        days_ago = days_ago.apply(lambda x: x.days)
        return (days_ago > 30).astype(int)

class OneHotEncoder:
    """A transformer for one-hot encoding a categorical feature.
    
    Attributes:
    levels: np.array
      The unique levels in the feature.  Computed when `fit` is called.
    """
    def __init__(self):
        self.levels = None

    def fit(self, X, *args, **kwargs):
        """Memorize the levels in the categorical array X.

        Parameters
        ----------
        X: pd.Series
          A pandas Series containing categorical data.
        """
        self.levels = X.unique()
        return self

    def transform(self, X, **transform_params):
        """One-hot-encode an array.

        Parameters
        ----------
        X: pd.Series
          A pandas Series containing categorical data.

        Returns
        -------
        encoded: pd.DataFrame
          A two dimensional pandas DataFrame containing the one-hot-encoding of X.
        """
        result = np.zeros((X.size,len(self.levels)))
        name = X.name
        column_names = []
        columns = self._compute_column_names()
        for i in range(len(self.levels)):
            result[:,i] = (X==self.levels[i]).astype(int)
            column_names.append(name + '_' + columns[i])
        return pd.DataFrame(result, columns=column_names)

    def _compute_column_names(self):
        columns = []
        for lvl in self.levels:
           columns.append('is_' + lvl)
        return np.asarray(columns, dtype='<U9')

#
# LuxuryBoolean
#
class LuxuryBoolean():
    '''
    Transformer makes 'luxury_car_user' a 0-1 integer
    In: The 'luxury_car_user' feature boolean object
    Out: New feature 'is_luxury_user' as 1-0 data
    '''

    def __init__(self):
        pass

    def fit(self):
        '''
        Does this need to do anything?
        '''
        return self

    def transform(self, bool):
        '''Create the new feature'''
        lux = bool.apply(lambda x: x is True).astype(int)
        return lux

#
# Total miles
#
class MilesPer30Days():
    """Transformer Class
            Input:
                [DataFrame] [['trips_in_first_30_days', 'avg_dist']]
            Calculation:
                trips_in_first_30_days * average_distance
            Returns:
                [DataFrame] ['miles_per_30d']
    """
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, dataframe):
        # [DataFrame] [['trips_in_first_30_days', 'avg_dist']]
        df_copy = dataframe.copy()
        df_miles_per_30d = df_copy['trips_in_first_30_days'] * df_copy['avg_dist']
        return df_miles_per_30d

#
# NaNs
#
# Phone
#
class PhoneNan():
    '''
    Transformer makes 'phone' a group of one-hot features
    In: The 'phone' feature categorical object (396 NaNs)
    Out: New features to be named 'phone_is_nan', 'phone_is_android', 'phone_is_apple'
    '''

    def __init__(self):
        # self.levels = None
        pass

    def fit(self):
        '''
        Does this need to do anything?
        '''
        return self

    def transform(self, phn):
        '''Create the new feature'''
        nan = phn.isna().astype(int)
        android = (phn == 'Android').astype(int)
        apple = (phn == 'iPhone').astype(int)
        return nan, android, apple


# This is Alex's test

### Steven's log loss function  ########

def log_loss_score( y_hats, y_act):
    return -np.sum(  ( ((y_act) * np.log (1 + y_hats)) + (( 1 - y_act) * np.log (1 + y_hats)   )  )) / (len(y_hats))


def NullsToMean(col):
    '''input: column with NaNs
        output: column with Means'''
    df = pd.DataFrame(traindata[col])
    #this makes the array trues and falses depending on if the value is NaN
    maskdf = traindata[col].isnull()
    df['is_NaN'] = maskdf
    return df.fillna(np.mean(df[col])).head() #df.drop(['avg_rating_of_driver'], axis = 1)


def NullsToMean(col):
    '''input: column with NaNs
        output: column with Means'''
    df = pd.DataFrame(traindata[col])
    #this makes the array trues and falses depending on if the value is NaN
    maskdf = traindata[col].isnull()
    df['is_NaN'] = maskdf
    return df.fillna(np.mean(df[col])).head() #df.drop(['avg_rating_of_driver'], axis = 1)
