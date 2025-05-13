import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os
from datetime import datetime
from datetime import timedelta
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, KFold, TimeSeriesSplit
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy
from scipy import stats

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# Function to calculate Sen's slope
def sens_slope(x, y, time_target='year'):
    
    n = len(x)
    slopes = []
    
    times_dict = {'day': 60 * 60 * 24,'month': 60 * 60 * 24 * 30.5,'year': 60 * 60 * 24 * 365.25}
    
    time_factor = times_dict[time_target]

    # Convert datetime objects to numeric values (e.g., days since the first date)
    x_numeric = (x - x.min()).dt.total_seconds()  # Convert to the number of seconds since the first date
    
    # Ensure that x and y are numpy arrays to allow proper indexing
    x = np.array(x_numeric)  # x should be a numeric array (days)
    y = np.array(y)  # y should be a numeric array (predictions)

    for i in range(n):
        for j in range(i + 1, n):
            # Calculate slope between pairs of points
            slope = (y[j] - y[i]) / (x[j] - x[i])  # x_numeric is now numeric values (days)
            slopes.append(slope)

    median_slope = np.median(slopes) # slope in units per second

    # Compute intercepts using the median slope
    intercepts = y - median_slope * x
    median_intercept = np.median(intercepts)
    
    #Convert slope and itnercept to target time unit
    median_slope *= time_factor
    
    return median_slope, median_intercept

class ResidualCorrector(BaseEstimator, RegressorMixin):
    def __init__(self, base_model = None, residual_model = None):
        self.base_model = base_model
        self.residual_model = residual_model

    def fit(self, X, y):
        self.base_model_ = clone(self.base_model)
        self.base_model_.fit(X,y)
        residuals = y - self.base_model_.predict(X)

        self.residual_model_ = clone(self.residual_model)
        self.residual_model_.fit(X,residuals)
        return self
    
    def predict(self,X):
        base_pred = self.base_model_.predict(X)
        residual_pred = self.residual_model_.predict(X)
        return base_pred + residual_pred
