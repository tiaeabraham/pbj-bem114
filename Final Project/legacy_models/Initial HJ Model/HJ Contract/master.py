'''
I want one file that will run all the models and return the results.

Compare old spread to new spread performance

How much of a difference does the predicted price need to be form the actual price to make a trade?
 - definitely some modeling that can be done to optimize this
 
 - Sharpe ratio, risk adjusted return
'''

### <PACKAGE IMPORTSL> ###
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
### </PACKAGE IMPORTS> ###

### <MODEL IMPORTSL> ###
# from spread_performance_SGD import evaluate_spread_model
from spread_performance_HMM import evaluate_spread_model
from sizing_performance import evaluate_sizing_model
#### </MODEL IMPORTSL> ###

# Load data
data_path = '/Users/bramschork/Desktop/data.csv'
df = pd.read_csv(data_path)

### <EVALUATE SPREAD MODEL> ###
# SGDRegressor Vars
'''max_iter = 1000
tol = 1e-3
random_state = 42
backtest_period = 26
bet_size = 100

evaluate_spread_model(
    df, max_iter, tol, random_state, backtest_period, bet_size)'''

n_components = 3  # Number of hidden states in the HMM
covariance_type = 'diag'  # Type of covariance to use in the HMM
random_state = 42
backtest_period = 26
bet_size = 100

evaluate_spread_model(
    df, n_components, covariance_type, random_state, backtest_period, bet_size)


### </EVALUATE SPREAD MODEL> ###

# Evaluate Sizing Model
