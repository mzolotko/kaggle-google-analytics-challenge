import pandas as pd


# rolling window parameters for generation of time windows
global_params = dict(
global_rev_var='totals_transactionRevenue',   
num_last_revenues=10,
present_delta=500,
future_delta=108,
num_windows=8,
abs_first_day=pd.to_datetime('2016-08-01'),
abs_last_day=pd.to_datetime('2018-10-16')
)


# Light GBM training parameters
lgb_params = {
        'boosting_type': 'gbdt', 
        'colsample_bytree': 0.7000000000000001,   # feature_fraction
        'learning_rate': 0.029840198781320736,
        'max_depth': 8, 
        'metric': 'root_mean_squared_error', 
        'min_split_gain': 0.001623776739188721, # min_gain_to_split
        'n_jobs': 4, 
        'num_boost_round': 1000,   # num_iterations
        'num_leaves': 32, 
        'objective': 'root_mean_squared_error',   # regression
        'reg_alpha': 1.438449888287663,    # lambda_l1
        'reg_lambda': 0.006158482110660267, # lambda_l2
        'seed': 13, 
        'subsample': 0.8,    # bagging_fraction
        'early_stopping_rounds': 200
        }