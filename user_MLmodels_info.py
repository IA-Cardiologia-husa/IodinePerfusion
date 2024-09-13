# In this archive we have to define the dictionary ml_info. This is a dictionary of dictionaries, that for each of the ML models we want
# assigns a dictionary that contains:
#
# clf: a scikit-learn classifier, or any object that implements the functions fit, and predict_proba or decision_function in the same way.
# formal_name: name to be used in plots and report

import sklearn.linear_model as sk_lm
import sklearn.model_selection as sk_ms
import sklearn.pipeline as sk_pl
import sklearn.preprocessing as sk_pp
import sklearn.impute as sk_im
import xgboost as xgb
import numpy as np

ML_info ={}

grid_params_bt=[{'max_depth': [None, 2, 3, 5],
				 'reg_alpha':[10**n for n in range(-3,4)]}]
tuned_bt=sk_ms.GridSearchCV(xgb.XGBRegressor(n_estimators=100),grid_params_bt, cv=10,scoring ='r2', return_train_score=False, verbose=1)

ML_info['BT'] = {'formal_name': 'XGBoost',
				 'clf': tuned_bt}
				 

ML_info['Linear'] = {'formal_name': 'Linear Regression',
					'clf': sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median').set_output(transform="pandas")), 
												 ("std", sk_pp.StandardScaler()),
												 ("lr",sk_lm.LinearRegression())])}

pipeline_lasso = sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median').set_output(transform="pandas")), 
									   ("std", sk_pp.StandardScaler()),
									   ("ls",sk_lm.Lasso())])
grid_params_lasso=[{'ls__alpha':[10**n for n in np.arange(-5,-0.99,1/3)]}]
tuned_lasso=sk_ms.GridSearchCV(pipeline_lasso,grid_params_lasso, cv=10,scoring ='r2', return_train_score=False, verbose=1)

ML_info['HypLasso'] = {'formal_name': 'Lasso Regression (Hyperparameter Tuning)',
					   'clf': tuned_lasso}

pipeline_ridge = sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median').set_output(transform="pandas")), 
									   ("std", sk_pp.StandardScaler()),
									   ("rr",sk_lm.Ridge())])
grid_params_ridge=[{'rr__alpha':[10**n for n in np.arange(-1,3.99, 1/3)]}]
tuned_ridge=sk_ms.GridSearchCV(pipeline_ridge,grid_params_ridge, cv=10,scoring ='r2', return_train_score=False, verbose=1)

ML_info['HypRidge'] = {'formal_name': 'Ridge Regression (Hyperparameter Tuning)',
					   'clf': tuned_ridge}

