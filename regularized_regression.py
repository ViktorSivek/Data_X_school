### Dependencie ----------------------------------------------------------------
import pandas as pd # used for working with data sets
import numpy as np # used for working with arrays
import matplotlib.pyplot as plt # used for plotting

# Usefull functions
from sklearn.model_selection import train_test_split # for split of data into training and testing part
from sklearn.metrics import mean_squared_error # MSE

# Used models
from sklearn.linear_model import Ridge # Ridge regression 
from sklearn.linear_model import Lasso # Lasso regression
from sklearn.linear_model import ElasticNet # Elasticnet regression

### Data and previous linear regression results  -------------------------------
# Continue with environment from previous lecture - dataset - Insurance.csv
# -> run Regression.py
# make sure you have mse_model_1 and mse_model_2 avaliable for comparison

### Model_regularised_regression -----------------------------------------------

# In presentations, book and R (glmnet package)

# Alpha = 0 - ridge regression - coefficient restriction - L2 penalization
# Aplha = 1 - LASSO regression - variable selection - L1 penalization
# Aplha = (0,1) - elasticnet regression - combination of L1 and L2

# Lambda - coefficient penalization strength
# Lambda = 0 -> OLS regression, lambda = infinity -> constant

# In sci-kit learn l1_ratio is "our" Alpha and Alpha is "our" Lambda

### Ridge model definition -----------------------------------------------------
ridge_model = Ridge(
  alpha = 6, 
  fit_intercept = True,
  # normalize = False, 
  random_state = seed) 

# Fit model to our training dataset
ridge_model.fit(x_train, y_train)

# See coefficients - how many of them do you expect?
# Compare them with model_2 coefficients
ridge_model.coef_
ridge_model.intercept_

model_2.coef_
model_2.intercept_

# Hyper parameters: l1_ratio and alpha
# Need to be set by modeller - it's your decesion!

# R2 score
ridge_model.score(x_train, y_train)

# Make predictions on the test data
y_pred = ridge_model.predict(x_test)

# Calculate MSE for test data 
mse_model_ridge = mean_squared_error(y_test, y_pred)
mse_model_2

### LASSO model definition -----------------------------------------------------
lasso_model = Lasso(
  alpha = 6, 
  fit_intercept = True,
  # normalize = False, 
  random_state = seed) 

# Fit model to our training dataset
lasso_model.fit(x_train, y_train)

# See coefficients and compare them with other models
lasso_model.coef_
lasso_model.intercept_

# R2 score
lasso_model.score(x_train, y_train)

# Make predictions on the test data
y_pred = lasso_model.predict(x_test)

# Calculate MSE for test data 
mse_model_lasso = mean_squared_error(y_test, y_pred)


### Elasticnet model definition ------------------------------------------------
elnet_model = ElasticNet(
  l1_ratio = 0.5, 
  alpha = 6, 
  fit_intercept = True,
  # normalize = False, 
  random_state = seed) 

# Fit model to our training dataset
elnet_model.fit(x_train, y_train)

# See coefficients and compare them with other models
elnet_model.coef_
elnet_model.intercept_

# R2 score
elnet_model.score(x_train, y_train)

# Make predictions on the test data
y_pred = elnet_model.predict(x_test)

# Calculate MSE for test data 
mse_model_elnet = mean_squared_error(y_test, y_pred)

mse_model_2
mse_model_ridge
mse_model_lasso
mse_model_elnet


# Final notes on linear regression ---------------------------------------------

# Results interpretation:
# Coefficients in multivariate linear models represent the dependency between 
# a given feature and the target, conditional on the other features.

# Coefficients vs Feature importance:
# Coefficients must be scaled to the same unit of measure to retrieve feature importance!
# In such case linear regression coefficients are the feature importance

# Cross validation:
# Correlated features induce instabilities in the coefficients of linear models - use crossvalidation
# Note that versions of each of used sci-kit learn model exists also in CV verison, 
# e.g.: linear_model.Ridge() -> linear_model.RidgeCV()
# 
# Answer to setting "correct" hyper parameters
