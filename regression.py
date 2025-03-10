### Install dependencie --------------------------------------------------------
# pip install sklearn
# pip install statsmodels

### Dependencie ----------------------------------------------------------------
import pandas as pd # used for working with data sets
import numpy as np # used for working with arrays
import matplotlib.pyplot as plt # used for plotting
 
from sklearn.model_selection import train_test_split # for split of data into training and testing part
from sklearn.linear_model import LinearRegression # linear regression model
from sklearn.metrics import mean_squared_error # MSE

import statsmodels.api as sm # More detailed statistical approach to linear regression

### Data insurance -------------------------------------------------------------
# source: https://github.com/stedy/Machine-Learning-with-R-datasets
# path: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
# Individual medical costs billed by health insurance

path_to_data = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv" 

data_insurance = pd.read_csv(path_to_data)

# View data
data_insurance
data_insurance.describe(include='all') 

data_insurance.dtypes

# Categorical data
data_insurance["region"] = data_insurance["region"].astype("category")
data_insurance["region"].dtype

data_insurance["sex"] = data_insurance["sex"].astype("category")
data_insurance["sex"].dtype

data_insurance["smoker"] = data_insurance["smoker"].astype("category")
data_insurance["smoker"].dtype

# Dummy variables
data_insurance_dummy = pd.get_dummies(data = data_insurance, drop_first = True)
data_insurance_dummy

### Prepare data ---------------------------------------------------------------
# Split the dataset into training and testing part
# Set seed for pseudorandom number generator for reproducibility
# Assign response and explanatroy variables
seed = 500

x_train, x_test, y_train, y_test = train_test_split(
    data_insurance_dummy.drop(['charges'], axis = 1), # explanatory variable
    data_insurance_dummy[['charges']], # response variable
    test_size = 0.2, # hold out size
    random_state = seed
    )

# Types
type(x_train)
x_train.dtypes
y_train.dtypes

# Changes
data_insurance.describe(include='all') 
x_train.describe()
x_test.describe()

y_train.hist()
y_test.hist()

### Model_1 --------------------------------------------------------------------
# Fitting linear regression model
# Model dependence of variable charges on age of policy holder
# Write equation of estimated model 
# Y = ... , where y = charges

plt.plot(data_insurance.age, data_insurance.charges, 'o')

# Create linear regression model 
x_train_1 = np.array(x_train.loc[:,'age']).reshape(-1,1)
x_test_1 =  np.array(x_test.loc[:,'age']).reshape(-1,1)

# Fit model
model_1 = LinearRegression().fit(x_train_1, y_train)

# Model coefficients
model_1.intercept_
model_1.coef_

# Add line to previous plot
plt.plot(data_insurance.age, data_insurance.charges, 'o')
plt.plot([20, 60], [3364 + 256*20, 3364 + 256*60], 'k-')

# R2 score
model_1.score(x_train_1, y_train)

# Make predictions on the test data
y_pred = model_1.predict(x_test_1)

# Calculate MSE for test data - manually
(y_pred - y_test)
np.square((y_pred - y_test))
np.square((y_pred - y_test)).sum()
np.square((y_pred - y_test)).sum()/len(y_test.index)

# Calculate MSE for test data - using library sklearn
mse = mean_squared_error(y_test, y_pred)

rmse = mse**(1/2)

# save final MSE 
mse_model_1 = mse

### More detailed modelling with statsmodels ------------------------------------
# Prepare data
x_train_1_const = sm.add_constant(x_train_1)
x_test_1_const = sm.add_constant(x_test_1)

# Fit model
model_1_stats = sm.OLS(y_train, x_train_1_const)
fit_1 = model_1_stats.fit()

# See results
fit_1.summary()

# Make prediction
fit_1.predict(x_test_1_const)

### Model_2 --------------------------------------------------------------------
# Repeat process of modelling for variable charges as dependant and all other 
# variables from dataset as independent variables

# Fit model
model_2 = LinearRegression(fit_intercept = True).fit(x_train, y_train)

# Model coefficients
model_2.coef_
model_2.intercept_

# R2 score

# Make predictions on the test data
y_pred = model_2.predict(x_test)
# Calculate MSE for test data - manually

# Calculate MSE for test data - using library sklearn
mse = mean_squared_error(y_test, y_pred)
# save final MSE 
mse_model_2 = mse

# Compare both models 
mse_model_1
mse_model_2
