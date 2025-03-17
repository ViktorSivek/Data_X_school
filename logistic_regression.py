### Dependencie ----------------------------------------------------------------
import pandas as pd # used for working with data sets
import numpy as np # used for working with arrays
import matplotlib.pyplot as plt # used for plotting

# Usefull functions
from sklearn.model_selection import train_test_split # for split of data into training and testing part
import sklearn.metrics as metrics # Different evaluation metrics

# Used models
from sklearn.linear_model import LogisticRegression # Logistic regression

# Hint: To see all output from pandas outputs use:
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

### Data about Australian weather ----------------------------------------------
# source: https://rdrr.io/github/grayskripko/rattle/man/weather.html
# path: weather_data.csv (in files on MS Teams)
# Weather observations from a number of locations around Australia

path_to_data = "c:/Users/filip/Documents/Resources/Výuka/DataX/2020_2021_LS/weather_data.csv" 

data_weather = pd.read_csv(path_to_data, encoding = 'UTF-8')

# View data
data_weather
data_weather.describe(include='all') 

data_weather.dtypes

# Regression tasks - dependent variable (y) - numerical
# Classification tasks - dependent variable (y) - categorical

### Data wrangling ------------------------------------------------------------
# Categorical data
data_weather["RainTomorrow"] = data_weather["RainTomorrow"].astype("category")
data_weather["RainTomorrow"].dtype

data_weather["RainToday"] = data_weather["RainToday"].astype("category")
data_weather["RainToday"].dtype

data_weather["WindGustDir"] = data_weather["WindGustDir"].astype("category")
data_weather["WindGustDir"].dtype

data_weather["WindDir9am"] = data_weather["WindDir9am"].astype("category")
data_weather["WindDir9am"].dtype

data_weather["WindDir3pm"] = data_weather["WindDir3pm"].astype("category")
data_weather["WindDir3pm"].dtype

# Convert Date to date variable
data_weather['Date'] = pd.to_datetime(data_weather['Date']) 
data_weather['Date']

data_weather.dtypes

# Dummy variables
data_weather_dummy = pd.get_dummies(data = data_weather, drop_first = True)
data_weather_dummy

# Meaningful feature selection
# What represents Date, Location, RISK_MM

# More sofisticated approach to to column exclusion
data_weather.columns.isin(['Date', 'Location', 'RISK_MM'])
~data_weather.columns.isin(['Date', 'Location', 'RISK_MM'])

data_weather_short = data_weather.loc[ : , ~data_weather.columns.isin(['Date', 'Location', 'RISK_MM'])] 

# Dummy variables
data_weather_dummy = pd.get_dummies(data = data_weather_short, drop_first = True)
data_weather_dummy

# What about categorical variables and number of categories
data_weather['WindGustDir'].unique().tolist()
# Manual change of categories - takes a lot of writing 
# Goal is to get only 4 main wind directions North, South, West and East
data_weather['WindGustDir'] = data_weather['WindGustDir'].replace({'NW':'N'})
# More sofisticated approach
data_weather.loc[data_weather['WindGustDir'].str.startswith('N').fillna(False), 'WindGustDir'] = 'N'
data_weather.loc[data_weather['WindGustDir'].str.startswith('S').fillna(False), 'WindGustDir'] = 'S'
data_weather.loc[data_weather['WindGustDir'].str.startswith('E').fillna(False), 'WindGustDir'] = 'E'
data_weather.loc[data_weather['WindGustDir'].str.startswith('W').fillna(False), 'WindGustDir'] = 'W'

data_weather['WindGustDir']
data_weather["WindGustDir"] = data_weather["WindGustDir"].astype("object").astype("category")
data_weather["WindGustDir"].dtype

# Dummy variables
data_weather_short = data_weather.loc[ : , ~data_weather.columns.isin(['Date', 'Location', 'RISK_MM', 'WindDir9am', 'WindDir3pm'])] 

data_weather_dummy = pd.get_dummies(data = data_weather_short, drop_first = True)
data_weather_dummy = data_weather_dummy.dropna()
data_weather_dummy

### Prepare data ---------------------------------------------------------------
# Split the dataset into training and testing part
# Set seed for pseudorandom number generator for reproducibility
# Assign response and explanatroy variables
seed = 500

x_train, x_test, y_train, y_test = train_test_split(
    data_weather_dummy.drop(['RainTomorrow_Yes'], axis = 1), # explanatory variable
    data_weather_dummy[['RainTomorrow_Yes']], # response variable
    test_size = 0.2, # hold out size
    random_state = seed
    )

# Differences between train and test data
x_train.describe()
x_test.describe()

y_train.hist()
y_test.hist()

### Logistic regression --------------------------------------------------------
# Binary logistic regression / Multinomic logistic regression

logistic_model = LogisticRegression(
  penalty = 'none', # {'l1', 'l2', 'elasticnet', 'none'}
  fit_intercept = True,
  random_state = seed) 

# Fit model to our training dataset
logistic_model.fit(x_train, y_train)

# See coefficients and compare them with other models
logistic_model.classes_
logistic_model.coef_
logistic_model.intercept_

# Score method to get accuracy of model ((TP+TN)/total)
logistic_model.score(x_train, y_train)

# Make predictions on the test data
y_pred = logistic_model.predict(x_test)

# Calculate confusion matrix (actual = rows (0,1), predicted = columns (0,1))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix

# Calculate further evaluation metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # ((TP+TN)/total) = (7+52)/(52+4+8+7)
print("Precision:",metrics.precision_score(y_test, y_pred)) # (TP/(TP + FP)) = (7)/(7+4)
print("Recall:",metrics.recall_score(y_test, y_pred)) # (TP/(TP + FN)) = (7)/(7+8)

# Complete evaluation metrics summary
metrics.classification_report(y_test, y_pred)

# AUC and ROC curve
# Prediction probabilities, True Positive Rate (TPR) and False Positive Rate (FPR) for different thresholds
y_pred_proba = logistic_model.predict_proba(x_test)[::,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)

# AUC - "Area under the ROC Curve" 
# Aggregate measure of performance across all possible classification thresholds
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# ROC curve - Performance of a classification model at all classification thresholds.
# X axes - TPR (TP/(TP + FN)), Y axes - FPR (FP/(FP + TN))
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

# Play around with decesion threshold 
# Are people more happy when it should not rain and it is actually raining?
threshold = 0.1
decisions = (y_pred_proba >= threshold).astype(int)

# Final notes on logistic regression ---------------------------------------------

# Simple model that is interpretable and provides a probability score for observations.
# Doesn't require scaling of features, but is not able to handle a large number of categorical features/variables.
# Easily overfits and can't solve non-linear problems (possibly solved by transformations in model)
# Bad performance with independent variables that are not correlated to the target variable and are very similar or correlated to each other.
