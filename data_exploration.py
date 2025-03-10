### Install dependencie --------------------------------------------------------
# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install seaborn

### Dependencie ----------------------------------------------------------------
import pandas as pd # used for working with data sets
import numpy as np # used for working with arrays
import matplotlib.pyplot as plt # used for plotting
import seaborn as sns # used for plotting, see examples at https://seaborn.pydata.org/examples/index.html
# ### Data insurance -------------------------------------------------------------
# source: https://github.com/stedy/Machine-Learning-with-R-datasets
# path: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
# Individual medical costs billed by health insurance

path_to_data = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv" 

data_insurance = pd.read_csv(path_to_data)
data_insurance

# View data
data_insurance
data_insurance.head(5)
data_insurance.tail(5)

# Get basic data characteristics 
data_insurance.info()
data_insurance.shape
data_insurance.describe()
data_insurance.describe(include='all') 

# List all columns
list(data_insurance.columns)

# Subset data
data_insurance["age"] # columns
data_insurance[0:2] # rows
data_insurance[5:] # rows

data_insurance["age"][0:1] # rows and columns
data_insurance.iloc[[1,2], [1,2]] # integer location
data_insurance.loc[[1,2], ["age", "sex"]] # "name" location
data_insurance.loc[[1,2], data_insurance.columns[3:5]]

# Number of elements in the array
data_insurance.size
data_insurance["age"].size

# Data types
type(data_insurance)
type(data_insurance["age"])

data_insurance.dtypes
data_insurance["age"].dtype
data_insurance.age.dtype

data_insurance = data_insurance.convert_dtypes()
data_insurance.dtypes

# Categorical data
data_insurance["region"].dtype
data_insurance["region"] = data_insurance["region"].astype("category")
data_insurance["region"].dtype

# Create categories from numerical variable
data_insurance["age_cat"] = pd.cut(
  data_insurance["age"],
  [0, 25, 50, 75],
  right=False)

data_insurance["age_cat"].dtype

# Build categories by range
print(list(range(0, 100, 10))) 

# Descriptive statistics
data_insurance.describe()
data_insurance.describe(include='all')

data_insurance["age"].max()
data_insurance["age"].min()
data_insurance["age"].mean()
data_insurance["age"].std()
data_insurance["age"].median()
data_insurance["age"].quantile([0.5])
data_insurance["age"].quantile([0.1, 0.3, 0.6, 0.9])

# Further statistics and values
data_insurance["sex"].count()
data_insurance["sex"].unique()
data_insurance['sex'].value_counts()

# Correlation
data_insurance.corr()

# What numbers don't tell
# https://www.autodesk.com/research/publications/same-stats-different-graphs

# Visualise! - package matplotlib, seaborn
# Basic plot - scatter plot

# How many dimensions of data can you show in common scatter plot?

# Statistical plots
# Histogram - shows frequency
data_insurance.hist(['age'])

# Boxplots - shows variability
data_insurance.boxplot(['age'])

# Pair plots
sns.pairplot(data_insurance)

# Correlogram
sns.heatmap(data_insurance.corr())

# For more see https://python-graph-gallery.com/

# Stay AWAY from pie plots unless you have something to hide
# https://www.data-to-viz.com/caveat/pie.html

# Data by groups
data_insurance.groupby(by=["region"], dropna=False).mean()
data_insurance.groupby(by=["region"], dropna=False).min()

# Filter data
data_new = data_insurance[(data_insurance.region == "northwest") & 
(data_insurance.sex == "male")]

# Creating the Second Dataframe using dictionary 
data_new = pd.DataFrame({"age":[30, 50, 60, 19], 
                        "sex":["male", "female", "male", "female"],
                        "bmi":[18, 26, 37, 27.9],
                        "children":[1, 20, 2, 0],
                        "smoker":[np.nan, np.nan, np.nan, "yes"],
                        "region":["southnorth", "southeast", "northwest", "southwest"],
                        "charges":[100, 16000, 27000, 16884.924],
                        "age_cat":[np.nan, np.nan, np.nan, "[0.0,25.0)"]}) 

# Create new incorrect data
data_incorrect = data_insurance.append(data_new, ignore_index=True) 

# What has changed?
data_incorrect["age"].mean()
data_incorrect["age"].median()
# What has changed?
data_incorrect["region"].count()
data_incorrect["region"].unique()

# Is null, na value
data_incorrect.isnull()
data_incorrect[data_incorrect.isna().any(axis=1)]

# If the value is northsouth, set it to southeast:
for x in data_incorrect.index:
  if data_incorrect.loc[x, "region"] == "northsouth":
    data_incorrect.loc[x, "region"] = "southeast"

# Delete rows where "region" is northsouth:
for x in data_incorrect.index:
  if data_incorrect.loc[x, "region"] == "northsouth":
    data_incorrect.drop(x, inplace = True)

# Remove rows with a NULL value in the "Date" column:
data_incorrect.dropna(subset=["smoker"], inplace = True)

# Removing Duplicates
data_incorrect.duplicated()
data_incorrect.drop_duplicates(inplace = True)


### Data about Australian weather ----------------------------------------------
# source: https://rdrr.io/github/grayskripko/rattle/man/weather.html
# path: weather_data.csv (in files on MS Teams)
# Weather observations from a number of locations around Australia

path_to_data = "c:/Users/filip/Documents/Resources/Výuka/DataX/2022_2023_ZS/weather_data.csv" 

data_weather = pd.read_csv(path_to_data, encoding = 'UTF-8')

# Hint
# Convert to date
# df['Date'] = pd.to_datetime(df['Date']) 

