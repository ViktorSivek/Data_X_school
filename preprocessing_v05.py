#!/usr/bin/env python
# coding: utf-8

# # Basics of ML preprocessing
# 
# ## Data
# Source: [https://www.kaggle.com/datasets/mirichoi0218/insurance]
# Individual medical costs billed by health insurance
# EDA: e.g [https://datauab.github.io/medical_insurance/]
# 
# - Popular dataset
# - Relatively small sample (certainly not big data)
# - Categorical as well as numerical features
# - Nonlinearities
# - Generated data with some unrealic patterns
# - Main purpose is to demonstarte basic ideas and syntax.
# - Within the class time.
# - Codes and applications are developed for teaching purposes. Therefore inefficiences occur and only aspects relevatn to the topic taught are mentioned.

# In[1]:


# get_ipython().system('pip install category_encoders')


# In[2]:


### Dependencies----------------------------------------------------------------
import pandas as pd #data frames and other stuff
import pickle # to save data in binary file

# Usefull functions
#from sklearn.preprocessing import KBinsDiscretizer #for binning variables
from sklearn.model_selection import train_test_split #for split into training and testing part
from category_encoders import TargetEncoder #for target (mean) encoding
from sklearn.preprocessing import MinMaxScaler #for normalization


# # Load the Pandas libraries with alias 'pd'

# In[5]:


#Local paths
path_csv_file = '../data/insurance.csv' #Input csv data
path_train_test_file = '../data/insurance_train_test.pkl' #Output pickle data

#Google drive paths
# file_id =  '1tP_FVn4s4IQkD3qXL5oI35UFVBu7K44X' #insurance.csv
# path_csv_file = 'https://drive.google.com/uc?id={}'.format(file_id)


# In[17]:


data_insurance = pd.read_csv(path_csv_file)

#Some basic display / overview functions
print(data_insurance.head) #prints begiining of file
print(data_insurance.describe(include='all')) #basic summary of the data frame (if 'all' not stated, only metric are described)
print(data_insurance.dtypes) #shows data types of columns of the data frame


# - Categorical data are imported as object. Make sure that all numerical variables are imported as numerical (integers or floats). If not, search...:-( Typically, this is decimal separator, blank spaces, string codes for NaNs etc...
# - Similarly with date variables (not present here)
# - You may convert the formats. For example it's a good practice to typecast categorical features to a category dtype because they make the operations on such columns much faster than the object dtype.  [https://www.datacamp.com/community/tutorials/categorical-data]
# 

# In[7]:


#Convert variables of the type object to category type
for col in data_insurance.select_dtypes(include='object').columns:
    data_insurance[col] = data_insurance[col].astype('category')

#Check
data_insurance.dtypes


# In[8]:


#Find binary variables
binary_vars = [col for col in data_insurance.columns if data_insurance[col].nunique() == 2]
print('Binary variables:',binary_vars)

#Find other categorical variables
multicategorical_vars = [col for col in data_insurance.select_dtypes(include='category').columns if col not in binary_vars]
print('Multi categorical variables:', multicategorical_vars)


# ## Apply one hot encoding
# - There are 2 popular options: Scikit-learn OneHotEncoder or Pandas has get_dummies
# - We use Pandas has get_dummies
# - See [https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02] for details dn scikit option.
# - For multi categorical variables dummy encoding is used with drop_first=False as it is more transparent and easier to explain
# - For binomial variables drop_first=True is selected as more transparent and efficient option.
# - Original variables are replaced by dummies.

# In[9]:


#Make a copy of the original region column
data_insurance['region_original'] = data_insurance['region'].copy()

#One hot encoding for binary variables
data_insurance = pd.get_dummies(data_insurance, columns=binary_vars, drop_first=True)
data_insurance = pd.get_dummies(data_insurance, columns=multicategorical_vars, drop_first=False)
data_insurance.head()


# ## Ordinal encoding and discretization
# - This is just to show the syntax.
# - We will not use these features in further calculations.
# - Discretize age using pd.cut() function
# - Encode with numerical code (Preserving ordinality) using the map() fucntion

# In[10]:


#Discretize age using pd.cut
data_insurance['age_binned']=pd.cut(
    x=data_insurance['age'],
    bins=[0,30,45,200],
    labels=["young", "middle", "not young"]
    )

#Display number of bins
print(data_insurance['age_binned'].value_counts())

# Alternativelly use sklearn for more sophisticated binning strategies
##instantiate binner to cut
#cut_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
##apply binner on age data
#age_binned = cut_bins.fit_transform(data_insurance[["age"]])


# In[11]:


#ordinal encoding of age variable
age_disc_dict= {
    "young":1,
    "middle":2,
    "not young":3
    }
data_insurance['age_binned_oe'] = data_insurance['age_binned'].map(age_disc_dict)
data_insurance[['age', 'age_binned', 'age_binned_oe']].head()


# ## Train test split
# - Use the sklearn.model_selection import train_test_split() function to split the train and test file.
# - Note that the function allows spliting only to two subsets. To create also validation set, you can use it twice consecutivelly.
# - Fix the seed of the generator to SEED = 500 to have consistent results in the class but even more importantnly to be able to replicate your calculation anytime in the future.
# -
# 

# In[12]:


SEED = 500
X_train, X_test, y_train, y_test= train_test_split(
    data_insurance.drop(['charges'], axis = 1), #explanatory
    data_insurance[['charges']], #response
    test_size=0.2, #hold out size
    random_state=SEED
    )


# ## Target encoder
# - As an alternative to one hot encoding of the multicategorial variable, is the target encoding.
# - In target encoding, the feature is transformed to a numerical variable using some conditional transformation of the target variable.
# - Mean encoding uses conditional mean of the target variable.
# - Unlike in one hot encoding, the output is just one feature.
# - Do not forget that training must be performed only on training data otherwise data leakage occurs

# In[13]:


#Use target encoder for mutlicategorical variables
#pip install category_encoders
encoder = TargetEncoder() #instantiate target encoder
encoder.fit(X_train['region_original'], y_train) #calculate means of target on training data (=fit the encoder)
X_train['region_me'] = encoder.transform(X_train['region_original']) #map the training data means to training data
X_test['region_me'] = encoder.transform(X_test['region_original']) #map the training data means to test data


# - We select one hot encoded variables for future work
# - Typically target encoder is used for categorical variables with high cardinality (>15)

# ## Scaling of numerical variables
# - There are numerous transformation that can be performed on numerical variables
# - Many algorithms require scaling such as standardization or normalization.
# - Syntax is similar instantiate -> fit -> transform
# - Try the normalization of age using scikit MinMaxScaler()
# - Eventually use StandardScaler() for standardization

# In[14]:


# fit scaler on training data
norm = MinMaxScaler() #instantiate MinMaxScaler()
norm.fit(X_train[['age']]) #calculates min and max (use training data only!)
age_norm_train = norm.transform(X_train[['age']])
age_norm_test = norm.transform(X_test[['age']])


# ## Save the data to pickle file
# - There are numerous formats that can be used to store the data.
# - Choice depends on specific task, amount of data and memory available.
# - csv is universal but converting data there and back is slow and you can onl;y have one file for one variable (table).
# - We qill use pickle. It is a binary format. You can not view the data in txt viewers. But you can store several variables in the same file and load them all at once. It is also much faster.
# - For big data check for instance parquet format or connection dirtectly to a database.
# 

# In[15]:


#Select features to save
fetures_to_save = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes',
       'region_northeast', 'region_northwest', 'region_southeast',
       'region_southwest']

#Save data in pickle file
with open(path_train_test_file, 'wb') as f:
    pickle.dump([X_train[fetures_to_save], X_test[fetures_to_save], y_train, y_test], f)


