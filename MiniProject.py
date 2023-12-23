#!/usr/bin/env python
# coding: utf-8

# # Mini Project

# In[1]:


# Importing modules

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


# Loading the dataset

df = pd.read_csv(r"C:\Users\dsraj\Downloads\9961_14084_bundle_archive\Train.csv")
df.head()


# In[3]:


# Statistical Information

df.describe()


# In[4]:


# Datatype of Attributes

df.info()


# In[5]:


# Checking Unique values in Dataset

df.apply(lambda x: len(x.unique()))


# # Preprocessing the Dataset

# In[6]:


# Checking for Null Values

df.isnull().sum()


# In[7]:


# Checking for Categorical Attributes

cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)
cat_col        


# In[8]:


# Removing unnecessary columns
 
cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
cat_col


# In[9]:


# Printing the Categorical columns 
 
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()


# In[10]:


# Filling in the missing values

item_weight_mean = df.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
item_weight_mean


# In[11]:


# Checking for the missing values of Item_Weight

miss_bool = df['Item_Weight'].isnull()
miss_bool


# In[12]:


for i, item in enumerate(df['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            df['Item_Weight'][i] = item_weight_mean.loc[item]
            ['Item_Weight']
        else:
            df['Item_Weight'][i] = np.mean(df['Item_Weight'])


# In[13]:


df['Item_Weight'].isnull().sum()


# In[14]:


# Checking for the missing values of Outler_Type

outlet_size_mode = df.pivot_table(values='Outlet_Size', columns= 'Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
outlet_size_mode


# In[15]:


# Filling in the missing values for Outlet_Size

miss_bool = df['Outlet_Size'].isnull()
df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])


# In[16]:


df['Outlet_Size'].isnull().sum()


# In[17]:


# Similarly, For Item_Visibility

sum(df['Item_Visibility']==0)


# In[18]:


# Replacing Zeros with Mean

df.loc[:, 'Item_Visibility'].replace([0], [df['Item_Visibility'].mean()], inplace=True)


# In[19]:


sum(df['Item_Visibility']==0)


# In[20]:


# Combining the Repeated Values of the Categorical column

df['Item_Fat_Content'].replace({'LF' : 'Low Fat', 'reg':'Regular', 'low fat': 'Low Fat'})
df['Item_Fat_Content'].value_counts()


# # Creation of New Attributes

# In[21]:


df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
df['New_Item_Type']


# In[22]:


# Filling some meaningful values in it

df['New_Item_Type'] = df['New_Item_Type'].map({'FD' : 'Food', 'NC' : 'Non-Consumable', 'DR' : 'Drinks'})
df['New_Item_Type'].value_counts()


# In[23]:


df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
df['Item_Fat_Content'].value_counts()


# In[24]:


# Create Small Values for Establishment Year

df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']


# In[25]:


df['Outlet_Years']


# In[26]:


df.head()


# # Exploratory Data Analysis

# In[27]:


sns.distplot(df['Item_Weight'])


# In[28]:


sns.distplot(df['Item_Visibility'])


# In[29]:


sns.distplot(df['Item_MRP'])


# In[30]:


sns.distplot(df['Item_Outlet_Sales'])


# In[31]:


# Log Transformation

df['Item_Outlet_Sales'] = np.log(1+df['Item_Outlet_Sales'])


# In[32]:


sns.distplot(df['Item_Outlet_Sales'])


# In[33]:


# Outlet_Establishment_Year column
sns.countplot(x='Outlet_Establishment_Year', data=df)
     


# In[34]:


# Item_Type column


sns.countplot(x='Item_Type', data=df)


# # Correlation Matrix

# In[35]:


corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[36]:


df.head()


# # Label Encoding

# In[37]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    df[col] = le.fit_transform(df[col])


# # One Hot Encoding

# In[38]:


df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'])
df.head()


# # Splitting the data for Training and Testing

# In[39]:


X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']


# # Model Training

# In[40]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    # training the model
    model.fit(X, y)
    
    # predicting the training set
    pred = model.predict(X)
    
    # performing cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Model Report")
    print("MSE:",mean_squared_error(y,pred))
    print("CV Score:", cv_score) 


# ## Linear Regression

# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

def train(model, X, y):
    model.fit(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
train(model, X_scaled, y)

coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
plt.show()


# ## Ridge

# In[42]:


from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

def train(model, X, y):
    model.fit(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge()
train(model, X_scaled, y)

coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
plt.show()


# ## Lasso

# In[43]:


from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt

def train(model, X, y):
    model.fit(X, y)
    
model = Lasso()
train(model, X, y)

coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
plt.show()


# ## Decision Tree

# In[44]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")


# ## Random Forest

# In[45]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")


# ## Extra Trees

# In[46]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")


# ## XGBoost Regressor

# In[47]:


X = df.drop(columns='Item_Outlet_Sales', axis=1)
Y = df['Item_Outlet_Sales']


# In[48]:


print(X)


# In[49]:


print(Y)

