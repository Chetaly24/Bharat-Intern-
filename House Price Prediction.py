#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


Data = pd.read_csv('Housing.csv')


# In[3]:


Data.head()


# In[4]:


Data.info()


# In[5]:


Data.describe()


# In[8]:


X = Data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = Data['Price']


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)


# In[13]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[14]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()


# In[15]:


sns.displot((y_test-predictions),bins=50,color="green")
plt.show()


# In[16]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

