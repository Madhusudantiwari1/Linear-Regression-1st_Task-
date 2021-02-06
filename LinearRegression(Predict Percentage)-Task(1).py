#!/usr/bin/env python
# coding: utf-8

# In[11]:


#importing all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


#Reading the data from url
url="http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")
df.head(10)


# In[13]:


#Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Score')  
plt.xlabel('study Hours')  
plt.ylabel('Percentage Score')  
plt.show()


# In[14]:


#Preparation of Data
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values 
print(y)


# In[15]:


#Split the data using Scikit-Learn's built-in train_test_split() method:
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[16]:


#Import linear Regression model
from sklearn.linear_model import LinearRegression  
Score_Regressor = LinearRegression()  
Score_Regressor.fit(X_train, y_train) 

print("Module executed.")


# In[17]:


# Plotting the regression line
line = Score_Regressor.coef_*X+Score_Regressor.intercept_ #(y=mx+ c)

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[18]:


#Prediction
print(X_test) #  data - In Hours
y_prediction = Score_Regressor.predict(X_test) # Predicting the scores


# In[19]:


# Comparing Actual vs Predicted
dataframe = pd.DataFrame({'Actual': y_test, 'Predicted': y_prediction})  
dataframe


# In[20]:


#Evaluate Result
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_prediction)) 


# In[ ]:





# In[ ]:




