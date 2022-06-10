#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


data=pd.read_csv("Salary.csv")


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


le=LabelEncoder()


# In[11]:


data["Position"]=le.fit_transform(data["Position"])


# In[12]:


data.head()


# In[13]:


x=data[["Position","Level"]]


# In[14]:


y=data["Salary"]


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[17]:


from sklearn.tree import DecisionTreeRegressor


# In[18]:


dt=DecisionTreeRegressor()


# In[19]:


dt.fit(x_train,y_train)


# In[20]:


y_pred=dt.predict(x_test)


# In[21]:


from sklearn import metrics


# In[22]:


mse=metrics.mean_squared_error(y_test,y_pred)


# In[23]:


mse


# In[24]:


r2=metrics.r2_score(y_test,y_pred)


# In[25]:


r2


# In[26]:


dt.predict([[5,6]])

