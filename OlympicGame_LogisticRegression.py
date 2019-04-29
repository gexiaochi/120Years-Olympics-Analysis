
# coding: utf-8

# <h1 align="center"> 
# DATS 6501 —Capstone Project (Part II)
# </h1> 
# 
# <h1 align="center"> 
# Olympic Games Analysis — Logistic Regression
# </h1> 
# 
# <h4 align="center"> 
# Author: Xiaochi Ge ([gexiaochi@gwu.edu](mailto:gexiaochi@gwu.edu))
# </h4>

# ## 1. Import Packagea & Read Data

# In[ ]:


import numpy as np
import pandas as pd


# In[2]:


athletes = pd.read_csv('athlete_events.csv')
regions = pd.read_csv('noc_regions.csv')
olympic_data = pd.merge(athletes, regions, on='NOC', how='left')


# In[3]:


olympic = olympic_data[['Sex','Age','Height','Weight','NOC','Sport','Medal']]
#olympic.head()


# ## 2. Fill-in Missing Data

# In[4]:


olympic['Medal'].fillna(('Lose'), inplace=True)
olympic['Age'].fillna((olympic['Age'].median()), inplace=True)
olympic['Height'].fillna((olympic['Height'].median()), inplace=True)
olympic['Weight'].fillna((olympic['Weight'].median()), inplace=True)


# In[5]:


#If medals = Gold,Silver,and Bronze, change it to Win; Otherwise, Lose
def medal(olympic):
    if (olympic['Medal'] == 'Gold'):
        return 'Win'
    elif (olympic['Medal'] == 'Silver'):
        return 'Win'
    elif (olympic['Medal'] == 'Bronze'):
        return 'Win'
    else:
        return 'Lose'


# In[6]:


olympic['Medal'] = olympic.apply(medal, axis=1)
#olympic.head()


# ## 3. Get Dummies

# In[7]:


sex = pd.get_dummies(olympic['Sex'],drop_first=True)
noc = pd.get_dummies(olympic['NOC'],drop_first=True)
sport = pd.get_dummies(olympic['Sport'],drop_first=True)


# In[8]:


olympic.drop(['Sex','NOC','Sport'],axis=1,inplace=True)


# In[9]:


olympic = pd.concat([olympic,sex,noc,sport],axis=1)


# ## 4. Train_Test_Split

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(olympic.drop('Medal',axis=1), 
                                                    olympic['Medal'], test_size=0.30, 
                                                    random_state=101)


# ## 5. Logistic Regression

# In[11]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

acc_log = round(logmodel.score(X_train, y_train) * 100, 2)
acc_log


# ## 6. Classification Report

# In[12]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# ## 7. Coeeficients

# In[13]:


coefficients = pd.DataFrame(logmodel.coef_[0], X_train.columns)
coefficients.columns = ['Coefficient']


# In[14]:


coefficients.sort_values(by='Coefficient', ascending=False)

