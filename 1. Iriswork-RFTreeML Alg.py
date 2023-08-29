#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

#loading pandas
import pandas as pd

#loading nummy

import numpy as np

#setting randomness
np.random.seed(0)

iris=load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris)



# In[16]:


df['species']=pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()


# In[17]:


df['is_train']=np.random.uniform(0,1,len(df))<=0.75
df.head()


# In[18]:


train, test = df [df['is_train']==True], df[df['is_train']==False]
print('Number of observations in the training data', len(train))
print('Number of observations in the testing data', len(test))


# In[22]:


features = df.columns[:4]
#view features use 'print(features)' or just 'features'
print(features)


# In[23]:


#converting each species names into digits
y=pd.factorize(train['species'])[0]
#viewing target
y


# In[24]:


#creating a Random Forest Classifier, variable CLF
clf=RandomForestClassifier(n_jobs=2, random_state=0)

#training the classifier
clf.fit(train[features],y)


# In[25]:


#applying the trained classifier to the test
clf.predict(test[features])


# In[26]:


#load features of the test dataset
test[features]


# In[27]:


#viewing the predicted peobability of the first 10 observations
clf.predict_proba(test[features])[0:10]


# In[28]:


#viewing the predicted peobability of the next 10 observations
clf.predict_proba(test[features])[10:20]


# In[29]:


#mapping names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]

#view the predicted species for the first 5 observations
preds[0:5]


# In[30]:


#view the predicted species for the next 10 observations
preds[5:15]


# In[31]:


#view the predicted species for the 32 observations
preds[0:32]


# In[32]:


#viewing the actual species for thr first 5 observations
test['species'].head()


# In[33]:


#confusion matrix
pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames = ['Predicted Species'])


# In[34]:


preds = iris.target_names[clf.predict(test[features])]


# In[37]:


preds = iris.target_names[clf.predict([[5.0, 3.6, 1.4, 2.0], [5.0, 3.6, 1.4, 2.0]])]
preds


# In[ ]:




