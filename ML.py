#!/usr/bin/env python
# coding: utf-8

# In[25]:


import sklearn


# In[34]:


from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[27]:


label_names = data['target_names']
labels= data['target']
feature_names = data['feature_names']
features = data['data']


# In[38]:


print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])


# In[29]:


train, test, train_labels, test_labels=train_test_split(features, labels, test_size=0.33, random_state=42)


# In[30]:


gnb=GaussianNB()


# In[31]:


model = gnb.fit(train, train_labels)


# In[36]:


preds = gnb.predict(test)
print(preds)


# In[35]:


print(accuracy_score(test_labels, preds))


# In[ ]:





# In[ ]:




