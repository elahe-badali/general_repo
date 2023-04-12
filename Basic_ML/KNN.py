#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import numpy as np
import pandas as pd
from collections import Counter


# In[7]:


# Reading datasets
main_train = pd.read_csv("training_set.csv")
test_dataset = pd.read_csv("test_set.csv")

main_X_train = main_train.values[:,:-1]
main_y_train = main_train.values[:,-1]

train_dataset = main_train[:800]
val_dataset = main_train[800:]

X_train = train_dataset.values[:,:-1]
y_train = train_dataset.values[:,-1]

X_val = val_dataset.values[:,:-1]
y_val = val_dataset.values[:,-1]

X_test = test_dataset.values 


# In[3]:


def euclidean_distance(x1 , x2):
    return np.sqrt(np.sum(x1 - x2)**2)

class KNN:
    def __init__(self , k):
        self.k = k
    
    def fit(self , X , y):
        self.X_train = X
        self.y_train = y
        
    def predict(self , X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    def _predict(self , x):
        
        # compute distance
        distance = [euclidean_distance(x , x_train) for x_train in self.X_train]
        
        # get k nearest samples
        k_index = np.argsort(distance)[:self.k]
        k_nearst_labels = [self.y_train[i] for i in k_index ]
        # majority vote
        
        most_common = Counter(k_nearst_labels).most_common(1)
        return most_common[0][0]


# # Validation

# In[6]:


classifier = KNN(k = 100)
classifier.fit(X_train , y_train)

Y_pred = classifier.predict(X_val)

acc = np.sum(Y_pred == y_val)/len(y_val)
print(acc)


# # Test

# In[8]:


classifier = KNN(k = 100)
classifier.fit(main_X_train , main_y_train)

Y_pred = classifier.predict(X_test)

Y_pred = np.where(Y_pred == 0 , -1 , 1)
Y_pred


# In[ ]:




