#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as num
import sys
import matplotlib.pyplot as plt

class LogisticRegression:
  def __init__(self,n_iter,learning_rate):
    self.learning_rate=learning_rate
    self.n_iter=n_iter
    self.weights=None
    self.bias=None

  def fit(self,x,y):
    n_samples,n_features=x.shape

    self.weights=num.zeros(n_features)
    self.bias=0
    for i in range(self.n_iter):
      out_y=num.dot(x,self.weights) + self.bias
      predict=1/(1+num.exp(-out_y))

      error=predict-y
      dw=1/(n_samples)*(num.dot(x.T,error))
      db=1/(n_samples)*num.sum(error)
      self.weights=self.weights - self.learning_rate*dw
      self.bias=self.bias-self.learning_rate*db

  def predict(self,x):
    out_y=num.dot(x,self.weights) + self.bias
    predict_list=[]
    
    predict=1/(1+num.exp(-out_y))
    for pred in predict:
        if pred>0.5:
            predict_list.append(1)
        else:
            predict_list.append(0)
    return num.array(predict_list)

from sklearn import datasets
from sklearn.model_selection import train_test_split

breast_cancer=datasets.load_breast_cancer()
x,y=breast_cancer.data,breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

classification=LogisticRegression(1000,0.001)

classification.fit(X_train,y_train)

predicted=classification.predict(X_train)

def accuracy(y_true,y_pred):
    score=0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            score=score+1
        else:
            score=score
    return score/len(y_true)

print(accuracy(y_train,predicted))


# In[ ]:





# In[ ]:





# In[ ]:




