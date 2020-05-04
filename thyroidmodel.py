# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:51:17 2020

@author: pavitra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df = pd.read_csv('thyroid.csv')
df.head()
df.describe()
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
y = df['output']
X = df.drop(['output'], axis=1)

kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)
clf = MLPClassifier(hidden_layer_sizes=(200,200,200), max_iter=2000, alpha=0.0001,solver='sgd', verbose=100,  random_state=21,tol=0.000000001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#print(y_pred)
a=accuracy_score(y_test, y_pred)
b=a*100
print("Accuracy is = ", b)
cm = confusion_matrix(y_test, y_pred)
print(cm)
p = precision_score(y_test, y_pred, labels=[0,1,2],average='macro')
print("Precision = ",p)

r = recall_score(y_test, y_pred, labels=[0,1,2],average='macro')
print("Recall = ", r)

f=f1_score(y_test, y_pred, labels=[0,1,2],average='macro')
print("Score = ",f)

print(metrics.classification_report(y_test,y_pred,digits=3))
#sns.heatmap(cm, center=True)
#plt.show()
pickle.dump(clf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[0.66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.053,0.016,0.113,0.129,0.088]]))
