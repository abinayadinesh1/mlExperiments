#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:38:42 2020

@author: abinayadinesh
"""
import pandas as pd
import numpy as np
import scipy
import sklearn
import matplotlib as mlb

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
ds = pd.read_csv(url, names = names)


#SUMMARIZING THE DATASET
#shape
print(ds.shape)

#first 2 rows
print(ds.head(20))

#summary of each attribute
print(ds.describe())

#see how many rows per class of iris
print(ds.groupby('class').size())

#DATA VISUALIZATION
#univariate plots are of each individual variable
#box and whisker plots
ds.plot(kind = "box", subplots = True, layout=(2,2), sharex=False, sharey = False)
pyplot.show()

#histogram
ds.hist()
pyplot.show()

#MULTIVARIATE plots help see interactions between all the variables
#scatterplot matrix
scatter_matrix(ds)
pyplot.show()

#EVALUATING ALGORITHMS
#creating a validation set
array = ds.values
#: is all of the rows, while 0:4 means only the first 4 columns
X = array[:, 0:4]

#shape is (150, 0), will print class for 150 rows?
y = array[:,4]

X_train, X_validation, Y_train, Y_validation, = train_test_split(X, y, test_size = 0.20, random_state = 1)

#uaing the k-fold cross-validation technique
#spot check algorithms
models = []
models.append(('LR', LogisticRegression(solver ='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma="auto")))
#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
#compare algorithms
pyplot.boxplot(results, labels=names)
pyplot.title = "Algorithm Comparison"
pyplot.show()

#make predictions using validation set
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

#evaluate predictions:
#comparing them to results in validation set, calculating classification accuraccy, and making a confusion matrix and classification report
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))









