#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print len(features_train[0])
#percentile=1	--> 379		features
#percentile=10	--> 3785	features

#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0,min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(pred, labels_test)
print accuracy
#1% --> lower (bi a bit) accuracy (97% dropped to 96%)
#########################################################

