#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
#must use the data from https://github.com/udacity/ud120-projects/blob/master/choose_your_own/class_vis.py
#so execute in the according folder 
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
#from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)


#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0)
    
    #iris = load_iris()
    #cross_val_score(clf, iris.data, iris.target, cv=10)

    clf.fit(features_train, labels_train)
    ### use the trained classifier to predict labels for the test features
    #clf.predict(features_test)
    
    
    return clf