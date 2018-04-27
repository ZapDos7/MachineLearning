#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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


#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100]

#########################################################
### your code goes here ###
from sklearn.svm import SVC
#clf = SVC(kernel="linear")
clf = SVC(kernel="rbf",C=10000) #tested C= 10, 100, 1000 with accuracy 0.61, 0.61 and 0.82 respectively
#with C=10.000, accuracy is 89%, and now it's more complex
#with C=10.000 and the entire train set, accuracy >99%
t0 = time()
### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
"""
one = pred[10]
two = pred[26]
three = pred[50]
print "10th element is class" + str(one)
print "26th element is class" + str(two)
print "50th element is class" + str(three)
"""
count = 0
for p in pred:
    if p == 1:
        count = count + 1
print "Count of Chris's is " + str(count)
print "predicting time:", round(time()-t1, 3), "s"
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print acc
#########################################################


