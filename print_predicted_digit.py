from sklearn import svm, datasets
import matplotlib.pyplot as plt

digits=datasets.load_digits()
clf=svm.SVC(gamma=0.001,C=100)
print len(digits.data)
X,y=digits.data[:-1], digits.target[:-1]
clf.fit(X,y)
X = X.reshape(-1, 1)
#y.reshape(-1, 1)
#print ('Prediction: ', clf.predict(digits.data[-1]))
print("Prediction of last:",clf.predict(digits.data[[-1]]))
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()