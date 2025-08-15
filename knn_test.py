import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN
import matplotlib.pyplot
iris_dataset = datasets.load_iris()
x,y = iris_dataset.data, iris_dataset.target
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=111)
#print(x_train.shape)
#print(x_test.shape)
clf = KNN(k=6)
clf.train_data(X_train, Y_train)
predictions = clf.predict(X_test)
accuracy = np.sum(predictions == Y_test)/len(Y_test)
print(accuracy)

#plt.figure()
#plt.scatter(x[:,2],x[:,3], c = y, edgecolors='k',s = 20)
#plt.show()