import numpy as np
from collections import Counter
def euclidian_distance(X1,X2):
    return np.sqrt(np.sum(X1-X2)**2)
class KNN:
    def __init__(self,k):
        self.k = k
    def train_data(self, x, y):
        self.X_train = x
        self.Y_train = y
    def predict(self,X):
        predicted_label = [self._predict(x) for x in X]
        return np.array([predicted_label])
    def _predict(self,x):
        # calculate distance
        distance = [euclidian_distance(x,x_train) for x_train in self.X_train]

        # get k nearest sample and labels
        k_nearest_indicis = np.argsort(distance)[:self.k]  # argsort sort the distance and return indicis
        k_labels = [self.Y_train[i] for i in k_nearest_indicis]
        # majority vote, output
        voted_class = Counter(k_labels).most_common(1)
        return voted_class[0][0]
