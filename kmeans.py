import pickle
import numpy as np
import matplotlib.pyplot as plt

class KMeans(object):
    """Implementation of KMeans"""
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.centers = X[np.random.randint(X.shape[0], size=self.k)]

        plt.plot(X[:,0], X[:,1], 'bx')
        plt.plot(self.centers[:,0], self.centers[:,1], 'ro')
        plt.show()

        while True:
            distances = np.sqrt(np.sum((X - self.centers[:, np.newaxis]) ** 2, axis=2))
            closestClusters = np.argmin(distances, axis=0)

            newCenters = np.array([np.mean(X[closestClusters == c], axis=0) for c in range(self.k)])
            if np.all(self.centers - newCenters < 1e-5):
                break
            self.centers = newCenters

        plt.plot(X[:,0], X[:,1], 'bx')
        plt.plot(self.centers[:,0], self.centers[:,1], 'ro')
        plt.show()

if __name__ == '__main__':
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['x']

    plt.plot(X[:,0], X[:,1], 'x')
    plt.show()

    kmeans = KMeans(k=2)
    kmeans.fit(X)
