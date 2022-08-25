import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from pyDOE import *
from scipy.stats.distributions import norm

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv("hydroparam.csv", header=None)
dataset = df.values
X = dataset[:, :4]
Y = dataset[:, 4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
Y = Y.reshape(-1, 1)

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

regressor = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.1)
regressor.fit(X, Y.ravel())

nTurbinParam = lhs(4, samples=1000, criterion='center')
means = [1.0135, 1.0555, 1.2735, 0.9775]
stdvs = [0.0675, 0.0475, 0.09316, 0.1628]
for i in range(4):
    nTurbinParam[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(nTurbinParam[:,i])

Y_pred = regressor.predict(nTurbinParam).flatten()

for i in range(len(Y_pred)):
    param = nTurbinParam[i]
    prediction = Y_pred[i]
    if(prediction>1):
        print(param)
        print("Predicted Cp: {:.3f}".format(prediction))


nMax = max(Y_pred)
print(nMax)
nMaxIndex = np.where(Y_pred == nMax)
print(nMaxIndex)
print(nTurbinParam[nMaxIndex])
