#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
from scipy.stats.distributions import norm
import pandas as pd
import tensorflow as tf
from pyDOE import *

#lhs로 범위 내 난수 생성
nTurbinParam = lhs(4, samples=10000, criterion='center')
means = [1.0135, 1.0555, 1.2735, 0.9775]
stdvs = [0.0675, 0.0475, 0.09316, 0.1628]
#distribution = [0.2025, 0.1425, 0.2795, 0.4885]

for i in range(4):
    nTurbinParam[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(nTurbinParam[:,i])


'''
nTurbinParam[:,0] = 1.306
nTurbinParam[:,1] = 1.129
nTurbinParam[:,2] = 1.109
nTurbinParam[:,3] = 0.489
'''



# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)


#data load
df = pd.read_csv("hydroparam.csv", header=None)
dataset = df.values
X = dataset[:,:4]
Y = dataset[:,4]

#model load
model = load_model('split3070.h5')

# 예측 값과 실제 값의 비교
Y_pred = model.predict(nTurbinParam).flatten()

#Cp > 1 찾기
for i in range(len(Y_pred)):
    #param = X[i]
    #label = Y[i]
    param = nTurbinParam[i]
    prediction = Y_pred[i]
    if(prediction>1.0):
        print(param)
        print("Predicted Cp: {:.3f}".format(prediction))
   #print("Real Cp: {:.3f}, Predicted Cp: {:.3f}".format(label, prediction))


#최대값 추적
nMax = max(Y_pred)
print(nMax)
nMaxIndex = np.where(Y_pred == nMax)
print(nMaxIndex)
print(nTurbinParam[nMaxIndex])