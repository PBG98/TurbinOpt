#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import numpy
import pandas as pd
import tensorflow as tf


# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("hydroparam.csv", header=None)
'''
print(df.info())
print(df.head())
'''
dataset = df.values
X = dataset[:,:4]
Y = dataset[:,4]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(18, input_dim=4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=1000, batch_size=1)
model.save('split3070.h5')

'''
X_test[:,0] = 1.306
X_test[:,1] = 1.129
X_test[:,2] = 1.109
X_test[:,3] = 0.489
'''


# 예측 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()
for i in range(len(Y_prediction)):
    param = X_test[i]
    label = Y_test[i]
    prediction = Y_prediction[i]
    print(param)
    print("Real Cp: {:.3f}, Predicted Cp: {:.3f}".format(label, prediction))

print('test R^2: %.5f' %(
    r2_score(Y_test, Y_prediction)
))