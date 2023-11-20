# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:24:43 2023

@author: EGabriele
"""

# Keras is a high level, deep-learning API for implementing neural networks.
import keras as keras
# Lossess is to compute quantity that a model should seek to minimize during training
# Metrics is function used to judge performance of model
# Activations ensures sloper larger than 1 for positive inputs.
# Optimizers is a method used to change attribute of model such as weights
#from keras import losses, activations, optimizers,metrics
# Loads and return the Iris dataset (classification)
#from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
# Perceptron is a classification algorithm
#from sklearn.linear_model import Perceptron

# Generates a lowpass filter
def generate_data(Tc):
    # setup
    N = 1000
    T = 10 #[s]
    dt = T/N

    # Lowpass filter (first order dynamic, one input)
    #Lowpass filter passes signals with a frequency lower than a selected cutoff frequency and attenuates signals with freq higher than cutoff freq.
    # Tc is Time constant, dt is Time step
    a = Tc / (Tc + dt)
    b = 1-a

    # simulation loop
    x = np.zeros((N, 1))
    t = np.zeros((N, 1))
    for i in range(N-1):
        u = 1 if i < N/2 else 0
        x[i+1] = a*x[i] + b*u
        t[i+1] = t[i] + dt

    return t, x


def structure_input(data, K):
    N = len(data)
    m = np.zeros((N-K, K))

    for k in range(K):
        m[:, k] = data[k:k+N-K, 0]

    x = m[:, 0:9]
    y = m[:, 9]

    return x,y

if __name__ == '__main__':
    t, x_sim = generate_data(Tc = 1)
    t, x_sim2 = generate_data(Tc = 2)

    #plt.plot(t,x)
    #plt.show()

    x_train, y_train = structure_input(x_sim, 10)
    x_test, y_test = structure_input(x_sim2, 10)

    # Sequntial API
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, activation=keras.activations.tanh))
    model.add(keras.layers.Dense(5, activation=keras.activations.tanh))
    model.add(keras.layers.Dense(1, activation=keras.activations.tanh))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1000, verbose=0)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    plt.plot(y_train, c='r', label='train')
    plt.plot(y_train_pred, c='b', label='train pred')

    plt.plot(y_test, c='g', label='test')
    plt.plot(y_test_pred, c='m', label='test pred')

    plt.legend()
    plt.show()