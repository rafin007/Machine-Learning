#dataset link: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

import numpy as np
from Perceptron import Perceptron

if __name__ == '__main__':

    #read from file
    file_data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')

    #-----------------initialize Perceptron variables---------------------
    neurons = file_data.shape[1] #number of input neurons
    epochs = 100 #number of epochs
    bias = np.random.uniform(0, 1) #Generate a random bias from 0 to 1
    eta = 0.01 #learning rate
    dimensionality = neurons - 1 #Get input dimensionality

    #------------------divide the samples---------------------------------
    num_training = int((file_data.shape[0] * 30) / 100) #allocate 30% as the training sample
    num_testing = file_data.shape[0] - num_training #allocate the rest 70% as the testing sample


    #------------------initialize Perceptron network----------------------
    perceptron = Perceptron(neurons, bias, epochs, file_data, eta, dimensionality, num_training, num_testing)

    print('Training Perceptron...')
    print('___________________________')

    #Train the perceptron
    perceptron.fit()

    #Plot training data
    perceptron.plot_fit()

    print('Testing Perceptron...')
    print('___________________________')

    #Test the perceptron
    perceptron.predict()

    #Plot testing data
    perceptron.plot_predict()

