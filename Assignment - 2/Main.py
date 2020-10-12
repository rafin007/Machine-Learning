#dataset link: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

import numpy as np
from LMS import LMS

if __name__ == '__main__':

    #read from file
    file_data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')

    #-----------------initialize Perceptron variables---------------------
    attributes = file_data.shape[1] #number of attributes
    epochs = 100 #number of epochs
    bias = 0 #bias
    eta = 0.1 #learning rate
    input_neurons = attributes - 1 #Get number of input neurons

    #------------------divide the samples---------------------------------
    num_training = int((file_data.shape[0] * 30) / 100) #allocate 30% as the training sample
    num_testing = file_data.shape[0] - num_training #allocate the rest 70% as the testing sample

    #shuffle the data
    np.random.shuffle(file_data)

    #training data
    training_data = file_data[0:num_training]

    #testing data
    testing_data = file_data[num_training:file_data.shape[0]]

    #------------------initialize LMS network----------------------
    lms = LMS(bias, epochs, training_data, testing_data, eta, input_neurons, num_training, num_testing)

    print('Training LMS...')
    print('___________________________')

    #Train the LMS
    lms.fit()

    #Plot training data
    lms.plot_fit()

    print('Testing LMS...')
    print('___________________________')

    #Test the LMS
    lms.predict()

    lms.plot_predict()

