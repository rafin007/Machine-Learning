from matplotlib import pyplot as plt
import numpy as np
from halfmoon import halfmoon
from Perceptron import Perceptron


if __name__ == "__main__":

    #samples
    num_training = 1000 #number of training data
    num_testing = 2000  #number of testing data
    num_sample = num_training + num_testing #total data
    epochs = 50

    #----------------Half moon--------------------
    radius = 10.0 #central radius of the half moon
    width = 2.0 #width of the half moon
    distance = 0.0 #distance between two half moons

    #fetch the halfmoon data
    data = halfmoon(radius, width, distance, num_sample)

    #-----------------initialize Perceptron variables---------------------
    neurons = data.shape[1]
    bias = distance / 2
    eta = 0.1 #learning rate
    dimensionality = neurons - 1 #number of input dimensionality

    # ------------------initialize Perceptron network----------------------
    perceptron = Perceptron(neurons, bias, epochs, data, eta, dimensionality, num_training, num_testing)

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


