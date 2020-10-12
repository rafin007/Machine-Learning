import numpy as np
from matplotlib import pyplot as plt

#Class for creating a Perceptron
class Perceptron:

    #initialize perceptron
    def __init__(self, neurons, bias, epochs, data, eta, dimensionality, num_training, num_testing):
        self.neurons = neurons #number of input neurons
        self.bias = bias #bias
        self.epochs = epochs #number of epochs
        self.data = data #training data
        self.eta = eta #learning rate
        self.dimensionality = dimensionality #dimensionality of the matrix
        self.ee = np.zeros(num_training) #error difference between predicted value and generated value
        self.mse = np.zeros(self.epochs) #mean squared error for plotting the graph
        self.weight = np.zeros(neurons) #initial weight
        self.num_training = num_training #number of training samples
        self.num_testing = num_testing #number of testing samples
        self.error_points = 0 #to keep track of the total testing error


    #returns the shuffled dataset
    def get_shuffled_dataset(self, sample):
        shuffle_sequence = np.random.choice(self.data.shape[0], sample)
        shuffled_data = self.data[shuffle_sequence]
        return shuffled_data

    #return the output of activation function
    def activation_func(self, x):
        y = np.sign(self.weight.dot(x) + self.bias)
        return y


    def fit(self): #learn through the number of traing samples

        for e in range(self.epochs):
            #shuffle the dataset for each epoch
            shuffled_data = self.get_shuffled_dataset(self.num_training)

            for i in range(self.num_training):
                #fetch data
                x = shuffled_data[i, 0:self.neurons]

                #fetch desired output from dataset
                d = shuffled_data[i, self.dimensionality]

                #activation function
                y = self.activation_func(x)

                #calculate difference
                self.ee[i] = d - y

                #new weight
                new_weight = self.weight + x.dot(self.ee[i] * self.eta)

                #at any point if the weights are similar, then skip to the next epoch
                if new_weight.any() == self.weight.any(): 
                    break
                
                #otherwise set the new weight as current weight
                self.weight = new_weight

            #calculate mean squared error for each epoch
            self.mse[e] = np.square(self.ee).mean()

        print('End of training.')
        print(f'Total points trained: {self.num_training}')
        print(f'Total epochs: {self.epochs}')
        print('____________________________________')


    #show graph of learning curve against mean squared error
    def plot_fit(self):
        plt.xlabel('Epochs')
        plt.ylabel('Mean squared error (mse)')
        plt.title('Training accuracy')
        plt.plot(self.mse)
        plt.show()


    def predict(self): #predict and calulate testing accuracy
        
        shuffled_data = self.get_shuffled_dataset(self.num_testing)

        for i in range(self.num_testing):
            #fetch data
            x = shuffled_data[i, 0:self.neurons]

            # activation function
            y = self.activation_func(x)

            #plot x based on the result of activation function
            if y == 1:
                plt.plot(x[0], x[1], 'rx')
            elif y == -1:
                plt.plot(x[0], x[1], 'k+')

            #calculate error points
            if abs(y - shuffled_data[i, self.dimensionality]) > 0.0000001:
                self.error_points += 1

        #calculate testing accuracy
        testing_accuracy = 100 - ((self.error_points/self.num_testing) * 100)

        print('End of testing.')
        print(f'Total points tested: {self.num_testing}')
        print(f'Total errror points: {self.error_points}')
        print(f'Testing accuracy: {testing_accuracy:.2f}%')

    #plot testing graph
    def plot_predict(self):
        plt.title('Testing accuracy')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.show()


