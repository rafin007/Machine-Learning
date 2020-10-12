import numpy as np
from matplotlib import pyplot as plt
from time import process_time

#Class for creating a LMS
class LMS:

    #initialize LMS
    def __init__(self, bias, epochs, training_data, testing_data, eta, input_neurons, num_training, num_testing):
        self.bias = bias #bias
        self.epochs = epochs #number of epochs
        self.training_data = training_data #training data
        self.testing_data = testing_data # testing data
        self.eta = eta #learning rate
        self.input_neurons = input_neurons #input_neurons of the matrix
        self.ee = np.zeros(num_training) #error difference between predicted value and generated value
        self.mse = np.zeros(self.epochs) #mean squared error for plotting the graph
        self.weight = np.random.rand(input_neurons) #initial weight
        self.num_training = num_training #number of training samples
        self.num_testing = num_testing #number of testing samples
        self.error_points = 0 #to keep track of the total testing error


    #returns the shuffled dataset
    def get_shuffled_dataset(self, sample, data_norm):
        shuffle_sequence = np.random.choice(data_norm.shape[0], sample)
        shuffled_data = data_norm[shuffle_sequence]
        return shuffled_data

    #return the output of activation function
    def activation_func(self, x):
        y = np.sign(np.transpose(self.weight).dot(x) + self.bias)
        return y

    def normalize_data(self, data_set):
        #transpose the data
        data = np.transpose(data_set)

        #take the last row
        last_row = data[-1:]

        #take everything but last row
        data = data[:-1]

        #normalize the data
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

        #push the last row to the data's last row
        data_norm = np.append(data_norm, last_row, axis=0)

        #transpose it back and return
        return np.transpose(data_norm)


    def fit(self): #learn through the number of traing samples

        #get the starting time
        starting_time = process_time()

        #normalize
        data_norm = self.normalize_data(self.training_data)

        for e in range(self.epochs):
            #shuffle the dataset for each epoch
            shuffled_data = self.get_shuffled_dataset(self.num_training, data_norm)

            # decrease the learning rate
            if e > 20:
                self.eta = 0.01
            elif e > 40:
                self.eta = 0.001

            for i in range(self.num_training):
                #fetch data
                x = shuffled_data[i,0:self.input_neurons]

                #fetch desired output from dataset
                d = shuffled_data[i, self.input_neurons]

                #calculate difference
                self.ee[i] = d - (np.transpose(self.weight).dot(x))

                #new weight
                new_weight = (self.eta * (np.dot(x, self.ee[i])))
                
                #update the current weight
                self.weight = self.weight + new_weight

            #calculate mean squared error for each epoch
            self.mse[e] = np.square(self.ee).mean()

        ending_time = process_time()
        time_taken = ending_time - starting_time

        print('End of training.')
        print(f'Total points trained: {self.num_training}')
        print(f'Total epochs: {self.epochs}')
        print(f'Total time taken for training: {time_taken:.2f} seconds')
        print('____________________________________')


    #show graph of learning curve against mean squared error
    def plot_fit(self):
        plt.xlabel('Epochs')
        plt.ylabel('Mean squared error (mse)')
        plt.title('Learning Curve')
        plt.plot(self.mse)
        plt.show()


    def predict(self): #predict and calulate testing accuracy

        starting_time = process_time()

        #normalize
        data_norm = self.normalize_data(self.testing_data)
        
        #shuffle data
        shuffled_data = self.get_shuffled_dataset(self.num_testing, data_norm)

        for i in range(self.num_testing):
            #fetch data
            x = shuffled_data[i,0:self.input_neurons]

            # activation function
            y = self.activation_func(x)

            #plot x based on the result of activation function
            if y == 1:
                plt.plot(x[0], x[1], 'rx')
            else:
                plt.plot(x[0], x[1], 'k+')

            #calculate error points
            if abs(y - shuffled_data[i, self.input_neurons]) > 0.0000001:
                self.error_points += 1

        ending_time = process_time()
        time_taken = ending_time - starting_time

        #calculate testing accuracy
        testing_accuracy = 100 - ((self.error_points/self.num_testing) * 100)

        print('End of testing.')
        print(f'Total points tested: {self.num_testing}')
        print(f'Total errror points: {self.error_points}')
        print(f'Total time taken for testing: {time_taken:.2f} seconds')
        print(f'Testing accuracy: {testing_accuracy:.2f}%')


    #plot testing graph
    def plot_predict(self):
        plt.title('Testing accuracy')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.show()


