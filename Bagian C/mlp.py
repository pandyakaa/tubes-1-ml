from utils import d_node, d_sigmoid
from functools import reduce
import numpy as np
import random
import math


class MyMlp(object):

    def __init__(self, input_layer, hidden_layer, output_layer):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.layers = np.append(
            np.append(input_layer, hidden_layer), output_layer)
        self.weights = []
        self.n_layer = 2 + len(hidden_layer)
        self.bias = []

        self.initialize_weights()
        self.initialize_biases()

    def initialize_weights(self):
        # Uses Xavier Initialization technique to initialize random weights
        # Initialize first set of weights input - hidden 1
        layer_1 = self.input_layer
        layer_2 = self.hidden_layer[0]
        weight_matrix = np.random.randn(
            layer_2, layer_1)*np.sqrt(2/layer_1 + layer_2)
        self.weights.append(weight_matrix)

        # Initialize weights for each set of hidden layers
        n_hidden = len(self.hidden_layer)
        for i in range(0, n_hidden-1):
            layer_1 = self.hidden_layer[i]
            layer_2 = self.hidden_layer[i+1]
            weight_matrix = np.random.randn(
                layer_2, layer_1)*np.sqrt(2/layer_1 + layer_2)
            self.weights.append(weight_matrix)

        # initialize final set of weights for hidden - output
        layer_1 = self.hidden_layer[-1]
        layer_2 = self.output_layer
        weight_matrix = np.random.randn(
            layer_2, layer_1)*np.sqrt(2/layer_1 + layer_2)
        self.weights.append(weight_matrix)

    def initialize_biases(self):
        # Count number of passes to make
        n_passes = len(self.weights)
        
        #initialize bias with 0.01
        for i in range(0,n_passes):
            self.bias.append(0.01)
        
    
    def feed_forward(self, input_values: np.array) :
        # Output : list of np.array of np.array 
        result_list = [] 
        # Count number of passes to make
        n_passes = len(self.weights)
        # Append result_list
        result_list.append(input_values)

        for i in range(0,n_passes) :
            input_values = np.dot(input_values,self.weights[i])
            # factor in biases
            input_values = input_values + self.bias[i]
            result_list = result_list.append(input_values)
        return result_list

    def back_propagation(self, output: list, target: np.array) -> list:
        # Output : array of np.array 2 dimensi sebagai representasi delta W
        all_avg_dw = list()

        for i in range(len(output), 0, -1):
            d_node = np.vectorize(d_node)(target, output[i])
            h = output[i - 1]
            data_count_per_batch = d_node.shape[1]

            all_dw_layer = list()

            for j in range(data_count_per_batch):
                dw = np.matmul(h[:, j:j+1], d_node[:, j:j+1].T)
                all_dw_layer.append(dw)

            avg_dw_layer = reduce(lambda x, y: x + y, all_dw_layer)
            avg_dw_layer = np.vectorize(
                lambda x: x/data_count_per_batch)(avg_dw_layer)

            all_avg_dw.insert(0, avg_dw_layer)

        return all_avg_dw

    def fit(self, x_train: np.array, y_train: np.array, treshold: float, mini_batch_size=10, epochs=1000, learning_rate=0.001):
        n = len(x_train)
        for epoch in range(epochs):
            x_mini_batches = [x_train[i:i+mini_batch_size]
                              for i in range(0, n, mini_batch_size)]
            y_mini_batches = [y_train[i:i+mini_batch_size]
                              for i in range(0, n, mini_batch_size)]
            for i in range(len(x_mini_batches)):
                mini_batch = np.concatenate(
                    [x_mini_batches[i], y_mini_batches[i]], axis=1)
                self.update_batch(mini_batch, learning_rate)

    def update_batch(self, mini_batch, learning_rate):
        mini_batch_data = mini_batch[0:, :-1]
        mini_batch_target = mini_batch[0:, -1:]

        feed_forward_result = self.feed_forward(mini_batch_data.T)
        target_matrix = mini_batch_target.T

        back_prop_result = self.back_propagation(
            feed_forward_result, target_matrix)

        for i in range(len(self.weights)):
            self.weights[i] += (learning_rate * back_prop_result[i])

    def predict(self, x_test: np.array) -> np.array:
        result = self.feed_forward(x_test.T)

        return result[-1]

    def score(self, x_test: np.array, y_test: np.array) -> float:
        # Output : akurasi dari model
        correct = 0
        y_predict = self.predict(x_test)
        for out, target in zip(y_predict, y_test):
            if(np.array_equal(out, target)):
                correct += 1
        return correct/x_test.shape[0]

    def __str__(self):
        for l in range(0, len(self.weights)):
            weight = self.weights[l]
            layer = 'Layer-'+str(l)+'('
            for i in range(0, weight.shape[0]):
                weightRow = weight[i]
                for j in range(0, len(weightRow)):
                    w = layer+str(j)+'-'+str(i)+') : '+str(weightRow[j])
                    print(w, end='  ')
            print('\n')
