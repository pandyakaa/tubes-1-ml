from .utils import d_node, d_sigmoid, sigmoid, one_hot_encoder
from functools import reduce

import numpy as np
import random
import math


class MyMlp(object):
    def __init__(self, input_layer, hidden_layer, output_layer, treshold=0.01, mini_batch_size=10, epochs=500, learning_rate=0.01):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.layers = np.append(
            np.append(input_layer, hidden_layer), output_layer)
        self.weights = []
        self.n_layer = 2 + len(hidden_layer)
        self.bias = []
        self.treshold = treshold
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.initialize_weights()
        self.initialize_biases()

    def initialize_weights(self):
        # Uses Xavier Initialization technique to initialize random weights
        # Initialize first set of weights input - hidden 1
        layer_1 = self.input_layer
        layer_2 = self.hidden_layer[0]
        weight_matrix = np.random.randn(
            layer_2, layer_1) % 1
        self.weights.append(weight_matrix)

        # Initialize weights for each set of hidden layers
        n_hidden = len(self.hidden_layer)
        for i in range(0, n_hidden-1):
            layer_1 = self.hidden_layer[i]
            layer_2 = self.hidden_layer[i+1]
            weight_matrix = np.random.randn(
                layer_2, layer_1) % 1
            self.weights.append(weight_matrix)

        # initialize final set of weights for hidden - output
        layer_1 = self.hidden_layer[-1]
        layer_2 = self.output_layer
        weight_matrix = np.random.randn(
            layer_2, layer_1) % 1
        self.weights.append(weight_matrix)

    def initialize_biases(self):
        # Count number of passes to make
        n_passes = len(self.weights)

        # initialize bias with 0.01
        for i in range(0, n_passes):
            self.bias.append(0.01)

    def feed_forward(self, input_values: np.array):
        # Output : list of np.array of np.array
        result_list = []
        # Count number of passes to make
        n_passes = len(self.weights)
        # Append result_list
        result_list.append(input_values)
        # print(input_values)
        # print(self.weights)
        sigmoid_vect = np.vectorize(sigmoid)

        for i in range(0, n_passes):
            input_values = np.dot(self.weights[i], input_values)
            # factor in biases
            input_values = input_values + self.bias[i]
            input_values = sigmoid_vect(input_values)
            result_list.append(input_values)

        return result_list

    def back_propagation(self, output: list, target: np.array, current_weight: list) -> list:
        # Output : array of np.array 2 dimensi sebagai representasi delta W
        all_avg_dw = list()

        # print("target")
        # print(target, end="\n\n")

        for i in range(len(output) - 1, 0, -1):
            h = output[i - 1]
            all_dw_layer = list()
            data_count_per_batch = target.shape[1]

            if (i == len(output) - 1):
                error = np.vectorize(d_node)(target, output[i])

                for j in range(data_count_per_batch):
                    dw = np.matmul(error[:, j:j+1], h[:, j:j+1].T)
                    all_dw_layer.append(dw)

            else:
                d_error_next_layer = error

                # dE/dOout1 * dOout1/din1
                # dE/dOout2 * dOout2/din2

                d_out_in = np.vectorize(d_sigmoid)(output[i])

                # dh2out1/dh2in1 = d_sigmoid

                out_weight = current_weight[i]
                # print(out_weight)

                # w11 w21 w31
                # w12 w22 w32

                error = list()

                for j in range(data_count_per_batch):
                    d_out_in_w = np.matmul(
                        d_out_in[:, j:j+1], h[:, j:j+1].T)  # 4x3

                    # dh2out1/dh2in1 * h1out1    dh2out1/dh2in1 * h1out2    dh2out1/dh2in1 * h1out3    dh2out1/dh2in1 * h1out4
                    # dh2out2/dh2in2 * h1out1    dh2out2/dh2in2 * h1out2    dh2out2/dh2in2 * h1out3    dh2out2/dh2in2 * h1out4
                    # dh2out3/dh2in3 * h1out1    dh2out3/dh2in3 * h1out2    dh2out3/dh2in3 * h1out3    dh2out3/dh2in3 * h1out4

                    d_error_next_layer_j = d_error_next_layer[:, j:j+1]
                    d_error_j = np.matmul(d_error_next_layer_j.T, out_weight).T
                    d_error_j_for_all = np.repeat(
                        d_error_j, d_out_in_w.shape[1], axis=1)

                    # dE/dh2out1 dE/dh2out1 dE/dh2out1 dE/dh2out1
                    # dE/dh2out2 dE/dh2out2 dE/dh2out2 dE/dh2out2
                    # dE/dh2out3 dE/dh2out3 dE/dh2out3 dE/dh2out3

                    # dE/dh2out1 = dE/dOout1 * dOout1/din1 * w11 + dE/dOout2 * dOout1/din2 * w12
                    # dE/dh2out2 = dE/dOout1 * dOout1/din1 * w21 + dE/dOout2 * dOout1/din2 * w22
                    # dE/dh2out3 = dE/dOout1 * dOout1/din1 * w31 + dE/dOout2 * dOout1/din2 * w32

                    dw = np.multiply(d_error_j_for_all, d_out_in_w)

                    error.append(d_error_j.flatten())
                    all_dw_layer.append(dw)

                error = np.array(error).T
                # print(error)

            avg_dw_layer = reduce(lambda x, y: x + y, all_dw_layer)
            avg_dw_layer = np.vectorize(
                lambda x: x/data_count_per_batch)(avg_dw_layer)

            # print("output")
            # print(output[i], end="\n\n")
            # print("h:")
            # print(h, end="\n\n")
            # print("delta:")
            # print(avg_dw_layer, end="\n\n")

            all_avg_dw.insert(0, avg_dw_layer)

        # print("all_avg_dw")
        # print(all_avg_dw, end="\n\n\n\n")
        return all_avg_dw

    def fit(self, x_train: np.array, y_train: np.array):
        treshold = self.treshold
        mini_batch_size = self.mini_batch_size
        epochs = self.epochs
        learning_rate = self.learning_rate

        n = len(x_train)
        y_encoded = one_hot_encoder(y_train)
        y_train = y_encoded['encoded']
        self.y_dict = y_encoded['dict']

        for epoch in range(epochs):
            x_mini_batches = [x_train[i:i+mini_batch_size]
                              for i in range(0, n, mini_batch_size)]
            y_mini_batches = [y_train[i:i+mini_batch_size]
                              for i in range(0, n, mini_batch_size)]

            for i in range(len(x_mini_batches)):
                self.update_batch(
                    x_mini_batches[i], y_mini_batches[i], learning_rate)

    def update_batch(self, mini_batch_data, mini_batch_target, learning_rate):
        feed_forward_result = self.feed_forward(mini_batch_data.T)
        target_matrix = mini_batch_target.T

        back_prop_result = self.back_propagation(
            feed_forward_result, target_matrix, self.weights)

        for i in range(len(self.weights)):
            self.weights[i] -= (learning_rate * back_prop_result[i])

    def predict_proba(self, x_test: np.array) -> np.array:
        result = self.feed_forward(x_test.T)[-1].T

        return result

    def predict(self, x_test: np.array) -> np.array:
        result = self.feed_forward(x_test.T)[-1].T
        classes = []

        for i in range(3):
            print(result[i])

        for t in result:
            temp = np.array(np.zeros(t.shape[0], dtype=int))
            temp[np.argmax(t)] = 1
            class_prediction = self.get_class_from_array(temp)
            classes.append(class_prediction)

        classes = np.array(classes)
        return classes

    def score(self, x_test: np.array, y_test: np.array) -> float:
        # Output : akurasi dari model
        correct = 0
        y_predict = self.predict(x_test)
        for out, target in zip(y_predict, y_test):
            if(np.array_equal(out, target)):
                correct += 1
        return correct/x_test.shape[0]

    def print(self):
        for l in range(0, len(self.weights)):
            weight = self.weights[l]
            layer = 'Layer-'+str(l)+'('
            for i in range(0, weight.shape[0]):
                weightRow = weight[i]
                for j in range(0, len(weightRow)):
                    w = layer+str(j)+'-'+str(i)+') : '+str(weightRow[j])
                    print(w, end='  ')
            print('\n')

    def get_class_from_array(self, array: np.array):
        for cls_name, cls_value in self.y_dict.items():
            if all(cls_value == array):
                return cls_name
