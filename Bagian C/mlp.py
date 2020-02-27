from utils import d_node, d_sigmoid
from functools import reduce
import numpy as np
import random


class MyMlp(object):

    def __init__(self, input_layer, hidden_layer, output_layer):
        self.input = input_layer
        self.hidden = hidden_layer
        self.output = output_layer
        self.weights = []
        self.n_layer = len(input_layer) + len(hidden_layer) + len(output_layer)

    def initialize_weights(self):
        pass

    def feed_forward(self, input_values: np.array) -> np.array:
        # Output : np.array
        pass

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

    def fit(self, x_train: np.array, y_train: np.array, treshold: float, mini_batch_size=10, epochs=500):
        n = len(x_train)
        for epoch in range(epochs):
            random.shuffle(x_train)
            mini_batches = [x_train[i:i+mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch)

    def update_batch(self, mini_batch):
        # Update weight per batch
        pass

    def predict(self, x_test: np.array) -> np.array:
        # Output : array of hasil prediksi (target prediksi)
        pass

    def score(self, x_test: np.array, y_test: np.array) -> float:
        # Output : akurasi dari model
        correct = 0
        y_predict = self.predict(x_test)
        for out, target in zip(y_predict, y_test):
            if(np.array_equal(out, target)):
                correct += 1
        return correct/x_test.shape[0]


if __name__ == "__main__":
    input_layer = [0, 1]
    hidden_layer = [2]
    output_layer = [0, 1]
    x_train = np.array([1, 2, 3, 4])
    y_train = np.array([1, 2, 3, 4])
    mlp = MyMlp(input_layer, hidden_layer, output_layer)
    mlp.fit(x_train, y_train, 0.2)
