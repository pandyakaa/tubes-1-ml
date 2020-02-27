import numpy as np


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

    def back_propagation(self, output: np.array, target: np.array) -> np.array:
        # Output : array of np.array 2 dimensi sebagai representasi delta W
        pass

    def fit(self, x_train: np.array, y_train: np.array):
        # Output : none
        pass

    def predict(self, x_test: np.array) -> np.array:
        # Output : array of hasil prediksi (target prediksi)
        pass

    def score(self, x_test: np.array, y_test: np.array) -> float:
        # Output : akurasi dari model
        pass

    pass
