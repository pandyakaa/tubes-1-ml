import numpy as np
import random


class MyMlp(object):

    [np.array([[1,2],[3,4]])]

    def __init__(self, input_layer, hidden_layer, output_layer):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.layers = np.append(np.append(input_layer,hidden_layer), output_layer)
        self.weights = []
        self.n_layer = 2 + len(hidden_layer)

    def initialize_weights(self):
        pass

    def feed_forward(self, input_values: np.array) -> np.array:
        # Output : np.array
        pass

    def back_propagation(self, output: np.array, target: np.array) -> np.array:
        # Output : array of np.array 2 dimensi sebagai representasi delta W
        pass

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
        pass
    def score(self, x_test: np.array, y_test: np.array) -> float:
        # Output : akurasi dari model
        correct = 0
        y_predict = self.predict(x_test)
        for out, target in zip(y_predict, y_test):
            if(np.array_equal(out,target)):
                correct+=1
        return correct/x_test.shape[0]

    def __str__(self):
        for l in range(0,len(self.weights)):
            weight = self.weights[l]
            layer = 'Layer-'+str(l)+'('
            for i in range(0,weight.shape[0]):
                weightRow = weight[i]
                for j in range(0,len(weightRow)):
                    w = layer+str(j)+'-'+str(i)+') : '+str(weightRow[j])
                    print(w, end='  ')    
            print('\n')



if __name__ == "__main__":
    input_layer = [0, 1]
    hidden_layer = [2]
    output_layer = [0, 1]
    x_train = np.array([1, 2, 3, 4])
    y_train = np.array([1, 2, 3, 4])
    mlp = MyMlp(input_layer, hidden_layer, output_layer)
    mlp.fit(x_train, y_train, 0.2)
