from mlp import MyMlp
import numpy as np
from utils import read_csv, oneHotEncoder

if __name__ == "__main__":
    dataset = read_csv('datasets/iris.csv')
    label = dataset[0]
    data = dataset[1:]
    target_values = data[0:, -1:].flatten()

    data_feature = data[0:, :-1].astype(float)
    data_target = oneHotEncoder(target_values)

    input_layer = len(data_feature[0])
    output_layer = len(set(target_values))
    hidden_layer = [4, 3]

    mlp = MyMlp(input_layer, hidden_layer, output_layer)
    mlp.fit(data_feature, data_target, 0.1, epochs=10000, learning_rate=0.01)

    mlp.print()
