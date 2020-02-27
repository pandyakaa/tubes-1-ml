import numpy as np
from sklearn.utils import shuffle
from mlp import MyMlp
from utils import read_csv, oneHotEncoder

if __name__ == "__main__":
    dataset = read_csv('datasets/iris.csv')
    label = dataset[0]
    data = dataset[1:]

    data = shuffle(data, random_state=0)
    target_values = data[0:100, -1:].flatten()

    data_feature = data[0:100, :-1].astype(float)
    data_target = oneHotEncoder(target_values)

    input_layer = len(data_feature[0])
    output_layer = len(set(target_values))
    hidden_layer = [4, 3]

    mlp = MyMlp(input_layer, hidden_layer, output_layer)
    mlp.fit(data_feature, data_target, 0.1, epochs=1000, learning_rate=0.01)

    mlp.print()

    target_values_test = data[100:, -1:].flatten()
    data_target_test = oneHotEncoder(target_values_test)
    data_feature_test = data[100:, :-1].astype(float)

    print(data_target_test)
    print(mlp.predict(data_feature_test).T)
    print(mlp.score(data_feature_test, data_target_test))
