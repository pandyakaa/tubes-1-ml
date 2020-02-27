import numpy as np
import math
import csv
import numpy as np

# CSV reader function


def read_csv(filename):
    with open(filename) as f:
        temp_mat = []
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            temp_mat.append(row)

        np_mat = np.array(temp_mat)

    return np_mat

# Sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Derivative of sigmoid function
def d_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

# one hot encoder
def oneHotEncoder(target):
    unique = list(set(target))
    unique_count = len(unique)
    encoded = []
    for t in target:
        temp = np.array(np.zeros(unique_count, dtype=int))
        temp[unique.index(t)] = 1
        encoded.append(temp)
    encoded = np.array(encoded)
    return encoded

def half_squared_err(delta):
    return 0.5*(delta**2)


def d_node(target, out):
    return -1*(target - out) * out * (1 - out)
