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


# One hot encoder
def one_hot_encoder(target):
    unique = list(set(target))
    unique_dict = dict()
    unique_count = len(unique)
    encoded = []
    for t in target:
        temp = np.array(np.zeros(unique_count, dtype=int))
        temp[unique.index(t)] = 1
        encoded.append(temp)

        if t not in unique_dict:
            unique_dict[t] = temp

    encoded = np.array(encoded)
    return {'encoded': encoded, 'dict': unique_dict}


def half_squared_err(delta):
    return 0.5*(delta**2)


def d_node(target, out):
    return -1*(target - out) * out * (1 - out)


def scale_data(data, lower_bound, upper_bound):
    nominator = (data-data.min(axis=0))*(upper_bound-lower_bound)
    denominator = data.max(axis=0) - data.min(axis=0)
    denominator[denominator == 0] = 1
    normalized_data = nominator/denominator + lower_bound
    return normalized_data
