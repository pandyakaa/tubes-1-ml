import numpy as np 
import math 
import csv

#CSV reader function
def read_csv(filename):
    with open(filename) as f:
        temp_mat = []
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            temp_mat.append(row)

        np_mat = np.array(temp_mat)

    return np_mat

#Sigmoid function 
def sigmoid(x) :
    return 0

#Derivative of sigmoid function
def d_sigmoid(x) :
    return 0 

