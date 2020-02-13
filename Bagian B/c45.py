from id3 import ID3
import numpy as np
from statistics import mode
from reader import read_csv
from Node import Node
from c45_numeric_handler import process_numeric
import math 

class C45(ID3):
    def __init__(self):
        super().__init__()

    @staticmethod
    def helper_missing_attribut(x, y, col) :
        temp = x[:,col]
        for i in range(len(temp)) :
            if temp[i] == '?' :
                y_temp = y[i]
                y_arr = []
                for j in range(len(y)) :
                    if y[j] == y_temp and temp[j] != '?' and temp[j] != None:
                        y_arr.append(temp[j])
                x[i][col] = mode(y_arr)

        return x

    @staticmethod
    def normalize_missing_attribute(x, y):
        for row in range(len(x)) :
            for col in range(len(x[row])) :
                if (x[row][col] == '?' or x[row][col] == None) :
                    x = C45.helper_missing_attribut(x, y, col)
        
        return x

    @staticmethod
    def splitinfo(attr) :
        total = len(attr)
        infosplit = 0
        array_of_values = []
        
        #Get Unique values in attribute
        for values in attr :
            for value in values :
                array_of_values.append(value)
        
        unique_values, counts = np.unique(array_of_values, return_counts=True)

        #Count split info
        for count in counts :
            infosplit = -count/total*(math.log2(count/total))

        return infosplit
    
    @staticmethod
    def fit(x, labels, y, default_val=False) :
        print('Dari C45')
        process_numeric(x,y)

        gain = list()

        if default_val == False:
            default_val = mode(y)

        # All target are the same value
        if np.all(y == y[0, ]):
            return Node(str(y[0]), [], True)

        # Empty attribute
        if x.shape[1] == 0:
            return Node(str(default_val), [], True)

        # Calculate gain
        entropy = ID3.count_entropy(y)
        for idx, attr in enumerate(x.T):
            gain.append(ID3.gain(entropy, attr, y) / C45.splitinfo(attr))

        # Create node from best attribute
        idx_max = np.argmax(gain)
        attr_values = np.unique(x.T[idx_max])

        node = Node(labels[idx_max], attr_values, False)

        # Delete label of best attribute
        next_labels = labels.copy()
        next_labels.pop(idx_max)

        # Split row based on best attribute unique value
        data_per_values = dict()
        for value in attr_values:
            value_x = np.array([])
            value_y = np.array([])
            for idx, example in enumerate(x):
                if (example[idx_max] == value):
                    if value_x.shape[0] == 0:
                        value_x = np.array([example])
                        value_y = np.array([y[idx]])
                    else:
                        value_x = np.vstack((value_x, example))
                        value_y = np.append(value_y, y[idx])

            value_x = np.delete(value_x, idx_max, axis=1)
            data_per_values[value] = (value_x, value_y)

        # Recursively set child for each attribute
        for value, data in data_per_values.items():
            print(data)
            node.set_child(value, ID3.fit(data[0], next_labels, data[1]))

        return node



if __name__ == "__main__":
    data = read_csv('play_tennis.csv')
    label = data[0, 1:-1].tolist()
    x = data[1:, 1:-1]
    target = data[1:, -1:].flatten()

    ex_x = C45.normalize_missing_attribute(x, target)
    print(ID3.fit(ex_x, label, target))