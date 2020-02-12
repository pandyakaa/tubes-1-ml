import numpy as np
import math
import Node

def mode(target): 
        return max(set(target), key = target.count) 

class ID3(object):
    def __init__(self, labels):
        self.root = None
        self.labels = labels

    @staticmethod
    def count_entropy(target_attributes):
        target_dictionary = dict()
        total = 0
        entropy = 0

        for val in target_attributes:
            if val in target_dictionary:
                target_dictionary[val] += 1
            else:
                target_dictionary[val] = 1
            total += 1

        for attr, val in target_attributes.items():
            entropy += -val/total*(math.log2(-val/total))

        return entropy

    @staticmethod
    def fit(x, labels, y, default_val = False):
        gain = []

        if default_val == False:
            default_val = mode(y)
        
        for target in y:
            if (all(element==target for element in y)) :
                return Node(target, [], True)

        if x.shape[1] == 0:
            return Node(default_val, [], True)

        
        entropy = ID3.count_entropy(y)

        for idx, attr in enumerate(x.transpose()):
            gain[idx] = gain(entropy,attr,y)
        
        idx_max = np.argmax(gain)
        attr_values = np.unique(x.transpose[idx_max])

        node = Node(labels[idx_max],  attr_values, False)

        labels.pop(idx_max)

        data_per_values = dict()
        for value in attr_values:
            value_x = np.matrix([])
            value_y = []
            for idx, example in enumerate(x):
                if(example[idx_max]==value):
                    np.vstack([value_x,example])
                    value_y.append(y[idx])

            x = np.delete(x, idx_max, axis=1)
            data_per_values[value] = (value_x,value_y)
            
        for value, data in data_per_values:
            node.set_child(value,ID3.fit(data[0],labels,data[1]))

        return node