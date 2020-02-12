import numpy as np
import math
import Node

def count_entropy(target_attributes):
    target_dictionary = dict()
    total = 0
    entropy = 0

    for val in target_attributes:
        if val in target_dictionary:
            target_dictionary[val] += 1
            total += 1
        else:
            target_dictionary[val] = 1
            total += 1

    for attr, val in target_attributes.items():
        entropy += -val/total*(math.log2(-val/total))

    return entropy

def mode(target): 
    return max(set(target), key = target.count) 

def fit(x, y, defaultVal = False):
    if defaultVal == False:
        defaultVal = mode(y)
    for target in y:
        if (all(element==target for element in y)) :
            return Node(target, [], True)