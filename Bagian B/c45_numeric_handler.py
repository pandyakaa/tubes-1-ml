#File containing numeric attribut processing functions
import numpy as np

def is_numeric(x) :
# Input : Np array 
    if len(np.unique(x)) > 2 :
        if x.dtype == 'float64' or x.dtype == 'int64' :
            return True
        else :
            return False 
    else :
        return False

def process_numeric(M, target) :
# Input : 2D numpy matrix containing attribute values 

    n_attribute = M.shape[1] # Get number of columns/attributes

    for column_index in range (0,n_attribute) :
        current_column = x[:,column_index] #Get current attribute
        
        if is_numeric(current_column) :
            threshold = find_breakpoint(current_column, target)
            discretize(M,column_index,threshold)

#[1,1,2,2,2]
def find_threshold(array, target) :
    index = 0
    dictionary = dict()

    for key, value in zip(array,target) :
        dictionary[key] = value
    
    sorted(dictionary)

    # for index in range (0,len(array)) :


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

    for attr, val in target_dictionary.items():
        entropy += -val/total*(math.log2(val/total))

    return entropy








