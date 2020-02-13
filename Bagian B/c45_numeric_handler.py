#File containing numeric attribut processing functions
import numpy as np
import math 

def is_numeric(x) :
    isnumeric = False
    i = 0
    while not(isnumeric) and i != len(x)-1 :
        isnumeric = x[i].isnumeric()
        i = i+1
    return isnumeric

def is_continuous(x) :
# Input : Np array 
    if is_numeric(x) :
        if len(np.unique(x)) > 2 :
            return True
        else :
            return False 
    else :
        return False

def process_numeric(M, target) :
# Input : 2D numpy matrix containing attribute values 
    n_attribute = M.shape[1] # Get number of columns/attributes
    discretize_vect = np.vectorize(discretize)
    for column_index in range (0,n_attribute) :
        current_column = M[:,column_index] #Get current attribute
        if is_continuous(current_column) :
            current_column = current_column.astype('float64')
            threshold = find_threshold(current_column, target)
            print("Column: %s" % (column_index))
            print("Threshold: %s" % (threshold))
            discrete_column = discretize_vect(M[:,column_index],threshold)
            M[:,column_index] = discrete_column
            # if column_index == 0 :
            #     discrete_cols = new_col
            # else :
            #     discrete_cols = np.vstack([discrete_cols,new_col])

    # return discrete_cols
#[1,2,1,2,2] [0,0,0,1,1]
def find_threshold(array, target) :
    index = 0
    dictionary = dict()
    list_of_tuple = []
    list_of_entropy = []

    for item  in zip(array,target) :
        list_of_tuple.append(item)
    
    list_of_tuple = sorted(list_of_tuple)
    for index in range (1,len(array)) :
        #split lists
        list_1 = [i[1] for i in list_of_tuple[:index]]
        list_2 = [i[1] for i in list_of_tuple[index:]]

        #Count Entropy
        entropy = 0
        entropy = count_entropy(list_1) + count_entropy(list_2)
        list_of_entropy.append(entropy)
    
    #Get index of minimum entropy
    threshold_index = list_of_entropy.index(min(list_of_entropy))
    return list_of_tuple[threshold_index][0]

def discretize(item, threshold) : 
    if float(item) <= threshold :
        return "<= %s" % (threshold)
    else :
        return "> %s" % (threshold)


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



m = np.array([[1,"b",1,"a"],[2,"b",2,"b"],[3,"b",3,"c"],[3,"b",3,"d"], [3,"b",3,"e"], [4,"b",4,"f"]])
target = np.array([0,0,1,1,1,1])

