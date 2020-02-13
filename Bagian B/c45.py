from id3 import ID3
import numpy as np
from statistics import mode
from reader import read_csv

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

if __name__ == "__main__":
    data = read_csv('play_tennis.csv')
    label = data[0, 1:-1].tolist()
    x = data[1:, 1:-1]
    target = data[1:, -1:].flatten()

    ex_x = C45.normalize_missing_attribute(x, target)
    print(ID3.fit(ex_x, label, target))