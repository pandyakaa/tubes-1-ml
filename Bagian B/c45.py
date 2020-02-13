from id3 import ID3
import numpy as np
from statistics import mode
from reader import read_csv
from c45_numeric_handler import process_numeric

class C45(ID3):
    def __init__(self):
        super().__init__()

    @staticmethod
    def helper_missing_attribut(self, x, y, col):
        temp = x[:, col]
        for i in range(len(temp)):
            if temp[i] == '?':
                y_temp = y[i]
                y_arr = []
                for j in range(len(y)):
                    if y[j] == y_temp and temp[j] != '?' and temp[j] != None:
                        y_arr.append(temp[j])
                x[i][col] = mode(y_arr)

        return x

    def normalize_missing_attribute(x, y):
        for row in range(len(x)):
            for col in range(len(x[row])):
                if (x[row][col] == '?' or x[row][col] == None):
                    x = C45.helper_missing_attribut(x, y, col)

        return x
    
    @staticmethod
    def fit(x, labels, y, default_val=False) :
        print('Dari C45')
        process_numeric(x,y)
        print("=====After numeric processing======")
        print(x)
        print("==============================")
        return ID3.fit(x, labels, y, True)


    @staticmethod
    def train_test_split(x, y):
        ln_x = ceil(len(x) * 0.8)
        ln_y = ceil(len(y) * 0.8)
        ln_x_not = len(x) - ln_x
        ln_y_not = len(y) - ln_y

        x_train = x[-ln_x:]
        y_train = y[-ln_x:]

        x_test = x[:ln_x_not]
        y_test = y[:ln_y_not]

        return x_test, y_test, x_train, y_train

    def count_accuracy(self, y, y_test) :
        count = 0
        for i in range(len(y)) :
            if y[i] == y_test[i] :
                count += 1
        
        return (count/len(y)*100)

    def prune(self, x_test, y_test, label):

        predictions = self.predict(x_test, label)
        acc = self.count_accuracy(predictions, y_test)

        print(str(acc) + '%')
    
    def predict_after_prune() :
        pass

if __name__ == "__main__":
    data = read_csv('play_tennis.csv')
    label = data[0, 1:-1].tolist()
    training_label = label.copy()
    x = data[1:, 1:-1]
    target = data[1:, -1:].flatten()

    x_test, y_test, x_train, y_train = C45.train_test_split(x,target)
    c45 = C45()
    c45.tree = c45.fit(x_train, label, y_train)
    print(c45.tree)
    c45.prune(x_test, y_test, training_label)