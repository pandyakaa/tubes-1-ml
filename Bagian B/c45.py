from id3 import ID3, mode
import numpy as np
from reader import read_csv
from Node import Node
from math import ceil, log2
from c45_numeric_handler import process_numeric
from Rule import Rule


def powerset(s):
    x = len(s)
    subs = []
    for i in range(1 << x):
        subs.append([s[j] for j in range(x) if (i & (1 << j))])

    subs.pop(0)

    return subs


class C45(ID3):
    def __init__(self):
        super().__init__()

    @staticmethod
    def helper_missing_attribut(x, y, col):
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

    @staticmethod
    def normalize_missing_attribute(x, y):
        for row in range(len(x)):
            for col in range(len(x[row])):
                if (x[row][col] == '?' or x[row][col] == None):
                    x = C45.helper_missing_attribut(x, y, col)

        return x

    @staticmethod
    def splitinfo(attr):
        total = len(attr)
        infosplit = 0
        array_of_values = []

        # Get Unique values in attribute
        for values in attr:
            for value in values:
                array_of_values.append(value)

        unique_values, counts = np.unique(array_of_values, return_counts=True)

        # Count split info
        for count in counts:
            infosplit = -count/total*(log2(count/total))

        return infosplit

    def fit(self, x, y, prune=False):
        self.tree = C45._fit(x, self.label, y)
        
    @staticmethod
    def _fit(x, labels, y, default_val=False, prune=False):
        x = C45.normalize_missing_attribute(x, y)
        process_numeric(x, y)

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
            node.set_child(value, ID3._fit(data[0], next_labels, data[1]))

        if prune :
            x_test, y_test, x_train, y_train = C45.train_test_split(x, y)
            ruleset = node.to_rule_list()
            node = ruleset
        
        return node

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

    @staticmethod
    def count_accuracy(y, y_test):
        count = 0
        for i in range(len(y)):
            if y[i] == y_test[i]:
                count += 1

        return (count/len(y)*100)

    @staticmethod
    def predict_from_rule_set(x_test, y_test, ruleset, label, default_value):
        temp = np.array([])
        for x in x_test:
            for rule in ruleset:
                if Rule.is_eq(rule['rules'], x, label):
                    temp = np.append(temp, rule['target'])
                    break
            else:
                temp = np.append(temp, default_value)

        return C45.count_accuracy(temp, y_test)

    @staticmethod
    def prune(x_test, y_test, ruleset, label, default_value):
        for i in range(len(ruleset)):
            subs = powerset(ruleset[i]['rules'])
            max_sub_acc = 0
            max_sub = None
            for sub in subs :
                ruleset[i]['rules'] = sub
                temp_acc = c45.predict_from_rule_set(x_test, y_test, ruleset, label, default_value)
                if max_sub_acc < temp_acc :
                    max_sub_acc = temp_acc
                    max_sub = sub
            ruleset[i]['rules'] = max_sub
        
if __name__ == "__main__":
    data = read_csv('play_tennis.csv')
    label = data[0, 1:-1].tolist()
    training_label = label.copy()
    x = data[1:, 1:-1]
    target = data[1:, -1:].flatten()

    x_test, y_test, x_train, y_train = C45.train_test_split(x, target)
    c45 = C45()
    c45.tree = c45.fit(x_train, label, y_train)
    default_value = mode(target)
    print(default_value)
    print(c45.tree)
    ruleset = c45.tree.to_rule_list()
    print(x_test)
    print(ruleset)
    c45.prune(x_test, y_test, ruleset, label, default_value)
    print(ruleset)
