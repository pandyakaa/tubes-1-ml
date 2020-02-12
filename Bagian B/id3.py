import numpy as np
import math
from statistics import mode
from Node import Node


# def mode(target): return mode(target)


class ID3(object):
    @staticmethod
    def gain(entropy, attr_values, target):
        # entropy - sum_i(frac_i * entropy_i)
        total = 0
        sum_next_entropy = 0
        attr_unique_targets = dict()

        for idx, attr in enumerate(attr_values):
            if attr in attr_unique_targets:
                attr_unique_targets[attr]['count'] += 1
                attr_unique_targets[attr]['targets'].append(target[idx])
            else:
                attr_unique_target = {'count': 1, 'targets': [target[idx]]}
                attr_unique_targets[attr] = attr_unique_target
            total += 1

        for attr, attr_unique_target in attr_unique_targets.items():
            sum_next_entropy += attr_unique_target['count'] / total * \
                ID3.count_entropy(attr_unique_target['targets'])

        return entropy - sum_next_entropy

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

        for attr, val in target_dictionary.items():
            entropy += -val/total*(math.log2(val/total))

        return entropy

    @staticmethod
    def fit(x, labels, y, default_val=False):
        gain = list()

        if default_val == False:
            default_val = mode(y)

        if np.all(y == y[0, ]):
            return Node(str(y[0]), [], True)

        if x.shape[1] == 0:
            return Node(str(default_val), [], True)

        entropy = ID3.count_entropy(y)

        for idx, attr in enumerate(x.T):
            gain.append(ID3.gain(entropy, attr, y))

        idx_max = np.argmax(gain)
        attr_values = np.unique(x.T[idx_max])

        node = Node(labels[idx_max], attr_values, False)

        labels.pop(idx_max)

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

        for value, data in data_per_values.items():
            node.set_child(value, ID3.fit(data[0], labels, data[1]))

        return node


if __name__ == "__main__":
    ex_x = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
    ex_target = np.array([1, 2, 3])
    ex_label = ['a', 'b', 'c']

    root_node = ID3.fit(ex_x, ex_label, ex_target)
    print(root_node)
