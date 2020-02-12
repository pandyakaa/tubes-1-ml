import numpy as np
import math
import Node


def mode(target): return max(set(target), key=target.count)


class ID3(object):
    def __init__(self, labels):
        self.root = None
        self.labels = labels

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

        for attr, attr_unique_target in attr_unique_targets:
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
                total += 1
            else:
                target_dictionary[val] = 1
                total += 1

        for attr, val in target_attributes.items():
            entropy += -val/total*(math.log2(-val/total))

        return entropy

    @staticmethod
    def fit(x, y, default_val=False):
        gain = []

        if default_val == False:
            default_val = mode(y)
        for target in y:
            if (all(element == target for element in y)):
                return Node(target, [], True)

        if a.shape[1] == 0:
            return Node(default_val, [], True)

        entropy = ID3.count_entropy(y)

        for idx, attr in x.transpose():
            gain[idx] = gain(entropy, attr, y)
