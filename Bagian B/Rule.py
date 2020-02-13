import numpy as np


class Rule(object):
    def __init__(self, attr_name: str, value: object):
        self.attr_name = attr_name
        self.value = value

    def __str__(self):
        return '(' + self.attr_name + ' = ' + self.value + ')'

    def __repr__(self):
        return '(' + self.attr_name + ' = ' + self.value + ')'

    @staticmethod
    def satisfies(rule: object, values: np.array, label: list):
        idx = label.index(rule.attr_name)
        rule_value = rule.value
        compared_value = values[idx]

        return rule_value == compared_value

    @staticmethod
    def is_eq(rules: list, values: np.array, label: list):
        satisfies_list = list()
        for rule in rules:
            satisfies_list.append(Rule.satisfies(rule, values, label))

        return all(satisfies_list)
