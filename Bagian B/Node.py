from Rule import Rule


class Node(object):
    def __init__(self, attr_name: str, attr_values: list, is_leaf: bool):
        self.attr_name = attr_name
        self.is_leaf = is_leaf

        if not is_leaf:
            self.children = dict()

        if not is_leaf and len(attr_values) > 0:
            for attr_value in attr_values:
                self.children[attr_value] = None

    def __str__(self, level=0):
        if self.is_leaf:
            return self.attr_name+"\n"
        else:
            ret = self.attr_name+"\n"
            for attr_value, child in self.children.items():
                ret += "|  "*level + "|- " + str(attr_value) + \
                    ": " + child.__str__(level+1)
            return ret

    def __repr__(self):
        return '<Node object>'

    def set_child(self, value: str, child: object):
        if value in self.children:
            self.children[value] = child
        else:
            raise KeyError()

    def get_child(self, value: str):
        print(value)
        for k in self.children:
            print(self.children)
        if value in self.children:
            return self.children[value]
        else:
            raise KeyError()

    def to_rule_list(self):
        # Returns tree as disjunction of conjunction of rule
        # Format: [{'target': leaf values, rules: [Rule]}, {}, {}]
        if self.is_leaf:
            return self.attr_name

        disjunction_of_conjunction = list()
        for value, child in self.children.items():
            next_node_conjunctions = child.to_rule_list()
            current_node_rule = Rule(self.attr_name, value)

            if type(next_node_conjunctions) == list:
                for conjunction in next_node_conjunctions:
                    conjunction['rules'].append(current_node_rule)
                disjunction_of_conjunction.extend(next_node_conjunctions)
            else:
                new_conjunction = {
                    'target': next_node_conjunctions,
                    'rules': [current_node_rule]
                }
                disjunction_of_conjunction.append(new_conjunction)

        return disjunction_of_conjunction


if __name__ == "__main__":
    node = Node('Windy', ['Yes', 'No'], False)
    node.set_child('Yes', Node('Yes', [], True))
    node.set_child('No', Node('No', [], True))

    print(node)
