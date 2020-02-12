class Node(object):
    def __init__(self, attr_name: str, attr_values: list, is_leaf: bool):
        self.attr_name = attr_name
        self.is_leaf = is_leaf

        if not is_leaf:
            self.children = dict()

        if not is_leaf and len(attr_values) > 0:
            for attr_value in attr_values:
                self.children[attr_value] = None

    def __str__(self):
        attr_values_str = '' if self.is_leaf else ' '.join(
            self.children.keys())
        return self.attr_name if self.is_leaf else self.attr_name + ': ' + attr_values_str

    def set_child(self, value: str, child: object):
        if value in self.children:
            self.children[value] = child
        else:
            raise KeyError()

    def get_child(self, value: str):
        if value in self.children:
            return self.children[value]
        else:
            raise KeyError()


if __name__ == "__main__":
    node = Node('windy', ['a', 'b'], False)
    print(node)
