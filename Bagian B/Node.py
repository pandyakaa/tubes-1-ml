class Node(Object):
    def __init__(self, attr_name: String, attr_values: List, is_leaf: Boolean):
        self.attr_name = attr_name
        self.is_leaf = is_leaf

        if not is_leaf:
            self.children = dict()

        if not is_leaf and len(attr_values) > 0:
            for attr_value in attr_values:
                self.children[attr_value] = None

    def set_child(self, value: String, child: Node):
        if value in self.children:
            self.children[value] = child
        else:
            raise KeyError()

    def get_child(self, value: String):
        if value in self.children:
            return self.children[value]
        else:
            raise KeyError()
