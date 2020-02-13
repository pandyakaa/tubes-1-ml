from id3 import ID3
from reader import read_csv
from c45 import C45
from c45_numeric_handler import process_numeric

if __name__ == "__main__":
    data = read_csv('iris.csv')
    label = data[0, 1:-1].tolist()
    x = data[1:, 1:-1]
    target = data[1:, -1:].flatten()

    # ID3
    id3 = ID3()
    id3.tree = ID3.fit(x, label, target)
    print(id3.tree)
    print(id3.tree.to_rule_list())
    print(id3.predict([x[-1, :]], label))

    # C45
    print(label)
    c45 = C45()
    c45.tree = C45.fit(x, label, target)
    print(c45.tree)
