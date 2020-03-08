from id3 import ID3
from reader import read_csv
from c45 import C45
from c45_numeric_handler import process_numeric
from Rule import Rule

if __name__ == "__main__":
    data = read_csv('Bagian B/datasets/iris.csv')
    # print(data)
    label = data[0, 0:-1].tolist()
    x = data[1:, 0:-1]
    target = data[1:, -1:].flatten()
    # print(label)
    # print(x)
    # print(target)

    # ID3
    print("=====ID 3=====")
    id3 = ID3()
    id3.label = label
    id3.fit(x,target)
    # print(id3.tree)

    # C45
    print("=====C45=====")
    c45 = C45()
    c45.label = label
    # print(x)
    # print(target)
    c45.fit(x, target)
    # print(c45.tree)

    print(c45.predict(x[0:1,:]))

    # if (prune):
    #     Rule.printset(c45.tree)
    # else:
    #     print(c45.tree)