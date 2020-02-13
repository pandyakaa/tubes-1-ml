from id3 import ID3
from reader import read_csv
from c45 import C45
from c45_numeric_handler import process_numeric
from Rule import Rule

if __name__ == "__main__":
    data = read_csv('datasets/play_tennis.csv')
    label = data[0, 1:-1].tolist()
    x = data[1:, 1:-1]
    target = data[1:, -1:].flatten()

    # ID3
    print("=====ID 3=====")
    id3 = ID3()
    id3.tree = ID3.fit(x, label, target)
    print(id3.tree)

    # C45
    print("=====C45=====")
    c45 = C45()
    prune = True
    c45.tree = C45.fit(x, label, target, prune=prune)

    if (prune):
        Rule.printset(c45.tree)
    else:
        print(c45.tree)
