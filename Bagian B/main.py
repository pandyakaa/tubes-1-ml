from id3 import ID3
from reader import read_csv

if __name__ == "__main__":
    data = read_csv(
        '/home/wirasuta/Documents/akademik/ml/tubes-1-ml/Bagian B/play_tennis.csv')
    label = data[0, 1:-1].tolist()
    training_label = label.copy()
    x = data[1:, 1:-1]
    target = data[1:, -1:].flatten()

    id3 = ID3()
    id3.tree = ID3.fit(x, training_label, target)
    print(id3.tree)
    print(id3.tree.to_rule_list())
    print(id3.predict([x[-1, :]], label))
