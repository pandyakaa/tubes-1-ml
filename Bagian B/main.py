from id3 import ID3
from reader import read_csv

if __name__ == "__main__":
    data = read_csv('play_tennis.csv')
    label = data[0, 1:-1].tolist()
    x = data[1:, 1:-1]
    target = data[1:, -1:].flatten()

    tree = ID3.fit(x, label, target)
    print(tree)
