from id3 import ID3
from reader import read_csv
from c45 import C45
from c45_numeric_handler import process_numeric

if __name__ == "__main__":
    data = read_csv(
        '/home/gardahadi/Devspace/Semester_6/ML/tubes-1-ml/Bagian B/play_tennis.csv')
    label = data[0, 1:-1].tolist()
    training_label = label.copy()
    x = data[1:, 1:-1]
    target = data[1:, -1:].flatten()

    #ID3
    id3 = ID3()
    id3.tree = ID3.fit(x, training_label, target)
    print(id3.tree)

    #C45
    print(label)
    c45 = C45()
    c45.tree = C45.fit(x, label, target)
    print(c45.tree)
