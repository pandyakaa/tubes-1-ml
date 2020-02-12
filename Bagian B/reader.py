# Importing some utils
import numpy as np
import csv


def read_csv(filename):

    with open(filename) as f:
        temp_mat = []
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            temp_mat.append(row)

        np_mat = np.array(temp_mat)

    return np_mat


if __name__ == "__main__":
    filename = 'play_tennis.csv'
    np_matrix = read_csv(filename)

    print(np_matrix)
