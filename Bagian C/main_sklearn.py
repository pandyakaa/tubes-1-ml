from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X, y = shuffle(X, y)

    clf = MLPClassifier(solver='sgd', activation='logistic',
                        hidden_layer_sizes=(4, 3), batch_size=10,
                        learning_rate_init=0.01, max_iter=1000, random_state=1)

    clf.fit(X[:100], y[:100])

    print('The actual label:')
    print(y[100:])
    print()

    print('The machine verdict:')
    print(clf.predict(X[100:]))
