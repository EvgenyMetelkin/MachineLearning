"""
Classification problem with kNN, LinearClassifier
"""

import argparse
import _pickle as cPickle
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

# def L_i_vectorrized(x, y, W):
#     scores = W.dot(x)
#     margins = np.maximum(0, scores - scores[y] + 1)
#     margins[y] = 0
#     loss_i = np.sum(margins)
#     return loss_i


def read_data(names):
    X = []
    y = []
    for name in names:
        with open(name, "rb") as f:
            data = cPickle.load(f, encoding="bytes")
        X.append(data[b"data"])
        y.append(data[b"labels"].flatten())
    # merge arrays
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y

    # if __name__ == "__main__":
    # parser = argparse.ArgumentParser("Tutorial 2: kNN, linear classifier")
    # parser.add_argument("--train", type=str, nargs="+",
    #     help="Train datasets",
    #     required=True)
    # parser.add_argument("--test", type=str, nargs="+",
    #     help="Test datasets",
    #     required=True)
    # parser.add_argument("--clf", type=str, choices=["kNN", "LC"],
    #     help="Train choose kNN or linear classifier",
    #     required=True)
    # args = parser.parse_args()
    # load data

print("Load data")

# names_d = ["mnist_dataset/data_batch_1", "mnist_dataset/data_batch_2", "mnist_dataset/data_batch_3"]
names_d = ["mnist_dataset/data_batch_1"]
names_t = ["mnist_dataset/test_batch"]
method = ["N"]


train_X, train_y = read_data(names_d)
test_X, test_y = read_data(names_t)

train_X, train_y = train_X, train_y.flatten()
test_X, test_y = test_X, test_y.flatten()

# TODO: load data here

# training
print("Training")
if method == "kNN":
    clf = KNeighborsClassifier()
else:
    clf = LogisticRegression(C=0.1, max_iter=10)
clf.fit(train_X, train_y)

# TODO: train here

# evaluation
print("Testing")
#
#
# W = np.random.uniform(-1, 1, (10, len(train_y)))
# print(W)

# L_i_vectorrized(train_X, )

pred = clf.predict(test_X)
# A = clf.score(test_X, test_y)
tp = np.sum(np.int8(pred == test_y))
A_my = tp/1000
print(A_my)

for l, w in enumerate(clf.coef_):
    plt.figure()
    plt.title(l)
    plt.imshow(w.reshape((28, 28)))
    plt.gray()
plt.show()


# TODO: evaluate quality here