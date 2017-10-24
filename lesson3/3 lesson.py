import numpy as np
import _pickle as cPickle
from math import log, fabs
import time

#from sklearn.neighbors import

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


def loss_func_SVM(W, X, y):
    Wx_r = W.dot(X)  # в массив заносится элементы W поэлементно умноженные на вектор X
    # print(Wx_r)
    for i in range(Wx_r.shape[1]):  # по столбцам
        Wx_r[:, i] = Wx_r[:, i] - Wx_r[y[i] - 1, i] + 1
    Wx_r = np.maximum(0, Wx_r)  #
    L = (np.sum(Wx_r) - Wx_r.shape[1])/Wx_r.shape[1]
    # for j in range(Wx_r.shape[0]):
    #     if Wx_r[j, i] < 0:
    #         Wx_r[j, i] = 0
    return L


def loss_func_Softmax(W, X, y):
    Wx_r = W.dot(X)
    Wx_r = np.exp(Wx_r)
    cl = np.zeros(y.shape[0])  # по кол-ву объектов
    for i in range(Wx_r.shape[1]):
        Wx_r[:, i] = Wx_r[:, i]/np.sum(Wx_r[:, i])
        cl[i] = -log(Wx_r[y[i]-1, i])
        L = np.sum(cl[i])/Wx_r.shape[1]
        return L


def gradL_num(method, step, W, X, y):
    W_new = W.copy()  # создаем копию вектора весов
    gradL = np.zeros((W.shape[0], W.shape[1]))  # создаем нулевую матрицу размерности W
    for i in range(W_new.shape[0]):  # идем по всем строкам
        for j in range(W_new.shape[1]):  # идем по всем элементам в строке
            W_new[i, j] = W[i, j] + step  # делаем смещение
            if method == 1:  # выбор функции потерь
                gradL[i, j] = (loss_func_SVM(W_new, X, y) - loss_func_SVM(W, X, y)) / step
            else:
                gradL[i, j] = (loss_func_Softmax(W_new, X, y) - loss_func_Softmax(W, X, y)) / step
            W_new[i, j] = W_new[i, j] - step
    return gradL


def grad_down(W, X, y, lamb_grad, err, method, step, max_iter):
    W_new = W.copy()
    stop = 1000;
    iter = 0;
    while stop >= err or iter > max_iter:
        iter = iter + 1
        W = W_new.copy()
        if method == 1:
            W_new = W - lamb_grad * gradL_num(1, step, W, X, y)
            stop = fabs(loss_func_SVM(W_new, X, y) - loss_func_SVM(W, X, y))
        else:
            W_new = W - lamb_grad * gradL_num(2, step, W, X, y)
            stop = fabs(loss_func_Softmax(W_new, X, y) - loss_func_Softmax(W, X, y))
        print(iter)
    return W_new, iter


method = 1  # loss_func_SVM, 2 иначе
step = 0.0001  # численная производная

num_class = 10
lamb_grad = 0.001  # шаг спуска
err = 0.001  # останова

data_train = ["mnist_dataset/data_batch_1"]  # mnist
data_test = ["mnist_dataset/test_batch"]

X_train, y_train = read_data(data_train)  # данные из обучающей выборки
X_train, y_train = X_train[:1000], y_train[:1000]  # обрезаем до 1000 элементов
X_test, y_test = read_data(data_test)

feature = X_train.shape[1]  # запоминаем количество элементов в строке
W = np.random.randn(num_class, feature)  # создаем массив весов
X_train = X_train.transpose()  # транспонируем матрицу

t1 = time.time()
print(gradL_num(method, step, W, X_train, y_train))
print(time.time() - t1)  # выводим время работы
#
# W, iter = grad_down(W, X_train, y_train, lamb_grad, err, method, step, 10000) #
# scopes = W.dot(X_train) #
# y_predict = np.argmax(scopes, axis=0) + 1 #
# accur = np.mean(y_predict == y_train) #
# print('Iterations', iter) #