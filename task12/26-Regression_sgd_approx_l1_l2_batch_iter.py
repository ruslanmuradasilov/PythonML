import numpy as np
import random

from task12.tests import generate_regression_data, test_regression_model


class Regression(object):
    '''Класс для предсказания действительно-значного выхода по входу - вектору из R^n.
    Используется линейная регрессия, то есть если вход X \in R^n, вектор весов W \in R^{n+1},
    то значение регрессии - это [X' 1] * W, то есть y = x1*w1 + x2*w2 + xn*wn + wn+1.
    Обучение - подгонка весов W - будет вестись на парах (x, y).

    Параметры
    ----------
    sgd : объект класса SGD
    trainiterator: объект класса TrainIterator
    n_epoch : количество эпох обучения (default = 1)
    batch_size : размер пакета для шага SGD (default = 16)
    '''

    def __init__(self, sgd, trainiterator, n_epoch=1, batch_size=16):
        self.sgd = sgd
        self.trainiterator = trainiterator
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.W = None

    def fit(self, X, y):
        '''Обучение модели.

        Параметры
        ----------
        X : двумерный массив признаков размера n_samples x n_features
        y : массив/список правильных значений размера n_samples

        Выход
        -------
        Метод обучает веса W
        '''
        random_index = random.randint(0, len(X) - 1)
        self.W = X[random_index]

        for subX, suby in self.trainiterator:
            self.W = self.sgd.step(subX, suby, self.W)

        return self

    def predict(self, X):
        """ Предсказание выходного значения для входных векторов

        Параметры
        ----------
        X : двумерный массив признаков размера n_samples x n_features

        Выход
        -------
        y : Массив размера n_samples
        """
        y_pred = list()
        y_pred = np.dot(X, self.W)

        return y_pred

    def score(self, y_gt, y_pred):
        """Возвращает точность регрессии в виде (1 - u/v),
        где u - суммарный квадрат расхождения y_gt с y_pred,
        v - суммарный квадрат расхождения y_gt с матожиданием y_gt

        Параметры
        ----------
        y_gt : массив/список правильных значений размера n_samples
        y_pred : массив/список предсказанных значений размера n_samples

        Выход
        -------
        accuracy - точность регрессии
        """
        u, v, sum, N = 0, 0, 0, len(y_gt)
        My_gt = y_gt.mean()

        for i in range(N):
            u += (y_gt[i] - y_pred[i]) ** 2
            v += (y_gt[i] - My_gt) ** 2

        if N == 1:
            v = 1

        accuracy = 1 - u / v

        return accuracy


class SGD(object):
    '''Класс для реализации метода стохастического градиентного спуска.

    Параметры
    ----------
    grad : функция вычисления градиента
    alpha : градиентный шаг (default = 1.)

    '''

    def __init__(self, grad, alpha=1.):
        self.grad = grad
        self.alpha = alpha

    def step(self, X, y, W):
        '''Один шаг градиентного спуска.

        Параметры
        ----------
        X : двумерный массив признаков размера n_samples x n_features
        y : массив/список правильных значений размера n_samples
        W : массив весов размера n_weights

        Выход
        -------
        Метод возвращает обновленные веса
        '''
        W_new = W - self.alpha * self.grad.grad(X, y, W) / len(X)

        return W_new


class Grad(object):
    '''Класс для вычисления градиента по весам от функции потерь.

    Параметры
    ----------
    loss : функция потерь
    delta : параметр численного дифференцирования (default = 0.000001)
    '''

    def __init__(self, loss, delta=0.000001):
        self.loss = loss
        self.delta = delta

    def grad(self, X, y, W):
        '''Вычисление градиента.

        Параметры
        ----------
        X : двумерный массив признаков размера n_samples x n_features
        y : массив/список правильных значений размера n_samples
        W : массив весов размера n_weights

        Выход
        -------
        Метод возвращает градиент по весам W в точках X от функции потерь
        '''
        loss_gradient = np.zeros(len(W))
        for i in range(len(loss_gradient)):
            e = np.zeros(len(loss_gradient))
            e[i] = 1
            loss = self.loss.val(X, y, W)
            loss_delta = self.loss.val(X, y, W + self.delta * e)
            loss_gradient[i] = (loss_delta - loss) / self.delta

        return loss_gradient


class Loss(object):
    '''Класс для вычисления функции потерь.

    Параметры
    ----------
    l1_coef : коэффициент l1 регуляризации (default = 0)
    l2_coef : коэффициент l2 регуляризации (default = 0)
    '''

    def __init__(self, l1_coef=0, l2_coef=0):
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def val(self, X, y, W):
        '''Вычисление функции потерь.

        Параметры
        ----------
        X : двумерный массив признаков размера n_samples x n_features
        y : массив/список правильных значений размера n_samples
        W : массив весов размера n_weights

        Выход
        -------
        Метод возвращает значение функции потерь в точках X
        '''
        loss, mse, cost = 0, 0, 0

        for i in range(len(X)):
            mse += (np.dot(W, X[i]) - y[i]) ** 2

        for i in range(len(W)):
            cost += self.l1_coef * abs(W[i]) + self.l2_coef * (W[i]) ** 2

        loss += mse + cost
        return loss


class TrainIterator(object):
    '''Класс итератора для работы с обучающими данными.

    Параметры
    ----------
    X : двумерный массив признаков размера n_samples x n_features
    y : массив/список правильных значений размера n_samples
    n_epoch : количество эпох обучения (default = 1)
    batch_size : размер пакета для шага SGD (default = 16)
    '''

    def __init__(self, X, y, n_epoch=1, batch_size=16):
        self.X = X
        self.y = y
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.counter = -1

    def __iter__(self):
        '''Нужно для использования итератора в цикле for
        Здесь ничего менять не надо
        '''
        return self

    def __next__(self):
        '''Выдача следующего батча.

        Выход
        -------
        Метод возвращает очередной батч как из X, так и из y
        '''
        n_samples = len(self.X)
        n_batches = (self.n_epoch * n_samples) // self.batch_size
        self.counter += 1
        if self.counter < n_batches:
            i = self.counter * self.batch_size - (n_samples * ((self.counter * self.batch_size) // n_samples))
            j = i + batch_size
            k = j - n_samples if j > n_samples else -n_samples
            return np.append(self.X[i:j], self.X[:k], axis=0), np.append(self.y[i:j], self.y[:k])
        else:
            raise StopIteration



# trainX = np.float64([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
# trainY = np.float64([0, 3, 6, 9, 12])  # y = x1 + 2*x2
# testX = np.float64([(6, 6), (7, 7), (8, 8), (9, 9), (10, 10)])
# testY = np.float64([18, 21, 24, 27, 30])

n_epoch = 100
batch_size = 20
alpha = 0.001
delta = 0.00000001
l1_coef = 0.001
l2_coef = 0.001

# trainiterator = TrainIterator(trainX, trainY, n_epoch, batch_size)
# loss = Loss(l1_coef, l2_coef)
# grad = Grad(loss, delta)
# sgd = SGD(grad, alpha)
#
# reg = Regression(sgd, trainiterator, n_epoch, batch_size)
# reg.fit(trainX, trainY)
# y_pred = reg.predict(testX)
# acc = reg.score(testY, y_pred)
# print('Your accuracy is %s' % str(acc))

trainX, trainY, testX, testY = generate_regression_data(Nfeat=100, Mtrain=150, Mtest=1)

trainiterator = TrainIterator(trainX, trainY, n_epoch, batch_size)
loss = Loss(l1_coef, l2_coef)
grad = Grad(loss, delta)
sgd = SGD(grad, alpha)

reg = Regression(sgd, trainiterator, n_epoch, batch_size)

test_regression_model(reg, trainX, trainY, testX, testY)

# # Test MNIST
# (trainX, trainY), (testX, testY) = mnist.load_data()
#
# trainX = trainX.reshape(trainX.shape[0], 784)
# testX = testX.reshape(testX.shape[0], 784)
# trainX = trainX.astype('float32')
# testX = testX.astype('float32')
#
# trainX /= 255
# testX /= 255
#
# trainiterator = TrainIterator(trainX[:1000], trainY[:1000], n_epoch, batch_size)
# loss = Loss(l1_coef, l2_coef)
# grad = Grad(loss, delta)
# sgd = SGD(grad, alpha)
#
# reg = Regression(sgd, trainiterator, n_epoch, batch_size)
# reg.fit(trainX[:1000], trainY[:1000])
# y_pred = reg.predict(testX[:100])
# acc = reg.score(testY[:100], y_pred)
# print('Your accuracy is %s' % str(acc))
